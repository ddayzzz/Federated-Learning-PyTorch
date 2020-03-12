#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# 执行图像分割的 FL 代码

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from options import brats2018_data_aug_args_parser
from update import BRATS2018DataAugmentationLocalUpdate
from unet import UNet
from utils import get_dataset, average_weights, exp_details


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    os.makedirs('../brats2018_data_aug_logs', exist_ok=True)
    logger = SummaryWriter('../brats2018_data_aug_logs')


    args = brats2018_data_aug_args_parser()
    exp_details(args)

    device = torch.device(args.gpu) if args.gpu is not None else 'cpu'
    if args.gpu:
        torch.cuda.set_device(device)

    # load dataset and user groups
    train_dataset, _, user_groups = get_dataset(args)

    # 使用 UNET
    if args.model == 'unet':
        global_model = UNet(n_channels=1, n_classes=1, bilinear=True)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, valid_dc = [], []
    print_every = 2
    save_every = args.save_per_epoch
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()  # 设置模型处于 training 模式
        m = max(int(args.frac * args.num_users), 1)  # 选择 train 的 client
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            #　每一次 epoch 就会创建一个 Update 对象, 表示 local 的模型
            local_model = BRATS2018DataAugmentationLocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            # 记录 client 得到的权重和 loss
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_dice = []
        global_model.eval()
        for c in range(args.num_users):
            # 这里记录的 Dice coefficient
            local_model = BRATS2018DataAugmentationLocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[c], logger=logger)
            dc = local_model.inference(model=global_model)
            list_dice.append(dc)
        valid_dc.append(sum(list_dice) / len(list_dice))  # DC 的平均

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print('Avg Validation Dice Coefficient: {:.2f}\n'.format(valid_dc[-1]))

        if (epoch + 1) % save_every == 0:
            # 保存模型
            model_filename = '../save/models/{}_{}_{}_C[{}]_balanced[{}]_E[{}]_B[{}]_train_rate[{}].pkl'. \
                format(args.dataset, args.model, epoch, args.frac, args.balanced,
                       args.local_ep, args.local_bs, args.train_rate)
            torch.save(global_model, model_filename)
            # 保存 loss 和 准确率的模型参数
            file_name = '../save/objects/{}_{}_{}_C[{}]_balanced[{}]_E[{}]_B[{}]_train_rate[{}].pkl'. \
                format(args.dataset, args.model, epoch, args.frac, args.balanced,
                       args.local_ep, args.local_bs, args.train_rate)
            with open(file_name, 'wb') as f:
                pickle.dump([train_loss, valid_dc], f)

    # Test inference after completion of training
    # test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Validation Dice Coefficient: {:.2f}".format(valid_dc[-1]))
    # print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_balanced[{}]_E[{}]_B[{}]_train_rate[{}]_final.pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.balanced,
               args.local_ep, args.local_bs, args.train_rate)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, valid_dc], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))