#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# 执行验证过程

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from options import brats2018_test_args_parser
from unet import UNet
from utils import get_dataset
from dice_loss import dice_coeff



if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')

    args = brats2018_test_args_parser()
    # exp_details(args)

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
    # 加载模型
    global_model = torch.load(args.model_path, map_location=device)
    global_model.eval()
    print(global_model)
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=8)
    # 输出数据转换
    dcs = []
    for i, (images, ground_truth) in enumerate(train_loader):
        images, ground_truth = images.to(device), ground_truth.to(device).squeeze(1).squeeze(0)  # ground truth 去掉 channel 维度
        logits = global_model(images)
        probs = torch.sigmoid(logits).squeeze(1).squeeze(0)  # 去掉 channel 和 batch 的维度
        mask = (probs > 0.5).float()
        dc = dice_coeff(ground_truth, mask).cpu().item()
        print(f'{i}, dc: {dc}')
        dcs.append(dc)
    dc_avg = np.average(dcs)
    dc_min = np.min(dcs)
    dc_max = np.max(dcs)
    print(f'DC Average: {dc_avg}, Min: {dc_min}, Max: {dc_max}')