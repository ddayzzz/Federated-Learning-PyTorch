#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from dice_loss import dice_coeff


class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        """
        给定原始的数据集和对应的 index, 产生在 index 中存在的子数据集
        :param dataset:
        :param idxs:
        """
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def  __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        将数据分开
        :param dataset: 数据集对象
        :param idxs: 索引, list
        :return: train, valid, test
        """
        # train, validation, test 的比例 0.8, 0.1, 0.1
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val) / 10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test) / 10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        model.train()
        epoch_loss = []

        # 设置优化器
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    # [已经处理的样本数/总共拥有的训练集样本数 比例%]
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


class BRATS2018LocalUpdate(object):

    def  __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.n_val = self.train_val(dataset, list(idxs), train_rate=args.train_rate)
        self.device = 'cuda' if args.gpu else 'cpu'
        # 网络输出的 logits
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)

    def train_val(self, dataset, idxs, train_rate=0.8):
        """
        将数据分开, 分为 train 和 valid 的 loader
        :param dataset:
        :param idxs:
        :return:
        """
        # split indexes for train, validation (train_rate, 1-train_rate)
        idxs_train = idxs[:int(train_rate*len(idxs))]
        idxs_val = idxs[int(train_rate*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True, num_workers=self.args.num_workers)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=self.args.local_bs, shuffle=False, num_workers=self.args.num_workers)
        return trainloader, validloader, len(idxs_val)

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                logits = model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        model.eval()
        tot = 0

        for batch_idx, (images, labels) in enumerate(self.validloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            for true_mask, pred in zip(labels, outputs):
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()
        return tot / self.n_val


# Elastic transformation of an image in Python. from https://gist.github.com/fmder/e28813c1e8721830ff9c
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import random
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as F


class ElasticTransform:

    def __init__(self, alpha, sigma, random_state=None):
        self.alpha, self.sigma, self.random_state = alpha, sigma, random_state

    def _trans(self, img, rs):
        shape = img.shape
        dx = gaussian_filter((rs), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((rs), self.sigma, mode="constant", cval=0) * self.alpha
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        return map_coordinates(img, indices, order=1).reshape(shape)

    def __call__(self, img, label):
        """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        """
        # https://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/prepro.html#elastic_transform_multi
        if self.random_state is None:
            self.random_state = np.random.RandomState(None)
        shape = img.shape
        rs = self.random_state.rand(*shape) * 2 - 1
        img = self._trans(img, rs)
        label = self._trans(label, rs)
        return img, label


class DatasetSplitForDataAug(DatasetSplit):

    elastic = ElasticTransform(alpha=720, sigma=24)

    def __init__(self, dataset, idxs, trans=False, p=0.5):
        """
        给定原始的数据集和对应的 index, 产生在 index 中存在的子数据集
        :param dataset:
        :param idxs:
        """
        super(DatasetSplitForDataAug, self).__init__(dataset, idxs)
        self.trans = trans
        self.p = p

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]  # image:[H,W]
        if self.trans:
            # 需要随机地对于 image 和 label 进行变换
            # 弹性伸缩, 这个自带了 random 性质
            image, label = self.elastic(image, label)
            # numpy->PIL.Image
            image = F.to_pil_image(image, mode='F')  # PIL
            label = F.to_pil_image(label, mode='F')
            # 水平翻转
            if random.random() < self.p:
                image, label = F.hflip(image), F.hflip(label)
            # 旋转
            if random.random() < self.p:
                angle = random.uniform(-20.0, 20.0)
                image, label = F.rotate(image, angle=angle, resample=False, center=None, expand=None), F.rotate(label, angle=angle, resample=False, center=None, expand=None)
            # 可以添加更多的变换
            # PIL.Image->tensor
        return F.to_tensor(image), F.to_tensor(label)


class BRATS2018DataAugmentationLocalUpdate(BRATS2018LocalUpdate):

    def  __init__(self, args, dataset, idxs, logger):

        self.p = args.augmentation_rate  # 进行数据增广的比例
        super(BRATS2018DataAugmentationLocalUpdate, self).__init__(args, dataset, idxs, logger)


    def train_val(self, dataset, idxs, train_rate=0.8):
        """
        将数据分开, 分为 train 和 valid 的 loader
        :param dataset:
        :param idxs:
        :return:
        """
        # split indexes for train, validation (train_rate, 1-train_rate)
        idxs_train = idxs[:int(train_rate * len(idxs))]
        idxs_val = idxs[int(train_rate * len(idxs)):]

        trainloader = DataLoader(DatasetSplitForDataAug(dataset, idxs_train, trans=True, p=self.p),
                                 batch_size=self.args.local_bs, shuffle=True, num_workers=self.args.num_workers)
        validloader = DataLoader(DatasetSplitForDataAug(dataset, idxs_val, trans=False), batch_size=self.args.local_bs, shuffle=False,
                                 num_workers=self.args.num_workers)
        #
        self.mim_frame_seq = self.args.minimum_trained_image_size if self.args.minimum_trained_image_size > 0 else len(idxs_train)
        return trainloader, validloader, len(idxs_val)

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            frame_seq = 0
            while frame_seq < self.mim_frame_seq:
                for batch_idx, (images, labels) in enumerate(self.trainloader):
                    images, labels = images.to(self.device), labels.to(self.device)

                    model.zero_grad()
                    logits = model(images)
                    loss = self.criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                    if self.args.verbose and (batch_idx % 10 == 0):
                        print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            global_round, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                                                100. * batch_idx / len(self.trainloader), loss.item()))
                    self.logger.add_scalar('loss', loss.item())
                    batch_loss.append(loss.item())
                    frame_seq += len(images)
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss


def brats2018_test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """
    raise NotImplementedError
    model.eval()
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            mask_pred = net(imgs)

            for true_mask, pred in zip(true_masks, mask_pred):
                pred = (pred > 0.5).float()
                if net.n_classes > 1:
                    tot += F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0)).item()
                else:
                    tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()
            pbar.update(imgs.shape[0])

    return tot / n_val

if __name__ == '__main__':
    from data.brats2018.dataset import BRATS2018Dataset
    training_dir = r'D:\Projects\pytorch_brats2018_fl\data\brats2018\training'
    ds = BRATS2018Dataset(training_dir=training_dir, img_dim=128, max_size=150)
    ori_image, ori_label = ds[90]  # todo 不需要增加的 channel=1 的维度, 这个只是临时测试
    aug1 = DatasetSplitForDataAug(ds, idxs=[90], trans=True, p=1.0)
    aug2 = DatasetSplitForDataAug(ds, idxs=[90], trans=False)
    # 小数据集的第一个 slice
    transed_image, transed_label = aug1[0]
    no_transed_image, no_transed_label = aug2[0]
    assert np.sum(ori_image - no_transed_image.numpy()) < 1e-6
    assert np.sum(ori_label - no_transed_label.numpy()) < 1e-6
    # 显示变换后的图像
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots(2,2)
    ax[0][0].imshow(ori_image, cmap=plt.cm.gray)
    ax[0][1].imshow(ori_label, cmap=plt.cm.gray)
    ax[1][0].imshow(transed_image[0].numpy(), cmap=plt.cm.gray)
    ax[1][1].imshow(transed_label[0].numpy(), cmap=plt.cm.gray)
    plt.show()