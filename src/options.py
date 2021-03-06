#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def _basic_args():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                            use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                            of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                            mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                            strided convolutions")
    parser.add_argument('--sgd_momentum', type=float, default=0.9, help='SGD optimizer momentum')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                            of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                            of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                            to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                            of optimizer")
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                            non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--save_per_epoch', type=int, default=10, help='每多少的 epoch 保存一次模型')
    return parser

def args_parser():
    parser = _basic_args()
    parser.add_argument('--iid', type=int, default=1, help='Default set to IID. Set to 0 for non-IID.')
    args = parser.parse_args()
    return args


def brats2018_args_parser():
    parser = _basic_args()
    parser.add_argument('--data_dir', type=str, default='./data/brats2018', help='设置处理后的 BRATS2018 数据集的目录')
    parser.add_argument('--balanced', type=int, default=1, help='设置是否 balanced')
    parser.add_argument('--num_workers', type=int, default=0, help='设置数据加载的进程数量(默认0即加载使用主进程)')
    parser.add_argument('--train_rate', type=float, default=0.8, help='设置训练集的比例')
    args = parser.parse_args()
    return args


def brats2018_test_args_parser():
    parser = _basic_args()
    parser.add_argument('--data_dir', type=str, default='./data/brats2018', help='设置处理后的 BRATS2018 数据集的目录')
    parser.add_argument('--model_path', type=str, help='设置模型的路径')
    parser.add_argument('--train_rate', type=float, default=0.8, help='设置训练集的比例')
    args = parser.parse_args()
    return args


def brats2018_data_aug_args_parser():
    parser = _basic_args()
    parser.add_argument('--augmentation_rate', type=float, default=0.5, help='设置进行增广的概率')
    parser.add_argument('--data_dir', type=str, default='./data/brats2018', help='设置处理后的 BRATS2018 数据集的目录')
    parser.add_argument('--balanced', type=int, default=1, help='设置是否 balanced')
    parser.add_argument('--num_workers', type=int, default=0, help='设置数据加载的进程数量(默认0即加载使用主进程)')
    parser.add_argument('--train_rate', type=float, default=0.8, help='设置训练集的比例')
    parser.add_argument('--minimum_trained_image_size', type=int, default=-1, help='设置每个 local epoch 需要训练的最小的样本数量, 不足的将会生成')
    args = parser.parse_args()
    return args