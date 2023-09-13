import argparse
import torch
import os
import numpy as np

import argparse


def add_args(parser):
    parser.add_argument('--logdir', type=str,
                        default='./log_conv_all_half_eikonal_1e-4_amsgrad_200_epoch_cos_1e-6_1500_2/',
                        help='log directory')
    parser.add_argument('--model_name', type=str, default='model', help='trained model name')
    parser.add_argument('--seed', type=int, default=3627473, help='random seed')
    parser.add_argument('--n_points', type=int, default=10000, help='number of points in each point cloud')
    parser.add_argument('--grid_res', type=int, default=128, help='uniform grid resolution')
    parser.add_argument('--with_normal', type=bool, default=False)

    # training parameters
    parser.add_argument('--dataset_path', type=str, default='/home/bearprin/DualOctreeGNN/data/dfaust/dataset',
                        help='path to dataset folder')
    parser.add_argument('--test_part_i', type=int, default=None)
    parser.add_argument('--train_split_path', type=str,
                        default='/home/bearprin/DualOctreeGNN/data/dfaust/filelist/train.txt',
                        help='path to split file')
    parser.add_argument('--test_split_path', type=str,
                        default='/home/bearprin/DualOctreeGNN/data/dfaust/filelist/test.txt',
                        help='path to split file')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--test_lr', type=float, default=1e-5, help='initial learning rate')
    parser.add_argument('--test_iter', type=int, default=3000, help='test_optim_iter')
    parser.add_argument('--grad_clip_norm', type=float, default=10.0, help='Value to clip gradients to')
    parser.add_argument('--batch_size', type=int, default=1, help='number of samples in a minibatch')

    # Network architecture and loss
    parser.add_argument('--init_type', type=str, default='siren',
                        help='initialization type siren | geometric_sine | geometric_relu | mfgi')
    parser.add_argument('--decoder_hidden_dim', type=int, default=256, help='length of decoder hidden dim')
    parser.add_argument('--decoder_n_hidden_layers', type=int, default=5, help='number of decoder hidden layers')
    parser.add_argument('--nl', type=str, default='sine', help='type of non linearity sine | relu')
    parser.add_argument('--sphere_init_params', nargs='+', type=float, default=[1.6, 0.1],
                        help='radius and scaling')
    parser.add_argument('--loss_type', type=str, default='siren_wo_n_w_morse')
    parser.add_argument('--test_loss_type', type=str, default='siren_wo_n')
    parser.add_argument('--decay_params', nargs='+', type=float, default=[3, 0.1, 3, 0.2, 0.001, 0],
                        help='epoch number to evaluate')
    parser.add_argument('--morse_type', type=str, default='l1', help='divergence term norm l1 | l2')
    parser.add_argument('--morse_decay', type=str, default='linear',
                        help='divergence term importance decay none | step | linear')
    parser.add_argument('--loss_weights', nargs='+', type=float, default=[7e3, 6e2, 1e2, 5e1, 1e2, 3, 1],
                        help='loss terms weights sdf | inter | normal | eikonal | div | morse')
    parser.add_argument('--test_loss_weights', nargs='+', type=float, default=[7e3, 6e2, 1e2, 5e1, 1e2, 0, 0],
                        help='loss terms weights sdf | inter | normal | eikonal | div | morse')
    parser.add_argument('--bidirectional_morse', action='store_true')
    parser.add_argument('--morse_near', default=True)
    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    return args
