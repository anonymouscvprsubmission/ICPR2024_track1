import os
import math
import argparse
import socket
from datetime import datetime
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils.dataset import TrainSetLoader, TestSetLoader
from utils.metric import SigmoidMetric, SamplewiseSigmoidMetric, PD_FA, ROCMetric, mIoU
from utils.engine import train_one_epoch, evaluate
from utils.loss import SoftLoULoss1 as SoftLoULoss
import torch.nn.functional as F



from config import load_config
from argparse import ArgumentParser
import torch.nn as nn
import os.path as ops
import time
import numpy as np

torch.cuda.manual_seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0, 2, 3, 4, 5, 6, 7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

resume = False


resume_dir = '/mlx_devbox/users/yanghuoren/playground/IR/PBT/runs/Jun30_21-31-24_track1_train_1_2_PBT/checkpoint/Best_nIoU_Epoch-160_IoU-0.0228_nIoU-0.1313.pth.tar'


def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of HCT model')
    #parser.add_argument('--model', type=str, default='hct_base_patch32_512', help='model_name:')
    parser.add_argument('--dataset', type=str, default='dataset',
                        help='dataset:IRSTD-1k; NUDT-SIRST; Flir; ')
    parser.add_argument('--suffix', type=str, default='.png')
    #
    # Training parameters
    #
    parser.add_argument('--aug', type=float, default=0.)
    parser.add_argument('--workers', type=int, default=8, metavar='N', help='dataloader threads')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size for training')
    parser.add_argument('--epochs', type=int, default=1500, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adagrad', help='Adam, Adagrad')
    parser.add_argument('--scheduler', default='CosineAnnealingLR', choices=['CosineAnnealingLR', 'ReduceLROnPlateau'])
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='9e-3learning rate (default: 0.1)')
    parser.add_argument('--min_lr', default=1e-2, type=float, help='3e-3minimum learning rate')
    #
    # Net parameters
    #

    #
    # Dataset parameters
    #

    args = parser.parse_args()
    return args
def save_ckpt(state, save_path, filename):
    torch.save(state, filename)


def load_dataset (root, dataset, split_method):
    train_txt = root + '/' + dataset + '/' + split_method + '/' + 'train.txt'
    test_txt  = root + '/' + dataset + '/' + split_method + '/' + 'train_repeat.txt'
    train_img_ids = []
    val_img_ids = []
    with open(train_txt, "r") as f:
        line = f.readline()
        while line:
            train_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            val_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    return train_img_ids,val_img_ids,test_txt


##################################################

class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


def _upsample_like(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')

    return src


### RSU-7 ###
class RSU7(nn.Module):  # UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        self.latents = []
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)

        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-6 ###
class RSU6(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()
        self.latents = []
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-5 ###
class RSU5(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
        self.latents = []

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4 ###
class RSU4(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
        self.latents = []

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4F ###
class RSU4F(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
        self.latents = []

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin

class AsymBiChaFuseReduce(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels=64, r=4):
        super(AsymBiChaFuseReduce, self).__init__()
        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )##512

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.out_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )#512

        ##############add spatial attention ###Cross UtU############
        self.bottomup = nn.Sequential(
            nn.Conv2d(self.low_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),
            # nn.Sigmoid(),

            SpatialAttention(kernel_size=3),
            # nn.Conv2d(self.bottleneck_channels, 2, 3, 1, 0),
            # nn.Conv2d(2, 1, 1, 1, 0),
            #nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid()
        )

        self.post = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )#512

    def forward(self, xh, xl):
        xh = self.feature_high(xh)

        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl * topdown_wei)
        xs1 = 2 * xl * topdown_wei  #1
        out1 = self.post(xs1)

        xs2 = 2 * xh * bottomup_wei    #1
        out2 = self.post(xs2)
        return out1,out2

        ##############################
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x

##### UIU-net ####
class UIUNET(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(UIUNET, self).__init__()

        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

        # self.fuse6 = self._fuse_layer(512, 512, 512, fuse_mode='AsymBi')
        self.fuse5 = self._fuse_layer(512, 512, 512, fuse_mode='AsymBi')
        self.fuse4 = self._fuse_layer(512, 512, 512, fuse_mode='AsymBi')
        self.fuse3 = self._fuse_layer(256, 256, 256, fuse_mode='AsymBi')
        self.fuse2 = self._fuse_layer(128, 128, 128, fuse_mode='AsymBi')

        self.latents = []

    def _fuse_layer(self, in_high_channels, in_low_channels, out_channels, fuse_mode='AsymBi'):  # fuse_mode='AsymBi'
        # assert fuse_mode in ['BiLocal', 'AsymBi', 'BiGlobal']
        # if fuse_mode == 'BiLocal':
        #     fuse_layer = BiLocalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        # el
        if fuse_mode == 'AsymBi':
            fuse_layer = AsymBiChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        # elif fuse_mode == 'BiGlobal':
        #     fuse_layer = BiGlobalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        else:
            NameError
        return fuse_layer

    def forward(self, x, return_attn=None):

        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------

        fusec51, fusec52 = self.fuse5(hx6up, hx5)
        hx5d = self.stage5d(torch.cat((fusec51, fusec52), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        fusec41, fusec42 = self.fuse4(hx5dup, hx4)
        hx4d = self.stage4d(torch.cat((fusec41, fusec42), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        fusec31, fusec32 = self.fuse3(hx4dup, hx3)
        hx3d = self.stage3d(torch.cat((fusec31, fusec32), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        fusec21, fusec22 = self.fuse2(hx3dup, hx2)
        hx2d = self.stage2d(torch.cat((fusec21, fusec22), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d22 = self.side2(hx2d)
        d2 = _upsample_like(d22, d1)

        d32 = self.side3(hx3d)
        d3 = _upsample_like(d32, d1)

        d42 = self.side4(hx4d)
        d4 = _upsample_like(d42, d1)

        d52 = self.side5(hx5d)
        d5 = _upsample_like(d52, d1)

        d62 = self.side6(hx6)
        d6 = _upsample_like(d62, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        # return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
        # return d0, d1, d2, d3, d4, d5, d6
        return [d6, d5, d4, d3, d2, d1, d0]


# def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
#     bce_loss = nn.BCELoss(size_average=True)
#     loss0 = bce_loss(torch.sigmoid(d0), labels_v)
#     loss1 = bce_loss(torch.sigmoid(d1), labels_v)
#     loss2 = bce_loss(torch.sigmoid(d2), labels_v)
#     loss3 = bce_loss(torch.sigmoid(d3), labels_v)
#     loss4 = bce_loss(torch.sigmoid(d4), labels_v)
#     loss5 = bce_loss(torch.sigmoid(d5), labels_v)
#     loss6 = bce_loss(torch.sigmoid(d6), labels_v)
#
#     loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
#     # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
#     #     loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(),
#     #     loss5.data.item(), loss6.data.item()))
#
#     return loss0, loss

def main(args):

    dataset = args.dataset
    cfg = load_config()
    root, split_method, size, batch, aug = cfg['dataset']['root'], cfg['dataset'][dataset]['split_method'], \
                                      cfg['dataset'][dataset]['size'], cfg['dataset'][dataset]['batch'], cfg['dataset'][dataset]['aug']
    args.img_size = size
    args.batch_size = batch
    args.aug = aug
    args.model = cfg['dataset'][dataset]['model']
    train_img_ids, train_repeat_img_ids, test_txt = load_dataset(root, dataset, split_method)

    # train_img_ids = train_img_ids[:64]
    # val_img_ids = val_img_ids[::50]

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', current_time + '_' + dataset + "_" + split_method + "_" + args.model)
    tb_writer = SummaryWriter(log_dir=log_dir)
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.26091782370802136, 0.26091782370802136, 0.26091782370802136], [0.017222260313435774
, 0.017222260313435774, 0.017222260313435774])
    ])

    dataset_dir = root + '/' + dataset
    args.use_prior = True
    print('use_prior_loss: ', args.use_prior)
    trainset = TrainSetLoader(dataset_dir, img_id=train_img_ids, base_size=size, crop_size=size,
                              transform=input_transform, suffix=args.suffix, aug=args.aug, useprior=True)
    print(len(trainset))

    trainrepeatset = TrainSetLoader(dataset_dir, img_id=train_repeat_img_ids, base_size=size, crop_size=size,
                            transform=input_transform, suffix=args.suffix, aug=args.aug, useprior=True)
    train_data = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, drop_last=True, pin_memory=True)
    train_repeat_data = DataLoader(dataset=trainrepeatset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, drop_last=True, pin_memory=True)
    val_data = DataLoader(dataset=trainset, batch_size=1, num_workers=1,
                          drop_last=False)

    model = UIUNET(3, 1)

    print('img size: ', size)
    print('dataset: ', dataset)
    print('# model_restoration parameters: %.2f M' % (sum(param.numel() for param in model.parameters()) / 1e6))
    print('device_count: ', torch.cuda.device_count())
    if torch.cuda.device_count() > 1:


        print("Let's use ", torch.cuda.device_count(), " GPUs!")
        # model = nn.DataParallel(model, device_ids=[0, 1])
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])

    model = model.to(device)

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=.9)
    else:
        raise
    if args.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=min(args.epochs,2000), eta_min=args.min_lr)
    else:
        raise
    restart = 0
    if resume == True:
        ckpt = torch.load(resume_dir)
        print(ckpt['mean_IOU'])
        model.load_state_dict(ckpt['state_dict'], strict=True)

        restart = ckpt['epoch']

        optimizer.load_state_dict(ckpt['optimizer'])
        # scheduler.load_state_dict(ckpt["scheduler"])
        print('resuming')




    folder_name = '%s_%s_%s' % (time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                                args.dataset, args.model)

    save_folder = log_dir
    save_pkl = ''
    if not ops.exists('result'):
        os.mkdir('result')
    if not ops.exists(save_folder):
        os.mkdir(save_folder)
    # if not ops.exists(save_pkl):
    #     os.mkdir(save_pkl)
    tb_writer.add_text(folder_name, 'Args:%s, ' % args)

    loss_func = SoftLoULoss(a=0.).to(device)
    last_name_miou = ' '
    last_name_niou = ' '
    if not os.path.exists(f'./result_pth'):
        os.mkdir(f'./result_pth')
    best_iou = 0
    best_iou_repeat = 0

    iou_metric = SigmoidMetric()
    niou_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    roc = ROCMetric(1, 10)
    pdfa = PD_FA(1, 10)
    miou = mIoU(1)
    for epoch in range(restart+1, args.epochs):
        if epoch % 300 <= 250:
            train_loss, current_lr, loss1, loss2 = train_one_epoch(model, optimizer, train_data, device, epoch, loss_func,)
            if epoch % 10 == 0:
                val_loss, iou_, niou_, miou_, ture_positive_rate, false_positive_rate, recall, precision, pd, fa = \
                         evaluate(model, val_data, device, epoch, iou_metric, niou_metric, pdfa, miou, roc, len(trainset), loss_func)
                if iou_ > best_iou:
                    best_iou = iou_
                    save_ckpt({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }, save_path=save_pkl,
                        filename='./result_pth/Best_train_.pth.tar')
        else:
            train_loss, current_lr, loss1, loss2 = train_one_epoch(model, optimizer, train_repeat_data, device, epoch,
                                                                   loss_func, )
            if epoch % 10 == 0:
                val_loss, iou_, niou_, miou_, ture_positive_rate, false_positive_rate, recall, precision, pd, fa = \
                    evaluate(model, val_data, device, epoch, iou_metric, niou_metric, pdfa, miou, roc, len(trainset),
                             loss_func)
                if iou_ > best_iou_repeat:
                    best_iou_repeat = iou_
                    save_ckpt({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }, save_path=save_pkl,
                        filename='./result_pth/Best_train_.pth.tar')

        # if epoch % 10 == 0 and epoch > -1:
        # if epoch >= 0:

            # val_loss, iou_, niou_, miou_, ture_positive_rate, false_positive_rate, recall, precision, pd, fa = \
            #     evaluate(model, val_data, device, epoch, iou_metric, niou_metric, pdfa, miou, roc, len(valset), loss_func)
            # tags = ['train_loss', 'val_loss', 'IoU', 'nIoU', 'mIoU', 'PD', 'tp', 'fa', 'rc', 'pr']
            # tb_writer.add_scalar('LR.a', -1, epoch)
            # tb_writer.add_scalar('LR', current_lr, epoch)
            # tb_writer.add_scalar(tags[0], train_loss, epoch)
            # tb_writer.add_scalar('loss1', loss1, epoch)
            # tb_writer.add_scalar('loss2', loss1, epoch)
            # tb_writer.add_scalar(tags[1], val_loss, epoch)
            # tb_writer.add_scalar(tags[2], iou_, epoch)
            # tb_writer.add_scalar(tags[3], niou_, epoch)

            # name = 'Epoch-%3d_IoU-%.4f_nIoU-%.4f.pth.tar' % (epoch, 0, 0)
            # if resume == True or (resume == False and epoch >= 10):
            #     if epoch % 10 != 0:
            #         save_ckpt({
            #             'epoch': epoch,
            #             'state_dict': model.state_dict(),
            #             'optimizer': optimizer.state_dict(),
            #             'scheduler': scheduler.state_dict()
            #         }, save_path=save_pkl,
            #             filename='Best_mIoU_' + name)
                    #
                    # if ops.exists(ops.join(save_pkl, 'Best_mIoU_' + name)):
                    #     os.remove(ops.join(save_pkl, 'Best_mIoU_' + name))



                    # save_ckpt({
                    #     'epoch': epoch,
                    #     'state_dict': model.state_dict(),
                    #
                    #     'optimizer': optimizer.state_dict(),
                    # }, save_path=save_pkl,
                    #     filename='Best_nIoU_' + name)
                    #
                    # if ops.exists(ops.join(save_pkl, 'Best_nIoU_' + last_name_niou)):
                    #     os.remove(ops.join(save_pkl, 'Best_nIoU_' + last_name_niou))
                    # last_name_niou = name
                # else:
                #     save_ckpt({
                #         'epoch': epoch,
                #         'state_dict': model.state_dict(),
                #         'optimizer': optimizer.state_dict(),
                #     }, save_path=save_pkl,
                #         filename='Best_nIoU_' + name)


def run_trainUIU():
    args = parse_args()
    main(args)
if __name__ == '__main__':

    args = parse_args()
    main(args)

