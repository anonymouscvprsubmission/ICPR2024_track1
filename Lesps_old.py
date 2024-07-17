import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import random
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.nn.functional as F
import os
import shutil
from math import sqrt
from skimage import measure
import argparse
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def seed_pytorch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_crop(img, mask, patch_size):  # HR: N*H*W
    h, w = img.shape
    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)), mode='constant')
        mask = np.pad(mask, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)), mode='constant')
        h, w = img.shape
    h_start = random.randint(0, h - patch_size)
    h_end = h_start + patch_size
    w_start = random.randint(0, w - patch_size)
    w_end = w_start + patch_size

    for i in range(10):
        img_patch = img[h_start:h_end, w_start:w_end]
        mask_patch = mask[h_start:h_end, w_start:w_end]
        if np.max(mask_patch) > 0:
            break

    return img_patch, mask_patch


def Normalized(img, img_norm_cfg):
    return (img - img_norm_cfg['mean']) / img_norm_cfg['std']


def Denormalization(img, img_norm_cfg):
    return img * img_norm_cfg['std'] + img_norm_cfg['mean']


class SoftIoULoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SoftIoULoss, self).__init__()

    def forward(self, preds, target, avg_factor=0):
        loss = 0
        for pred in preds:
            loss += self.forward_loss(pred, target)
        loss /= len(preds)

        return loss

    def forward_loss(self, pred, target, avg=0):
        # Old One
        # pred = torch.sigmoid(pred)
        smooth = 1

        # print("pred.shape: ", pred.shape)
        # print("target.shape: ", target.shape)

        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() - intersection.sum() + smooth)

        # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
        #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
        #         - intersection.sum(axis=(1, 2, 3)) + smooth)

        loss = 1 - loss.mean()
        # loss = (1 - loss).mean()

        return loss


class FocalLoss(nn.Module):
    """focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                preds,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                loss_reg = self.loss_weight * focal_loss(
                    pred,
                    target,
                    alpha=self.alpha,
                    gamma=self.gamma)
                loss_reg = weight_reduce_loss(loss_reg, weight, reduction, avg_factor)
                loss_total = loss_total + loss_reg
            return loss_total
        else:
            pred = preds
            loss_reg = self.loss_weight * focal_loss(
                pred,
                target,
                alpha=self.alpha,
                gamma=self.gamma)
            loss_reg = weight_reduce_loss(loss_reg, weight, reduction, avg_factor)
            loss_total = loss_reg
            return loss_total


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def focal_loss(pred, target, alpha=2.0, gamma=4.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>'

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = 1e-12
    pos_weights = target
    neg_weights = (1 - target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    return pos_loss + neg_loss


def get_img_norm_cfg(dataset_name, dataset_dir):
    if dataset_name == 'NUAA-SIRST':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'NUDT-SIRST':
        img_norm_cfg = dict(mean=107.80905151367188, std=33.02274703979492)
    elif dataset_name == 'IRSTD-1K':
        img_norm_cfg = dict(mean=87.4661865234375, std=39.71953201293945)
    elif dataset_name == 'SIRST3':
        img_norm_cfg = dict(mean=95.010, std=41.511)
    elif dataset_name == 'NUDT-SIRST-Sea':
        img_norm_cfg = dict(mean=43.62403869628906, std=18.91838264465332)
    elif dataset_name == 'SIRST4':
        img_norm_cfg = dict(mean=62.10432052612305, std=23.96998405456543)
    elif dataset_name == 'track1_full':
        img_norm_cfg = dict(mean=68.92799377441406, std=20.458635330200195)
    else:
        img_norm_cfg = dict(mean=68.92799377441406, std=20.458635330200195)
        # print('starts calculating')
        # with open(dataset_dir + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
        #     train_list = f.read().splitlines()
        # # with open(dataset_dir+'/img_idx/test_' + dataset_name + '.txt', 'r') as f:
        # #     test_list = f.read().splitlines()
        # # img_list = train_list + test_list
        # img_list = train_list
        # img_dir = dataset_dir + '/images/'
        # mean_list = []
        # std_list = []
        # for img_pth in img_list:
        #     if '.' not in img_pth:
        #         img_pth = img_pth + '.png'
        #     img = Image.open(img_dir + img_pth).convert('I')
        #     img = np.array(img, dtype=np.float32)
        #     mean_list.append(img.mean())
        #     std_list.append(img.std())
        # img_norm_cfg = dict(mean=float(np.array(mean_list).mean()), std=float(np.array(std_list).mean()))
    print(dataset_name + ':\t' + str(img_norm_cfg))
    return img_norm_cfg


def PadImg(img, times=32):
    h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0), (0, (w // times + 1) * times - w)), mode='constant')
    return img


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, label_type, patch_size, masks_update, img_norm_cfg=None):
        super(TrainSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.patch_size = patch_size
        self.tranform = augumentation()
        self.masks_update = masks_update
        with open(self.dataset_dir + '/idx64/' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.dataset_name = dataset_name
        ### ---------------------- for label update ----------------------
        self.label_type = label_type
        if isinstance(masks_update, str):
            if os.path.exists(masks_update):
                shutil.rmtree(masks_update)
            os.makedirs(masks_update)
            for img_idx in self.train_list:
                img_idx = img_idx.split('.')[0]
                shutil.copyfile(self.dataset_dir + '/' + '/points' + self.label_type + '/' + img_idx + '.png',
                                masks_update + '/' + img_idx + '.png')
        if isinstance(masks_update, list):
            self.masks_update = masks_update

    def __getitem__(self, idx):
        img = Image.open(self.dataset_dir + '/images64/' + self.train_list[idx].split('.')[0] + '.png').convert('I')
        if isinstance(self.masks_update, str):
            imgidx = self.train_list[idx]
            imgidx = imgidx.split('.')[0]
            mask = Image.open(self.masks_update + '/' + imgidx + '.png')
            mask = np.array(mask, dtype=np.float32) / 255.0
        elif isinstance(self.masks_update, list):
            mask = self.masks_update[idx]

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        img_patch, mask_patch = random_crop(img, mask, self.patch_size)
        img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        return img_patch, mask_patch

    def __len__(self):
        return len(self.train_list)


class TrainSetLoader_full(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        super(TrainSetLoader_full).__init__()
        self.dataset_dir = dataset_dir
        self.patch_size = patch_size
        self.tranform = augumentation()
        with open(self.dataset_dir + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.dataset_name = dataset_name

    def __getitem__(self, idx):
        img = Image.open(self.dataset_dir + '/images/' + self.train_list[idx] + '.png').convert('I')
        mask = Image.open(self.dataset_dir + '/masks/' + self.train_list[idx] + '.png')
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        img_patch, mask_patch = random_crop(img, mask, self.patch_size)
        img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        return img_patch, mask_patch

    def __len__(self):
        return len(self.train_list)


def window_partition_test(x, win_size, dilation_rate=1):
    B, C, H, W = x.shape
    if dilation_rate != 1:
        # x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                     stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.contiguous()  # B',C ,Wh ,Ww
    else:
        x = x.permute(0, 2, 3, 1).view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(-1, C, win_size, win_size)  # B',C ,Wh ,Ww
    return windows


def window_reverse_test(windows, win_size, H, W, dilation_rate=1):
    # B',C ,Wh ,Ww
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    B = 1 if B == 0 else B
    x = windows.view(B, H // win_size, W // win_size, -1, win_size, win_size)
    if dilation_rate != 1:
        x = windows.permute(0, 3, 4, 5, 1, 2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                   stride=win_size)
    else:
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)
    return x


class Update_mask(Dataset):
    def __init__(self, dataset_dir, dataset_name, label_type, masks_update, img_norm_cfg=None):
        super(Update_mask).__init__()
        self.label_type = label_type
        self.masks_update = masks_update
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        with open(self.dataset_dir + '/idx64/' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg

    def __getitem__(self, idx):
        img = Image.open(self.dataset_dir + '/images64/' + self.train_list[idx].split('.')[0] + '.png').convert('I')
        mask = Image.open(self.dataset_dir + '/images64/' + self.train_list[idx].split('.')[0] + '.png')
        if isinstance(self.masks_update, str):
            mask_update = Image.open(self.masks_update + '/' + self.train_list[idx].split('.')[0] + '.png')
            update_dir = self.masks_update + '/' + self.train_list[idx] + '.png'
            mask_update = np.array(mask_update, dtype=np.float32) / 255.0
            if len(mask_update.shape) > 2:
                mask_update = mask_update[:, :, 0]
        elif isinstance(self.masks_update, list):
            mask_update = self.masks_update[idx]
            update_dir = idx

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0

        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        h, w = img.shape
        # times = 32
        times = 32
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, (w // times + 1) * times - w)), mode='constant')
        mask = np.pad(mask, ((0, (h // times + 1) * times - h), (0, (w // times + 1) * times - w)), mode='constant')
        mask_update = np.pad(mask_update, ((0, (h // times + 1) * times - h), (0, (w // times + 1) * times - w)),
                             mode='constant')

        img, mask, mask_update = img[np.newaxis, :], mask[np.newaxis, :], mask_update[np.newaxis, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        mask_update = torch.from_numpy(np.ascontiguousarray(mask_update))
        return img, mask, mask_update, update_dir, [h, w]

    def __len__(self):
        return len(self.train_list)


class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir
        with open(self.dataset_dir + '/idx64/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg

    def __getitem__(self, idx):
        img = Image.open(self.dataset_dir + '/images/' + self.test_list[idx].split('.')[0] + '.png').convert('I')
        mask = Image.open(self.dataset_dir + '/masks/' + self.test_list[idx].split('.')[0] + '.png')

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0

        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        h, w = img.shape
        times = 32
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, (w // times + 1) * times - w)), mode='constant')
        mask = np.pad(mask, ((0, (h // times + 1) * times - h), (0, (w // times + 1) * times - w)), mode='constant')

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask, [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)


class InferenceSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(InferenceSetLoader).__init__()
        self.dataset_dir = dataset_dir
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg

    def __getitem__(self, idx):
        img = Image.open(self.dataset_dir + '/images/' + self.test_list[idx] + '.png').convert('I')

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)

        h, w = img.shape
        times = 32
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, (w // times + 1) * times - w)), mode='constant')

        img = img[np.newaxis, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        return img, [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)


class augumentation(object):
    def __call__(self, input, target):
        if random.random() < 0.5:
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random() < 0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random() < 0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input, target



class VGG_CBAM_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out

class DNANet(nn.Module):
    def __init__(self, num_classes=1,input_channels=1, block=Res_CBAM_block, num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256], deep_supervision=True, mode='test'):
        super(DNANet, self).__init__()
        self.mode = mode
        self.relu = nn.ReLU(inplace = True)
        self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)
        self.up_8  = nn.Upsample(scale_factor=8,   mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16,  mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv0_1 = self._make_layer(block, nb_filter[0] + nb_filter[1],  nb_filter[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3] + nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4] + nb_filter[2],  nb_filter[3], num_blocks[2])

        self.conv0_2 = self._make_layer(block, nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = self._make_layer(block, nb_filter[1]*2 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_2 = self._make_layer(block, nb_filter[2]*2 + nb_filter[3]+ nb_filter[1], nb_filter[2], num_blocks[1])

        self.conv0_3 = self._make_layer(block, nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = self._make_layer(block, nb_filter[1]*3 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])

        self.conv0_4 = self._make_layer(block, nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        self.conv0_4_final = self._make_layer(block, nb_filter[0]*5, nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        if self.deep_supervision:
            self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0),self.down(x0_1)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0),self.down(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1),self.down(x0_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0),self.down(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1),self.down(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2),self.down(x0_3)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        Final_x0_4 = self.conv0_4_final(
            torch.cat([self.up_16(self.conv0_4_1x1(x4_0)),self.up_8(self.conv0_3_1x1(x3_1)),
                       self.up_4 (self.conv0_2_1x1(x2_2)),self.up  (self.conv0_1_1x1(x1_3)), x0_4], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1).sigmoid()
            output2 = self.final2(x0_2).sigmoid()
            output3 = self.final3(x0_3).sigmoid()
            output4 = self.final4(Final_x0_4).sigmoid()
            if self.mode == 'train':
                return [output1, output2, output3, output4]
            else:
                return output4
        else:
            output = self.final(Final_x0_4).sigmoid()
            return output


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal(m.weight.data)


class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name

        if model_name == 'DNANet':
            if mode == 'train':
                self.model = DNANet(mode='train')
            else:
                self.model = DNANet(mode='test')
        # elif model_name == 'ACM':
        #     self.model = ACM()
        # elif model_name == 'ALCNet':
        #     self.model = ALCNet()
        self.model.apply(weights_init_xavier)
        self.cal_loss = FocalLoss()
        # self.cal_loss = SoftIoULoss()

    def forward(self, img):
        pred = self.model(img)
        # if isinstance(pred, list):
        #     pred = pred[-1]
        return pred

    def loss(self, pred, gt_mask):
        target_mask, avg_factor = gt_mask, max(1, (gt_mask.eq(1)).sum())
        loss = self.cal_loss(pred, target_mask, avg_factor=avg_factor)
        return loss

    def update_gt(self, pred, gt_masks, thresh_Tb, thresh_k, size):
        bs, c, feat_h, feat_w = pred.shape
        update_gt_masks = gt_masks.clone()
        background_length = 33
        target_length = 3

        label_image = measure.label((gt_masks[0, 0, :, :] > 0.5).cpu())
        for region in measure.regionprops(label_image, cache=False):
            cur_point_mask = pred.new_zeros(bs, c, feat_h, feat_w)
            cur_point_mask[0, 0, int(region.centroid[0]), int(region.centroid[1])] = 1
            nbr_mask = ((F.conv2d(cur_point_mask,
                                  weight=torch.ones(1, 1, background_length, background_length).to(gt_masks.device),
                                  stride=1, padding=background_length // 2)) > 0).float()
            targets_mask = ((F.conv2d(cur_point_mask,
                                      weight=torch.ones(1, 1, target_length, target_length).to(gt_masks.device),
                                      stride=1, padding=target_length // 2)) > 0).float()

            ### Candidate Pixels Extraction
            max_limitation = size[0] * size[1] * 0.0015
            threshold_start = (pred * nbr_mask).max() * thresh_Tb
            threshold_delta = (thresh_k * ((pred * nbr_mask).max() - threshold_start) * (
                        len(region.coords) / max_limitation).to(gt_masks.device)).to(gt_masks.device)
            threshold = threshold_start + threshold_delta
            thresh_mask = (pred * nbr_mask > threshold).float()

            ### False Alarm Elimination
            label_image = measure.label((thresh_mask[0, :, :, :] > 0).cpu())
            if label_image.max() > 1:
                for num_cur in range(label_image.max()):
                    curr_mask = thresh_mask * torch.tensor(label_image == (num_cur + 1)).float().unsqueeze(0).to(
                        gt_masks.device)
                    if (curr_mask * targets_mask).sum() == 0:
                        thresh_mask = thresh_mask - curr_mask

            ### Average Weighted Summation
            target_patch = (update_gt_masks * thresh_mask + pred * thresh_mask) / 2
            background_patch = update_gt_masks * (1 - thresh_mask)
            update_gt_masks = background_patch + target_patch

        ### Ensure initial GT point label
        update_gt_masks = torch.max(update_gt_masks, (gt_masks == 1).float())

        return update_gt_masks


def train():
    train_set = TrainSetLoader(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, label_type=opt.label_type,
                               patch_size=opt.patchSize, masks_update=opt.masks_update, img_norm_cfg=opt.img_norm_cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=opt.batchSize, shuffle=True)

    epoch_state = 0
    total_loss_list = []
    total_loss_epoch = []
    update_epoch_loss = []
    idx_epoch_list = []
    start_click = 0

    net = Net(model_name=opt.model_name, mode='train').cuda()
    if opt.resume:
        ckpt = torch.load(opt.resume)
        net.load_state_dict(ckpt['state_dict'])
        epoch_state = ckpt['epoch']
        total_loss_list = ckpt['total_loss']
        for i in range(len(opt.step)):
            opt.step[i] = opt.step[i] - epoch_state

    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.step, gamma=opt.gamma)
    iter_num = 0
    idx_epoch = 0
    while True:

        if idx_epoch > opt.nEpochs:
            break
        train_loader_iter = iter(train_loader)
        tqdm_train_loader = tqdm(range(len(train_loader)), desc='Epoch {}'.format(idx_epoch + 1))
        # for idx_iter, (img, gt_mask) in enumerate(train_loader):

        for idx_iter in tqdm_train_loader:
            iter_num += opt.batchSize
            img, gt_mask = next(train_loader_iter)
            net.train()
            img, gt_mask = Variable(img).cuda(), Variable(gt_mask).cuda()
            pred = net.forward(img)
            loss = net.loss(pred, gt_mask)
            total_loss_epoch.append(loss.detach().cpu())
            tqdm_train_loader.desc = 'loss:{}'.format(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter_num > 4000:
                idx_epoch += 1
                iter_num -= 4000
                if (idx_epoch + 1) % 1 == 0:
                    total_loss_list.append(float(np.array(total_loss_epoch).mean()))
                    print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,' % (idx_epoch + 1, total_loss_list[-1]))
                    opt.f.write(
                        time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,\n' % (idx_epoch + 1, total_loss_list[-1]))
                    total_loss_epoch = []

                # first update
                # if (idx_epoch + 1) > opt.LESPS_Tepoch and start_click == 0:
                if total_loss_list[-1] < opt.LESPS_Tloss and start_click == 0:
                    # if start_click == 0:
                    print('update start')
                    start_click = 1
                    save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.save_perdix + '_' + str(
                        idx_epoch + 1) + '.pth.tar'
                    save_checkpoint({
                        'epoch': idx_epoch + 1,
                        'state_dict': net.state_dict(),
                        'total_loss': total_loss_list,
                        # 'train_iou_list': opt.train_iou_list,
                        # 'test_iou_list': opt.test_iou_list,
                    }, save_pth)
                    update_gt_mask(save_pth, thresh_Tb=opt.LESPS_Tb, thresh_k=opt.LESPS_k)
                    # test(save_pth)
                    update_epoch_loss.append(total_loss_list[-1])

                # subsequent update
                if start_click == 1 and (idx_epoch + 1) % opt.LESPS_f == 0:
                    print('updating')
                    if idx_epoch not in idx_epoch_list:
                        idx_epoch_list.append(idx_epoch)
                        save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.save_perdix + '_' + str(
                            idx_epoch + 1) + '.pth.tar'
                        save_checkpoint({
                            'epoch': idx_epoch + 1,
                            'state_dict': net.state_dict(),
                            'total_loss': total_loss_list,
                            # 'train_iou_list': opt.train_iou_list,
                            # 'test_iou_list': opt.test_iou_list,
                        }, save_pth)
                    update_gt_mask(save_pth, thresh_Tb=opt.LESPS_Tb, thresh_k=opt.LESPS_k)
                    # test(save_pth)
                    update_epoch_loss.append(total_loss_list[-1])

        scheduler.step()


def window_partition_test(x, win_size, dilation_rate=1):
    B, C, H, W = x.shape
    if dilation_rate != 1:
        # x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                     stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.contiguous()  # B',C ,Wh ,Ww
    else:
        x = x.permute(0, 2, 3, 1).view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(-1, C, win_size, win_size)  # B',C ,Wh ,Ww
    return windows


def window_reverse_test(windows, win_size, H, W, dilation_rate=1):
    # B',C ,Wh ,Ww
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    B = 1 if B == 0 else B
    x = windows.view(B, H // win_size, W // win_size, -1, win_size, win_size)
    if dilation_rate != 1:
        x = windows.permute(0, 3, 4, 5, 1, 2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                   stride=win_size)
    else:
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)
    return x


def update_gt_mask(save_pth, thresh_Tb, thresh_k):
    update_set = Update_mask(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, label_type=opt.label_type,
                             masks_update=opt.masks_update, img_norm_cfg=opt.img_norm_cfg)
    update_loader = DataLoader(dataset=update_set, num_workers=1, batch_size=1, shuffle=False)

    net = Net(model_name=opt.model_name, mode='test').cuda()
    ckpt = torch.load(save_pth)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()


    for idx_iter, (img, gt_mask, gt_mask_update, update_dir, size) in tqdm(enumerate(update_loader)):
        img, gt_mask_update = Variable(img).cuda(), Variable(gt_mask_update).cuda()
        # b, c, h, w = img.shape
        # img = window_partition_test(img, 64)
        # batch = img.shape[0]
        # num_iter = batch // 32 + 1
        # predlist = []
        # for k in range(num_iter):
        #     imgbatch = img[k * 32: (k + 1) * 32, :, :, :] if (k + 1) * 32 <= batch else img[k * 32: batch, :, :, :]
        #     with torch.no_grad():
        #         pred = net.forward(imgbatch)
        #         predlist.append(pred)
        # # pred = net.forward(img)
        # pred = torch.cat(predlist, dim=0)
        # pred = window_reverse_test(pred, 64, h, w)
        pred = net.forward(img)
        if isinstance(pred, list):
            pred = pred[-1]
        pred = pred[:, :, :size[0], :size[1]]
        gt_mask = gt_mask[:, :, :size[0], :size[1]]
        gt_mask_update = gt_mask_update[:, :, :size[0], :size[1]]
        b, c, h, w = pred.shape
        # pred = window_partition_test(pred, 256)
        # gt_mask_update = window_partition_test(gt_mask_update, 256)

        update_mask = net.update_gt(pred, gt_mask_update, thresh_Tb, thresh_k, size)

        # update_mask = window_reverse_test(update_mask, 256, h, w)
        if isinstance(update_dir, torch.Tensor):
            opt.masks_update[update_dir] = update_mask[0, 0, :size[0], :size[1]].cpu().detach().numpy()
        else:
            img_save = transforms.ToPILImage()((update_mask[0, :, :size[0], :size[1]]).cpu())
            img_save.save(update_dir[0])


def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)


parser = argparse.ArgumentParser(description="PyTorch LESPS train")
parser.add_argument("--model_names", default=['DNANet'], nargs='+',
                    help="model_name: 'ACM', 'ALCNet', 'DNANet'")
parser.add_argument("--dataset_names", default=['al64', 'll64', 'sn64', 'ss64'], nargs='+',
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'NUDT-SIRST-Sea', 'SIRST3'")
parser.add_argument("--label_type", default='64', type=str, help="label type: centroid, coarse")
parser.add_argument("--LESPS_Tepoch", default=50, type=int, help="Initial evolution epoch, default: 50")
parser.add_argument("--LESPS_Tloss", default=10, type=float, help="Tb in evolution threshold, default: 0.5")
parser.add_argument("--LESPS_Tb", default=0.5, type=float, help="Tb in evolution threshold, default: 0.5")
parser.add_argument("--LESPS_k", default=0.5, type=float, help="k in evolution threshold, default: 0.5")
parser.add_argument("--LESPS_f", default=2, type=int, help="Evolution frequency, default: 5")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--dataset_dir", default='./dataset/cut64', type=str,
                    help="train_dataset_dir, default: './datasets/SIRST3")
parser.add_argument("--batchSize", type=int, default=128, help="Training batch sizse, default: 16")
parser.add_argument("--patchSize", type=int, default=64, help="Training patch size, default: 256")
parser.add_argument("--save", default='./log', type=str, help="Save path, default: './log")
parser.add_argument("--resume", default=None, type=str, help="Resume path, default: None")
parser.add_argument("--nEpochs", type=int, default=500, help="Number of epochs, default: 400")
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate, default: 5e-4")
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma, default: 0.1')
parser.add_argument("--step", type=int, default=[200, 300],
                    help="Sets the learning rate decayed by step, default: [200, 300]")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, default: 1")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test, default: 0.5")
parser.add_argument("--cache", default=False, type=str,
                    help="True: cache intermediate mask results, False: save intermediate mask results")

global opt
opt = parser.parse_args()

def run_lesps(dasaset='', idx=''):

    # opt.dataset_dir = dasaset
    # opt.dataset_names = idx
    dataset_names = opt.dataset_names
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            opt.save_perdix = opt.model_name + '_LESPS_' + opt.label_type

            if opt.cache:
                ### cache intermediate mask results
                with open(opt.dataset_dir + '/idx64/' + opt.dataset_name + '.txt', 'r') as f:
                    train_list = f.read().splitlines()
                opt.masks_update = []
                for idx in range(len(train_list)):
                    mask = Image.open(opt.dataset_dir + '/masks' + opt.label_type + '/' + train_list[idx] + '.png')
                    mask = np.array(mask, dtype=np.float32) / 255.0
                    opt.masks_update.append(mask)
            else:
                ### save intermediate mask results to
                opt.masks_update = opt.dataset_dir + '/' + opt.dataset_name + '_' + opt.save_perdix + '_'

            ### save intermediate loss vaules
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_LESPS_' + opt.label_type + '_' + (
                time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')


            print(opt.dataset_name + '\t' + opt.model_name)
            train()
            print('\n')
            opt.f.close()





if __name__ == '__main__':
    run_lesps(dasaset=f'', idx=f'')