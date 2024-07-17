import numpy as np
import torch
from tqdm             import tqdm
from PIL import Image
from torch.nn import init
import scipy.io as scio
from datetime import datetime
# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
# Metric, loss .etc
import os
import cv2
# Model


ckptdir = f'./result_DNA/DNA.pth.tar'
batchsize = 128
if not os.path.exists(f'./dataset/masks'):
    os.mkdir(f'./dataset/masks')
saveroot = f'./dataset/masks'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import torch
import torch.nn as nn


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
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter,deep_supervision=False):
        super(DNANet, self).__init__()
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
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(Final_x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(Final_x0_4)
            return output



class TestSetLoader(Dataset):
    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id,transform=None,base_size=512,crop_size=480,suffix='.png'):
        super(TestSetLoader, self).__init__()
        self.transform = transform
        self._items    = img_id
        self.masks     = dataset_dir+'/'+'masks'
        self.images    = dataset_dir+'/'+'images'
        self.points = dataset_dir+'/'+'points'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _testval_sync_transform(self, img, point):
        base_size = self.base_size
        # img  = img.resize ((base_size, base_size), Image.BILINEAR)
        # mask = mask.resize((base_size, base_size), Image.NEAREST)
        # point = point.resize((base_size, base_size), Image.NEAREST)
        # final transform
        img, point = np.array(img), np.array(point, dtype=np.float32)  # img: <class 'mxnet.ndarray.ndarray.NDArray'> (512, 512, 3)
        return img, point

    def __getitem__(self, idx):
        # print('idx:',idx)
        img_id = self._items[idx]  # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
        img_path   = self.images+'/'+img_id+self.suffix    # img_id的数值正好补了self._image_path在上面定义的2个空
        # label_path = self.masks +'/'+img_id+self.suffix
        point_path = self.points +'/'+img_id+self.suffix
        img  = Image.open(img_path).convert('L')  ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        # mask = Image.open(label_path)
        point = Image.open(point_path)
        # synchronized transform
        img, point = self._testval_sync_transform(img, point)

        img = img.astype('float32')
        img = (img - 68.92799377441406) / 20.458635330200195
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        # mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0
        point = np.expand_dims(point, axis=0).astype('float32') / 255.0
        point =  torch.from_numpy(point)
        img, point, p_axis, point_ori = make_window(img,  point)
        try:
            if img != [] and point != [] and p_axis != [] and point_ori != []:
                img = torch.stack(img)

                point = torch.stack(point)
                p_axis = torch.stack(p_axis)
        except:
            return [], [], [], []

        return img, point, p_axis, point_ori  # img_id[-1]

    def __len__(self):
        return len(self._items)


def make_window(img, point):

    img_tensor = torch.nn.functional.pad(img, (32, 32, 32, 32), mode='reflect')

    point_tensor = torch.nn.functional.pad(point, (32, 32, 32, 32), mode='reflect')
    img_tensor = img_tensor[0, :, :]

    point_tensor = point_tensor[0, :, :]
    h, w = img_tensor.size()

    img = img_tensor.detach().numpy()

    point = point_tensor.detach().numpy()
    points = np.argwhere(point > 0)
    new_p = []
    for i, p in enumerate(points):
        if p[0] >= 32 and p[1] >= 32 and p[0] < h - 32 and p[1] < w - 32:
            new_p.append(p)

    img_list = []

    point_list = []
    p_axis = []
    for p in new_p:
        x, y = p
        start_x = x - 32
        start_y = y - 32
        end_x = x + 32
        end_y = y + 32
        img_crop = img[start_x:end_x, start_y:end_y]
        img_crop = np.expand_dims(img_crop, axis=0)
        img_crop = torch.from_numpy(img_crop)
        img_list.append(img_crop)
        # mask_crop = mask[start_y:end_y, start_x:end_x]
        # mask_crop = np.expand_dims(mask_crop, axis=0)
        # mask_crop = torch.from_numpy(mask_crop)
        # mask_list.append(mask_crop)
        point_mask_crop = point[start_x:end_x, start_y:end_y]
        point_mask_crop = np.expand_dims(point_mask_crop, axis=0)
        point_mask_crop = torch.from_numpy(point_mask_crop)
        point_list.append(point_mask_crop)
        p_tensor = torch.from_numpy(np.array((x, y)))
        p_axis.append(p_tensor)

    return img_list, point_list, p_axis, point_tensor
def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal(m.weight.data)

def load_dataset (root, dataset, split_method):
    namelist = os.listdir(f'./dataset/images')
    with open(f'./dataset/train.txt', 'w') as f:
        for name in namelist:
            f.write(name.split('.')[0] + '\n')
    train_txt = root + '/' + dataset + '/' + split_method + '/' + 'train.txt'
    test_txt  = root + '/' + dataset + '/' + split_method + '/' + 'train.txt'
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


def load_param(channel_size, backbone):
    if channel_size == 'one':
        nb_filter = [4, 8, 16, 32, 64]
    elif channel_size == 'two':
        nb_filter = [8, 16, 32, 64, 128]
    elif channel_size == 'three':
        nb_filter = [16, 32, 64, 128, 256]
    elif channel_size == 'four':
        nb_filter = [32, 64, 128, 256, 512]

    if   backbone == 'resnet_10':
        num_blocks = [1, 1, 1, 1]
    elif backbone == 'resnet_18':
        num_blocks = [2, 2, 2, 2]
    elif backbone == 'resnet_34':
        num_blocks = [3, 4, 6, 3]
    elif backbone == 'vgg_10':
        num_blocks = [1, 1, 1, 1]
    return nb_filter, num_blocks
class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args

        self.save_prefix = '_'.join([args.model, args.dataset])
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt=load_dataset(args.root, args.dataset,args.split_method)
        # # TODO: just for debugging
        # val_img_ids = val_img_ids[::100]
        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          # transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=64, crop_size=64, transform=input_transform,suffix='.png')
        self.test_data  = DataLoader(dataset=testset,  batch_size=1, num_workers=0,drop_last=False, shuffle=False)

        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'DNANet':
            model       = DNANet(num_classes=1,input_channels=2, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        model           = model.to(device)
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # Initialize evaluation metrics
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Load trained model
        checkpoint        = torch.load(ckptdir, map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])

        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)

        with torch.no_grad():
            num = 0
            for i, (img, point, p_axis, point_ori) in enumerate(tbar):
                if img != [] and point != [] and p_axis != [] and point_ori != []:
                    pass
                else:
                    continue
                img, point, p_axis, mask = img.squeeze(0), point.squeeze(0), p_axis.squeeze(0), point_ori.squeeze()
                num_iter = img.shape[0] // batchsize + 1
                for j in range(num_iter):
                    start = j * batchsize
                    end = (j + 1) * batchsize if (j + 1) * batchsize < img.shape[0] else img.shape[0]
                    img = img[start:end]
                    point = point[start:end]
                    p_axis = p_axis[start:end]
                    # img, mask = img.unsqueeze(1), mask.unsqueeze(1)
                    data = torch.cat([img, point], dim=1)
                    data = data.to(device)
                    output = self.model(data)[-1]
                    # output = torch.sigmoid(output)
                    output[output < 0] = 0
                    output[output > 0] = 1
                    # print(torch.max(output))
                    point_ori = point_ori.to(device)
                    for k in range(output.shape[0]):
                        out = output[k, 0, :, :]
                        axis_x, axis_y = p_axis[k, 0], p_axis[k, 1]
                        x_start , x_end = axis_x - 32, axis_x + 32
                        y_start, y_end = axis_y - 32, axis_y + 32
                        point_ori[0, x_start:x_end, y_start:y_end] += out
                mask[mask > 0] = 1
                mask = point_ori.detach().cpu().numpy()
                mask = mask[0, 32:-32, 32:-32]

                mask = mask * 255.# .astype(np.uint8)
                # mask[mask<254.5] = 0

                mask = mask.astype(np.uint8)
                mask_num = val_img_ids[i]
                cv2.imwrite(saveroot + '/' + val_img_ids[i] + '.png', mask)
                # print(i)
        if not os.path.exists('./dataset/idx'):
            os.makedirs('./dataset/idx')
        with open('./dataset/idx/train.txt', 'w') as f:
            for name in os.listdir('./dataset/masks'):
                f.write(name.split('.')[0] + '\n')
        with open('./dataset/idx/train_repeat.txt', 'w') as f:
            for name in os.listdir('./dataset/masks'):
                img = cv2.imread('./dataset/masks/' + name, cv2.IMREAD_GRAYSCALE)
                s = img.shape[0] * img.shape[1] // (512*512) + 1
                for i in range(s):
                    f.write(name.split('.')[0] + '\n')

def main(args):
    trainer = Trainer(args)
import argparse
def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Dense_Nested_Attention_Network_For_SIRST')
    # choose model
    parser.add_argument('--model', type=str, default='DNANet',
                        help='model name: DNANet')
    # parameter for DNANet
    parser.add_argument('--channel_size', type=str, default='three',
                        help='one,  two,  three,  four')
    parser.add_argument('--backbone', type=str, default='resnet_18',
                        help='vgg10, resnet_10,  resnet_18,  resnet_34 ')
    parser.add_argument('--deep_supervision', type=str, default='True', help='True or False (model==DNANet)')


    # data and pre-process
    parser.add_argument('--dataset', type=str, default='dataset',
                        help='dataset name:  NUDT-SIRST, NUAA-SIRST, NUST-SIRST')
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--test_size', type=float, default='0.5', help='when mode==Ratio')
    parser.add_argument('--root', type=str, default='./')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--split_method', type=str, default='',
                        help='50_50, 10000_100(for NUST-SIRST)')
    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=2,
                        help='in_channel=3 for pre-process')
    parser.add_argument('--base_size', type=int, default=60,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=60,
                        help='crop image size')

    #  hyper params for training
    parser.add_argument('--epochs', type=int, default=1500, metavar='N',
                        help='number of epochs to train (default: 110)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--train_batch_size', type=int, default=128,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test_batch_size', type=int, default=128,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')
    parser.add_argument('--min_lr', default=1e-5,
                        type=float, help='minimum learning rate')
    parser.add_argument('--optimizer', type=str, default='Adagrad',
                        help=' Adam, Adagrad')
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau'])
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.1)')
    # cuda and logging
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')


    args = parser.parse_args()
    # make dir for save result
    args.save_dir = make_dir(args.deep_supervision, args.dataset, args.model)
    # save training log
    save_train_log(args, args.save_dir)
    # the parser
    return args

def make_dir(deep_supervision, dataset, model):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    if deep_supervision:
        save_dir = "%s_%s_%s_wDS" % (dataset, model, dt_string)
    else:
        save_dir = "%s_%s_%s_woDS" % (dataset, model, dt_string)
    os.makedirs('result/%s' % save_dir, exist_ok=True)
    return save_dir
def save_train_log(args, save_dir):
    dict_args=vars(args)
    args_key=list(dict_args.keys())
    args_value = list(dict_args.values())
    with open('result/%s/train_log.txt'%save_dir ,'w') as  f:
        now = datetime.now()
        f.write("time:--")
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write(dt_string)
        f.write('\n')
        for i in range(len(args_key)):
            f.write(args_key[i])
            f.write(':--')
            f.write(str(args_value[i]))
            f.write('\n')
    return

def test_dna():
    args = parse_args()
    main(args)

if __name__ == '__main__':
    args = parse_args()
    main(args)

    pass