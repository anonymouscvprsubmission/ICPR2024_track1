import os
import pandas as pd
import cv2
import numpy as np
import torch
from PIL import Image
from skimage import measure
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
        # img_crop = np.expand_dims(img_crop, axis=0)
        img_crop = torch.from_numpy(img_crop)
        img_list.append(img_crop)
        # mask_crop = mask[start_y:end_y, start_x:end_x]
        # mask_crop = np.expand_dims(mask_crop, axis=0)
        # mask_crop = torch.from_numpy(mask_crop)
        # mask_list.append(mask_crop)
        point_mask_crop = point[start_x:end_x, start_y:end_y]
        # point_mask_crop = np.expand_dims(point_mask_crop, axis=0)
        point_mask_crop = torch.from_numpy(point_mask_crop)
        point_list.append(point_mask_crop)
        p_tensor = torch.from_numpy(np.array((x, y)))
        p_axis.append(p_tensor)

    return img_list, point_list, p_axis, point_tensor

def cut64window(img_path=f'./dataset/images', mask_path=f'./dataset/points', target_path=f'./dataset/cut64'):
    img_list = os.listdir(img_path)
    pointlist = os.listdir(mask_path)
    for i in range(len(pointlist)):
        print(i)
        img = Image.open(os.path.join(img_path, img_list[i])).convert('L')
        point = Image.open(os.path.join(mask_path, pointlist[i])).convert('L')
        img, point = np.array(img), np.array(point, dtype=np.float32)
        point = np.expand_dims(point, axis=0).astype('float32')
        img = np.expand_dims(img, axis=0).astype('float32')
        img_list2, point_list2, p_axis, point_ori = make_window(torch.from_numpy(img), torch.from_numpy(point))
        for k in range(len(img_list2)):
            name = img_list[i].split('.')[0] + '_' + str(k) + '.png'
            img2 = img_list2[k].numpy().astype('uint8')
            point2 = point_list2[k].numpy().astype('uint8')
            if not os.path.exists(target_path + '/images64'):
                os.makedirs(target_path + '/images64')
            if not os.path.exists(target_path + '/points64'):
                os.makedirs(target_path + '/points64')
            cv2.imwrite(os.path.join(target_path + '/images64', name), img2)
            cv2.imwrite(os.path.join(target_path + '/points64', name), point2)
    split_idxes()
def split_idxes():
    # TODO
    with open('statistics.txt', 'r') as file:
        lines = file.readlines()

    AL = []
    LL = []
    SN = []
    SS = []
    for line in lines:
        parts = line.split('\t')
        if len(parts) >= 3 and parts[1] == 'Air' and parts[2] == 'LWIR':
            AL.append(parts[0])
        if len(parts) >= 3 and parts[1] == 'Land' and parts[2] == 'LWIR':
            LL.append(parts[0])
        if len(parts) >= 3 and parts[1] == 'Space' and parts[2] == 'NIR':
            SN.append(parts[0])
        if len(parts) >= 3 and parts[1] == 'Space' and parts[2] == 'SWIR':
            SS.append(parts[0])
    namelist = os.listdir('./dataset/cut64/images64')
    AL64 = []
    LL64 = []
    SN64 = []
    SS64 = []
    for i in range(len(namelist)):
        if namelist[i].split('_')[0] in AL:
            AL64.append(namelist[i])
        elif namelist[i].split('_')[0] in LL:
            LL64.append(namelist[i])
        elif namelist[i].split('_')[0] in SN:
            SN64.append(namelist[i])
        elif namelist[i].split('_')[0] in SS:
            SS64.append(namelist[i])
    targetpath = f'./dataset/cut64/idx64'
    if not os.path.exists(targetpath):
        os.makedirs(targetpath)
    with open(os.path.join(targetpath, 'al64.txt'), 'w') as file:
        for i in range(len(AL64)):
            file.write(AL64[i].split('.')[0] + '\n')
    with open(os.path.join(targetpath, 'll64.txt'), 'w') as file:
        for i in range(len(LL64)):
            file.write(LL64[i].split('.')[0] + '\n')
    with open(os.path.join(targetpath, 'sn64.txt'), 'w') as file:
        for i in range(len(SN64)):
            file.write(SN64[i].split('.')[0] + '\n')
    with open(os.path.join(targetpath, 'ss64.txt'), 'w') as file:
        for i in range(len(SS64)):
            file.write(SS64[i].split('.')[0] + '\n')

def filter_img():
    filelist = os.listdir('./dataset/cut64')
    if not os.path.exists(f'./dataset/cut64/mask64'):
        os.mkdir(f'./dataset/cut64/mask64')
    for i in range(len(filelist)):
        if 'LESPS' in filelist[i]:
            namelist = os.listdir('./dataset/cut64/' + filelist[i])
            for j in range(len(namelist)):
                mask = cv2.imread('./dataset/cut64/' + filelist[i] + '/' + namelist[j], cv2.IMREAD_GRAYSCALE)
                mask[mask > 70] = 255
                mask[mask <= 70] = 0
                if np.sum(mask) <= 255 * 64 * 64 * 0.01:
                    cv2.imwrite('./dataset/cut64/mask64/' + namelist[j], mask)


def update_mask():
    if not os.path.exists(f'./dataset/cut64/update64'):
        os.mkdir(f'./dataset/cut64/update64')
    namelist = os.listdir('./dataset/cut64/mask64')
    for name in namelist:
        # if name != '00269_0.png':
        #     continue
        mask = cv2.imread(f'./dataset/cut64/mask64/' + name, cv2.IMREAD_GRAYSCALE)
        update = mask
        mk = measure.label(mask, connectivity=2)
        m = measure.regionprops(mk)
        ref = cv2.imread(f'./dataset/cut64/points64/' + name, cv2.IMREAD_GRAYSCALE)
        rf = measure.label(ref, connectivity=2)
        r = measure.regionprops(rf)
        for i in range(len(m)):
            flag = 0
            centroid_label = np.array(list(m[i].centroid))
            for j in range(len(r)):
                centroid_image = np.array(list(r[j].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                if distance < 3:
                    flag = 1
            if flag == 0:
                area_image = np.array(m[i].coords)
                update[area_image] = 0
        img = cv2.imread(f'./dataset/cut64/images64/' + name, cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((7, 7), np.uint8)
        update_ = cv2.erode(update, kernel, iterations=1)
        update = update if np.sum(update_) == 0 else update_

        img_temp = img.copy()
        img_temp2 = img.copy()
        img_temp[update == 0] = 0
        mean_pixel = (np.sum(img_temp) / np.sum(update) ) * 255
        kernel = np.ones((7, 7), np.uint8)

        # 膨胀操作
        dilation = cv2.dilate(update, kernel, iterations=1)

        kernal2 = np.ones((9, 9), np.uint8)
        dilation2 = cv2.dilate(update, kernal2, iterations=1)
        # dilation2[dilation != 0] = 0
        img_scale_mean = np.sum(img[dilation != 0]) / np.sum(dilation2) * 255
        img_temp2[dilation == 0] = 0
        update[img_temp2 > (mean_pixel + img_scale_mean) * 0.6] = 255

        mk = measure.label(update, connectivity=2)
        m = measure.regionprops(mk)
        ref = cv2.imread(f'./dataset/cut64/points64/' + name, cv2.IMREAD_GRAYSCALE)
        rf = measure.label(ref, connectivity=2)
        r = measure.regionprops(rf)
        for i in range(len(m)):
            flag = 0
            centroid_label = np.array(list(m[i].centroid))
            for j in range(len(r)):
                centroid_image = np.array(list(r[j].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                if distance < 3:
                    flag = 1
            if flag == 0:
                area_image = np.array(m[i].coords)
                update[area_image] = 0

        mk = measure.label(update, connectivity=2)
        m = measure.regionprops(mk)
        ref = cv2.imread(f'./dataset/cut64/points64/' + name, cv2.IMREAD_GRAYSCALE)
        rf = measure.label(ref, connectivity=2)
        r = measure.regionprops(rf)
        temp = r
        r = m
        m = temp
        l = 0
        for i in range(len(m)):
            flag = 0
            centroid_label = np.array(list(m[i].centroid))
            for j in range(len(r)):
                centroid_image = np.array(list(r[j].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                if distance < 3:
                    flag = 1
            if flag == 0:
                l += 1


        if np.sum(update) != 0 and l == 0:
            cv2.imwrite(f'./dataset/cut64/update64/' + name, update)
    with open(f'./dataset/cut64/idx64/train64.txt', 'w') as f:
        trainlist = os.listdir('./dataset/cut64/update64')
        for i in range(len(trainlist)):
            f.write(trainlist[i].split('.')[0] + '\n')

if __name__ == '__main__':
    # cut64window()
    # split_idxes()
    # filter_img()
    update_mask()







