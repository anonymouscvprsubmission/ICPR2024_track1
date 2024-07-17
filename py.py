import os

namelist = os.listdir(f'./dataset/images')
with open(f'./dataset/train.txt', 'w') as f:
    for name in namelist:
        f.write(name.split('.')[0] + '\n')