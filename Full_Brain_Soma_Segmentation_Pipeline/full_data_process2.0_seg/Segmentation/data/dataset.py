# -*- coding: utf-8 -*-
# @Time    : 2020/5/16 12:50
# @Author  : Bo Hu
# @Email   : hubosist@mail.ustc.edu.cn
# @Software: PyCharm

import os
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
import torchvision
import torch.nn.functional as F
import random
from .augmentation import Flip
from .augmentation import Elastic
from .augmentation import Grayscale # contrast/brightness
from .augmentation import Rotate
from .augmentation import Rescale


class CellSeg_set(Dataset):
    def __init__(self, dir, mode):
        self.dir = dir
        self.mode = mode
        if (self.mode != "train") and (self.mode != "validation"):
            raise ValueError("The value of dataset mode must be assigned to 'train' or 'validation'")
        self.path = os.path.join(dir, mode)
        self.id_num = os.listdir(self.path)
        self.id_num.sort(key=lambda x: int(x))

        self.augs_init()

    def __len__(self):
        return len(self.id_num)

    def __getitem__(self, id):
        data = sitk.ReadImage(os.path.join(self.path,str(id).zfill(3),"data.nii.gz"))
        data = sitk.GetArrayFromImage(data)
        
        label = sitk.ReadImage(os.path.join(self.path,str(id).zfill(3),"label.nii.gz"))
        label = sitk.GetArrayFromImage(label)
        label = label
        # z,h,w = label.shape
        # z_patch,h_patch,w_patch = (192,384,384)
        data = data.astype(np.float32) / 255.0

        if self.mode == 'train':

            data_pair = {'image': data, 'label': label}
            if random.random() > 0.5:
                data_pair = self.aug_flip(data_pair)
            if random.random() > 0.7:
                data_pair = self.aug_rotation(data_pair)
            # if random.random() > 0.5:
            #     data_pair = self.aug_rescale(data_pair)
            if random.random() > 0.5:
                data_pair = self.aug_elastic(data_pair)
            if random.random() > 0.5:
                data_pair = self.aug_grayscale(data_pair)
            data = data_pair['image']
            label = data_pair['label']

        data = data*255.0
        data = torch.from_numpy(data.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))


        # label = self.padding3d(label,(128,384,384))
        # data = self.padding3d(data,(128,384,384))

        # shape = data.shape
        # z,h,w = shape[0],shape[1],shape[2]
        # c_z = z//2
        # c_h = h//2
        # c_w = w//2
        # label = label[c_z-128//2:c_z+128//2,c_h-384//2:c_h+384//2,c_w-384//2:c_w+384//2]
        # data = data[c_z-128//2:c_z+128//2,c_h-384//2:c_h+384//2,c_w-384//2:c_w+384//2]
        # label = self.RandomCrop(label,(128, 384, 384))
        # data = self.RandomCrop(data,(128, 384, 384))

        print(label.shape)

        return data.unsqueeze(0), label.unsqueeze(0)/255

    def augs_init(self):
        # https://zudi-lin.github.io/pytorch_connectomics/build/html/notes/dataloading.html#data-augmentation
        self.aug_rotation = Rotate(p=0.5)
        self.aug_rescale = Rescale(p=0.5)
        self.aug_flip = Flip(p=1.0, do_ztrans=0)
        self.aug_elastic = Elastic(p=0.5, alpha=16, sigma=4.0)
        self.aug_grayscale = Grayscale(p=0.75)

    def padding3d(self,input,patch_size):
        z,h,w = input.shape
        z_patch,h_patch,w_patch = patch_size

        if z<z_patch :
            if (z_patch-z)%2 ==0:
                pad_z1 = (z_patch-z)/2
                pad_z2 = pad_z1
            else:
                pad_z1 = (z_patch-z)//2
                pad_z2 = pad_z1+1
            input = F.pad(input, (0, 0, 0, 0, int(pad_z1), int(pad_z2)), mode='constant', value=0)
        else:
            input = input

        z, h, w = input.shape
        if h<h_patch :
            if (h_patch-h)%2 ==0:
                pad_h1 = (h_patch-h)/2
                pad_h2 = pad_h1
            else:
                pad_h1 = (h_patch-h)//2
                pad_h2 = pad_h1+1

            input = F.pad(input, (0, 0, int(pad_h1), int(pad_h2), 0, 0), mode='constant', value=0)
        else:
            input = input

        z, h, w = input.shape
        if w<w_patch :
            if (w_patch-w)%2 ==0:
                pad_w1 = (w_patch-w)/2
                pad_w2 = pad_w1
            else:
                pad_w1 = (w_patch-w)//2
                pad_w2 = pad_w1+1
            input = F.pad(input, (int(pad_w1),int(pad_w2),0,0,0,0), mode='constant', value=0)
        else:
            input = input


        return input

    def RandomCrop(self,input,patch_size):
        z, h, w = input.shape
        z_patch, h_patch, w_patch = patch_size
        if z>z_patch:
            range_z = torch.tensor(range(int(z_patch / 2), int(z - z_patch / 2)))
            center_z = range_z[torch.randint(0, range_z.__len__(), (1,)).item()]
        else:
            center_z = z_patch/2
        if h > h_patch:
            range_h = torch.tensor(range(int(h_patch / 2), int(h - h_patch / 2)))
            center_h = range_h[torch.randint(0, range_h.__len__(), (1,)).item()]
        else:
            center_h = h_patch/2
        if w>w_patch:
            range_w = torch.tensor(range(int(w_patch / 2), int(w - w_patch / 2)))
            center_w = range_w[torch.randint(0, range_w.__len__(), (1,)).item()]
        else:
            center_w = w_patch/2

        new_label = input[int(center_z)-int(z_patch/2):int(center_z)+int(z_patch/2),\
                    int(center_h)-int(h_patch/2):int(center_h)+int(h_patch/2),\
                    int(center_w)-int(w_patch/2):int(center_w)+int(w_patch/2)]

        return new_label



