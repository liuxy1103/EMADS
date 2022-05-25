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

class CellSeg_set(Dataset):
    def __init__(self, dir, mode):
        self.dir = dir
        self.mode = mode
        if (self.mode != "train") and (self.mode != "validation"):
            raise ValueError("The value of dataset mode must be assigned to 'train' or 'validation'")
        self.path = os.path.join(dir)
        self.id_num = os.listdir(self.path)
        # if self.mode == 'validation':
        #     self.id_num = ['z-1000_y-15328_x-42084_1']

        # self.id_num.sort(key=lambda x: int(x))

    def __len__(self):
        return len(self.id_num)

    def __getitem__(self, id):
        data = sitk.ReadImage(os.path.join(self.path,self.id_num[id],self.id_num[id]+"_raw.nii.gz"))
        data = sitk.GetArrayFromImage(data)
        data = torch.from_numpy(data.astype(np.float32))
        print(self.id_num[id][-2:],self.id_num[id][-2:] == '_1')

        if self.id_num[id][-2:] == '_1':
            label1 = sitk.ReadImage(os.path.join(self.path,self.id_num[id],self.id_num[id]+"_mask.nii.gz"))
            label1 = sitk.GetArrayFromImage(label1)
            label1 = torch.from_numpy(label1.astype(np.float32))
            label2 = sitk.ReadImage(os.path.join(self.path,self.id_num[id],self.id_num[id]+"_seed.nii.gz"))
            label2 = sitk.GetArrayFromImage(label2)
            label2 = torch.from_numpy(label2.astype(np.float32))
        else:
            label1 = sitk.ReadImage(os.path.join(self.path,self.id_num[id],self.id_num[id]+"_mask.nii.gz"))
            label1 = sitk.GetArrayFromImage(label1)
            label1 = torch.from_numpy(label1.astype(np.float32))
        # z,h,w = label.shape
        # z_patch,h_patch,w_patch = (192,384,384)



        label1 = self.padding3d(label1,(62, 459, 459))
        if self.id_num[id][-2:] == '_1':
            label2 = self.padding3d(label2,(62, 459, 459))
        data = self.padding3d(data,(62, 459, 459))

        label1 = self.RandomCrop(label1,(62, 459, 459))
        if self.id_num[id][-2:] == '_1':
            label2 = self.RandomCrop(label2,(62, 459, 459))
        data = self.RandomCrop(data,(62, 459, 459))
        # print(label.shape)
        if self.id_num[id][-2:] == '_1':
            return data.unsqueeze(0), label1.unsqueeze(0),label2.unsqueeze(0)
        else:
            return data.unsqueeze(0), label1.unsqueeze(0),label1.unsqueeze(0)


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



