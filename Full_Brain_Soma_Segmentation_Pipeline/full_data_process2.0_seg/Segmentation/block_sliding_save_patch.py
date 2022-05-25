# -*- coding: utf-8 -*-
# @Time     : 2020/7/22 18:30
# @Function : segment a full block by sliding windows using best 3D unet model
# @Software : PyCharm

import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.metric import dc_score
from pathlib import Path
from model.model import UNet3D
from data.dataset import CellSeg_set
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from utils.utils import save_nii, ensure_dir
import SimpleITK as sitk
import time
import json
import os
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--save_dir",type=str,default="/braindat/lab/liuxy/soma_seg/block_prediction")
parser.add_argument("-c", "--ckp_file",type=str,default="/gdata/liuxy/3d_unet/ckps_tri2_again/checkpoint-epoch90.pth")
parser.add_argument("-d", "--data_dir",type=str,default="/braindat/lab/liuxy/soma_seg/block_data/z-3500_y-21034_x-24866_1_raw.nii.gz")
parser.add_argument("-st", "--stride",type=int,default=[29,121,121])
parser.add_argument("-p", "--patch_size",type=int,default=[128,384,384])
parser.add_argument("-u", "--upsample",type=str,default=(1,4,4))
parser.add_argument("-r", "--resample",type=str,default=(1,0.25,0.25))
args = parser.parse_args()

def load_nii(path):
    nii = sitk.ReadImage(path)
    nii = sitk.GetArrayFromImage(nii)
    print(path.split("/")[-1], "loaded!")
    return nii

def save_nii(img,path):
    print(img.shape)
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img, path)
    print(path.split("/")[-1], "saving succeed!")

def cal_crop(w, w_stride, w_patch):
    num = (w-w_patch)/(w_stride)
    return num == int(num), int(num+1)

block_path = args.data_dir
block =  load_nii(block_path)
z,h,w = block.shape
z_stride,h_stride,w_stride = args.stride
z_patch, h_patch, w_patch = args.patch_size
flag1, num_w = cal_crop(w, w_stride, w_patch)
flag2, num_h = cal_crop(h, h_stride, h_patch)
flag3, num_z = cal_crop(z, z_stride, z_patch)
assert flag1 and flag2 and flag3
print("---dimension right!---")

# pre-processing on the block
block = torch.from_numpy(block).unsqueeze(0).unsqueeze(0).type(torch.float32)
print(block.shape)


device = torch.device("cuda:0")
ensure_dir(args.save_dir)
model = UNet3D()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

checkpoint = torch.load(args.ckp_file)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()

# inference the block and merge the block
init_block = np.zeros((np.array(block.shape[2:5]))).astype(np.float32)
# freq = np.zeros((np.array(block.shape[2:5])).astype(int))
#score_map = np.zeros((np.array(block.shape[2:5])).astype(int))
counts = num_z*num_h*num_w
print("There are {} patches need to be dealed with in the block. ".format(counts))
patch_num = 0
effective_num = 0
t1 = time.time()

with torch.no_grad():
    for i in range(num_z):
        for j in range(num_h):
            for k in range(num_w):
                patch = block[..., int(i * z_stride): int(i * z_stride + z_patch),
                              int(j * h_stride): int(j * h_stride + h_patch),
                              int(k * w_stride): int(k * w_stride + w_patch)]

                patch = patch.to(device)
                patch = F.upsample(patch, scale_factor=args.resample, mode="trilinear", align_corners=True)

                infer_patch = model(patch)
                #cls = F.softmax(cls, dim=1)
                #obj_score = cls[:, 1].cpu().numpy()

                infer_patch = F.softmax(infer_patch, dim=1)
                infer_patch = F.upsample(input=infer_patch, scale_factor=args.upsample, mode='trilinear',align_corners=True)[0, 1, ...] # 1 is a channel for foreground
                infer_patch = infer_patch.cpu().numpy()

                patch_path = os.path.join(args.save_dir,block_path.split("/")[-1][:-10]+'patch')
                ensure_dir(patch_path)
                patch_path = os.path.join(patch_path,str(i)+'_'+str(j)+'_'+str(k)+'.nii.gz')
                save_nii(infer_patch.astype(np.float32),patch_path)
                # if infer_patch > 0.3:
                #     infer_patch = infer_patch.cpu().numpy()
                # else:
                #     infer_patch = infer_patch.cpu().numpy()
                    # freq[..., int(i * d_s): int((i * d_s + d_p)),
                    # int(j * h_s): int((j * h_s + h_p)),
                    # int(k * w_s): int((k * w_s + w_p))] = freq[..., int(i * d_s): int((i * d_s + d_p)),
                    #                                       int(j * h_s): int((j * h_s + h_p)),
                    #                                       int(k * w_s): int((k * w_s + w_p))] + np.ones(infer_patch.shape)
                # else:
                #     infer_patch = np.zeros(infer_patch.shape)


                init_block[..., int(i * z_stride): int((i * z_stride + z_patch)),
                              int(j * h_stride): int((j * h_stride + h_patch)),
                              int(k * w_stride): int((k * w_stride + w_patch))] = np.maximum(init_block[..., int(i * z_stride): int((i * z_stride + z_patch)),
                                                                          int(j * h_stride): int((j * h_stride + h_patch)),
                                                                          int(k * w_stride): int((k * w_stride + w_patch))], infer_patch) #maximum in overlap

                patch_num += 1
                print("{}/{} successed!".format(patch_num, counts))
init_block = init_block.astype(np.float32)
t2 = time.time()
print("consume time:",t2-t1)
save_path = args.save_dir
ensure_dir(save_path)
save_path = os.path.join(save_path,block_path.split("/")[-1][:-10]+'output.nii.gz')
block =  save_nii(init_block,save_path)
print("save result successfully")