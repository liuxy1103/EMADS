from random import random
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pprint
import os
import time
from torch.nn import functional as F
from pathlib import Path
from .model.model import UNet3D
from .data.dataset import CellSeg_set
from .utils.utils import save_nii, ensure_dir
from scipy.ndimage.interpolation import zoom

def cal_crop(w, w_s, w_p):
    num = (w-w_p)/(w_s)
    return num == int(num), int(num+1)

def save_nii(img,path):
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img, path)
    print(path.split("/")[-1], "saving succeed!")

def main_predict(thre_0,thre_1,config_path, block,seed_map, model_state_file,if_save_patch,block_name):

    # cudnn related setting
    cudnn.benchmark = True
    cudnn.deterministic = True
    cudnn.enabled = True
    block = np.pad(block, pad_width=((60, 60), (120, 120), (120, 120)), mode='reflect')
    seed_map = np.pad(seed_map, pad_width=((20, 20), (30, 30), (30, 30)), mode='reflect')
    seed_map [-20:,-30:,-30:][seed_map [-20:,-30:,-30:]>0] =  seed_map [-20:,-30:,-30:][seed_map [-20:,-30:,-30:]>0] + seed_map.max()+1
    seed_map [:20,:30,:30][seed_map [:20,:30,:30]>0] = seed_map [:20,:30,:30][seed_map [:20,:30,:30]>0] + seed_map.max()+1
    # seed_map = zoom(seed_map,zoom=(1/3, 1/4, 1/4), order=0)
    # crop the block
    d, h, w = block.shape
    d_s, h_s, w_s = 30, 120, 120
    d_p, h_p, w_p = 130, 410, 410
    flag1, num_w = cal_crop(w, w_s, w_p)
    flag2, num_h = cal_crop(h, h_s, h_p)
    flag3, num_d = cal_crop(d, d_s, d_p)

    assert flag1 and flag2 and flag3
    # print("---dimension right!---")

    # pre-processing on the block
    block = torch.from_numpy(block).unsqueeze(0).unsqueeze(0).float()

    device = torch.device("cuda:0")
    model = UNet3D()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    checkpoint = torch.load(model_state_file)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # t1 = time.time()
    init_block = np.zeros((np.array(block.shape[2:5]))).astype(np.uint8)

    # score_map = np.zeros((np.array(block.shape[2:5])).astype(int))

    seeds = np.unique(seed_map)
    # print('seed list:',seeds)

    root_path = os.getcwd()
    seed_patch_path = os.path.join(root_path,'seed_patch') 
    # block_name
    seed_patch_path = os.path.join(seed_patch_path,block_name) 

    for seed in seeds:
        if seed == 0 :
            continue
        # print('current seed:',seed)
        coordinate = np.where(seed_map==seed)
        i = np.mean(coordinate[0])*3
        j = np.mean(coordinate[1])*4
        k = np.mean(coordinate[2])*4
        shift_list = [4]
        for shift in shift_list:
            # print('current shift:',shift)
            for ii in range(7):
                if ii ==0:
                    d_ss = d_s
                    h_ss = 0
                    w_ss = 0
                elif ii ==1:
                    d_ss = (-1)*d_s
                    h_ss = 0
                    w_ss = 0
                elif ii ==2:
                    d_ss = 0
                    h_ss = h_s
                    w_ss = 0 
                elif ii ==3:
                    d_ss = 0
                    h_ss = (-1)*h_s
                    w_ss = 0

                elif ii ==4:
                    d_ss = 0
                    h_ss = 0
                    w_ss = w_s

                elif ii ==5:
                    d_ss = 0
                    h_ss = 0
                    w_ss = (-1)*w_s
                else:
                    d_ss = 0
                    h_ss = 0
                    w_ss = 0

                try:
                    patch = block[..., max(int(i - d_p/2+shift*d_ss/2),0): int(max(int(i - d_p/2+shift*d_ss/2),0) + d_p),
                                max(int(j - h_p/2+shift*h_ss/2),0): int(max(int(j - h_p/2)+shift*h_ss/2,0) + h_p),
                                max(int(k - w_p/2+shift*w_ss/2),0): int(max(int(k - w_p/2+shift*w_ss/2),0) + w_p)] 
                    patch_tmp = F.upsample(patch, scale_factor=(1,0.25,0.25), mode="trilinear", align_corners=True)
                    with torch.no_grad():
                        infer_patch = model(patch_tmp.cuda())
                        infer_patch = F.softmax(infer_patch, dim=1)
                        infer_patch = F.upsample(input=infer_patch, size=patch.shape[-3:], mode='trilinear')

                        fg_mask = (infer_patch[0, 1, ...] > thre_1) * (infer_patch[0, 2, ...] < thre_0)
                        # infer_patch = infer_patch[0, 1, ...]

                        # infer_patch = fg_mask
                        # infer_patch = infer_patch.detach().cpu().numpy().astype(np.uint8)

                        fg_mask = fg_mask.detach().cpu().numpy().astype(np.uint8)
                except:
                    continue
            
                

                init_block[..., max(int(i - d_p/2+shift*d_ss/2),0): int(max(int(i - d_p/2+shift*d_ss/2),0) + d_p),
                        max(int(j - h_p/2+shift*h_ss/2),0): int(max(int(j - h_p/2)+shift*h_ss/2,0) + h_p),
                        max(int(k - w_p/2+shift*w_ss/2),0): int(max(int(k - w_p/2+shift*w_ss/2),0) + w_p)]    = \
                        np.maximum(init_block[..., max(int(i - d_p/2+shift*d_ss/2),0): int(max(int(i - d_p/2+shift*d_ss/2),0) + d_p),
                            max(int(j - h_p/2+shift*h_ss/2),0): int(max(int(j - h_p/2)+shift*h_ss/2,0) + h_p),
                            max(int(k - w_p/2+shift*w_ss/2),0): int(max(int(k - w_p/2+shift*w_ss/2),0) + w_p)], fg_mask)\


    #process border of the block
    with torch.no_grad():
        for i in range(num_d):

            for j in range(num_h):

                for k in range(num_w):

                    if not (i==0 or j==0 or k==0 or i == num_d-1 or j==num_h-1 or k==num_w-1):
                        continue
                    patch = block[..., int(i * d_s): int(i * d_s + d_p),
                                  int(j * h_s): int(j * h_s + h_p),
                                  int(k * w_s): int(k * w_s + w_p)]
                    patch_tmp = F.upsample(patch, scale_factor=(1,0.25,0.25), mode="trilinear", align_corners=True)
                    with torch.no_grad():
                        infer_patch = model(patch_tmp.cuda())
                    infer_patch = F.softmax(infer_patch, dim=1)
                    infer_patch = F.upsample(input=infer_patch, size=patch.shape[-3:], mode='trilinear')
                    fg_mask = (infer_patch[0, 1, ...] > thre_1) * (infer_patch[0, 2, ...] < thre_0)
                    infer_patch = fg_mask.detach().cpu().numpy().astype(np.uint8)

                    init_block[..., int(i * d_s): int((i * d_s + d_p)),
                                  int(j * h_s): int((j * h_s + h_p)),
                                  int(k * w_s): int((k * w_s + w_p))] = np.maximum(init_block[..., int(i * d_s): int((i * d_s + d_p)),
                                                                              int(j * h_s): int((j * h_s + h_p)),
                                                                              int(k * w_s): int((k * w_s + w_p))], infer_patch)
          
                    # print("{}/{} successed!".format(patch_num, counts))

    # t2 = time.time()

    # print("All process finished! Time consuming: ", t2-t1, "s")
    init_block = init_block[60:550+60, 120:650+120, 120:650+120]

    return init_block
