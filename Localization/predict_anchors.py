# -*- coding: utf-8 -*-
# @Time    : 2022/4/5 
# @Author  : Xiaoyu Liu
# @Email   : liuxyu@mail.ustc.edu.cn
# @Software: PyCharm

import torch
import argparse
import numpy as np
from .model.model import UNet3D
from .utils.utils import log_args
from .data.dataset import CellSeg_set
from torch.utils.data import DataLoader
from .utils.logger import Log
from .utils.utils import ensure_dir
from tensorboardX import SummaryWriter
from torch.nn.functional import interpolate
from .model.metric import dc_score, MeanIoU
from .model.loss import MSE_loss, DiceLoss, BCE_loss #引入不同loss
import os
import SimpleITK as sitk
import os
from copy import deepcopy
# import mahotas
from scipy.ndimage.interpolation import zoom
import skimage.morphology



def anchors(ckp_file,block):
  model = UNet3D()
  net_dict = model.state_dict()
  pretrain = torch.load(ckp_file, map_location='cuda:0')
  #pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
  pretrain_dict = {k.replace('module.',''): v for k, v in pretrain['state_dict'].items() if k.replace('module.','') in net_dict.keys()} #多卡训练的参数
  net_dict.update(pretrain_dict)
  model.load_state_dict(net_dict)
  device = torch.device("cuda:0")
  model.to(device)
  model.eval()
  # block = np.pad(block, pad_width=((6, 6), (48, 48), (48, 48)), mode='reflect')
  # print('block shape:',block.shape)
  data = zoom(block,zoom=(1/3, 1/4, 1/4), order=3)

  data = torch.from_numpy(data).to(device)
  data = data.unsqueeze(0).unsqueeze(0).float()
  # .float()
  scale = (0.5, 1, 1)
  data = interpolate(data, scale_factor=scale, mode="trilinear", align_corners=True)
  # print('downsample block into:',data.shape)
  output = model(data)
  output = interpolate(output, scale_factor=(1/scale[0], 1/scale[1], 1/scale[2]), mode="nearest")
  # output_tmp = output[0,1].detach().cpu().numpy()
  output = torch.argmax(output, dim=1)# two channels
  output = output.squeeze(0).cpu().numpy().astype(np.uint8)   # *255
  seed_map = skimage.morphology.label(output)
  # seed_map =  zoom(seed_map,zoom=(3, 4, 4), order=0)
  # seed_map = seed_map[6:186+6, 48:1836+48, 48:1836+48]
  # print(seed_map.max())
  # print('upsample seed_map into:',seed_map.shape)
  # output =  zoom(output,zoom=(3, 4, 4), order=0)
  # output = output[6:186+6, 48:1836+48, 48:1836+48]

  return seed_map
  # ,output,output_tmp









