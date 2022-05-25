# -*- coding: utf-8 -*-
# @Time    : 2020/5/17 下午7:36
# @Author  : Bo Hu
# @Email   : hubosist@mail.ustc.edu.cn
# @File    : predict.py
# @Software: PyCharm

import torch
import argparse
import numpy as np
import torch.nn as nn
from model.metric import dc_score
from pathlib import Path
from model.model import UNet3D
from data.dataset import CellSeg_set
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from utils.utils import save_nii, ensure_dir

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--save_dir",type=str,default="/ghome/hubo/CODE/BCE3_tri/prediction")
parser.add_argument("-c", "--ckp_file",type=str,default="/ghome/hubo/CODE/BCE3_tri/ckps/checkpoint-epoch10.pth")
parser.add_argument("-d", "--data_dir",type=str,default="/gdata/hubo/DATASET/CellSeg_v3.0")
parser.add_argument("-r", "--resample",type=str,default=(1,0.25,0.25))
args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

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
# model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.ckp_file).items()})


valset = CellSeg_set(dir=args.data_dir, mode="validation")
val_loader = DataLoader(valset, batch_size=1,shuffle=False)
dice_list = []
if __name__ == '__main__':
    print(args.ckp_file)
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.to(device)
            data = interpolate(data, scale_factor=args.resample, mode="trilinear", align_corners=True)
            target = target.squeeze(0).squeeze(0).cpu().numpy()
            output = model(data)
            output = interpolate(output, scale_factor=(args.resample[0],1/args.resample[1],1/args.resample[2]), mode="trilinear", align_corners=True)
            output = torch.argmax(output,dim=1)
            output = output.squeeze(0).cpu().numpy().astype(np.uint8)*255#*255
            dice_score = dc_score(output,target)
            print("DICE:",dice_score)
            if dice_score > 0.2:
                dice_list.append(dc_score(output,target))
            target_path = Path(args.save_dir,str(batch_idx).zfill(3)+"_label.nii.gz")
            output_path = Path(args.save_dir,str(batch_idx).zfill(3)+"_output.nii.gz")
            save_nii(target, target_path)
            save_nii(output, output_path)

    print("\nMEAN DICE:", np.mean(dice_list))
    print(dice_list)
    print("Finished!")









