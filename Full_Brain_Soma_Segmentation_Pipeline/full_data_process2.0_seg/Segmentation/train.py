# -*- coding: utf-8 -*-
# @Time    : 2022/4/8 
# @Author  : Xiaoyu Liu
# @Email   : liuxyu@mail.ustc.edu.cn
# @Software: PyCharm
import torch
import argparse
import numpy as np
from solver import Solver
from model.model import UNet3D
from utils.utils import log_args
from data.dataset import CellSeg_set
from torch.utils.data import DataLoader
from utils.logger import Log

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch",type=int,default=1)
parser.add_argument("-g", "--gpu_nums",type=int,default=1)
parser.add_argument("-e", "--epochs",type=int,default=500)
parser.add_argument("-r", "--lr",type=float,default=1e-4)
parser.add_argument("-p", "--lr_patience",type=int,default=50)
parser.add_argument("-n", "--network",type=str,default="UNet3D()")
parser.add_argument("-t", "--loss_type",type=str,default="BCE_loss")
parser.add_argument("-d", "--data_dir",type=str,default="/braindat/lab/liuxy/soma_seg/CellSeg_v3.0")
parser.add_argument("-l", "--logs_dir",type=str,default="./logs")
parser.add_argument("-c", "--ckps_dir",type=str,default="./ckps")
parser.add_argument("-s", "--resample",type=tuple,default=(1, 0.25, 0.25),help="resample rate:(z,h,w)")
parser.add_argument("-w", "--weight_rate",type=list,default=[0.1, 0.9])
parser.add_argument("-x", "--resume",type=bool,default=False)
parser.add_argument("-y", "--resume_path",type=str,default=None)
parser.add_argument("-z", "--tolerate_shape",type=tuple,default=(128, 384, 384))
args = parser.parse_args()

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)
log = Log()
if __name__ == '__main__':
    gpus = args.gpu_nums
    model = eval(args.network)
    criterion = args.loss_type
    metric = "dc_score"
    batch_size = args.batch
    epochs = args.epochs
    lr = args.lr
    trainset = CellSeg_set(dir=args.data_dir,mode="train")
    valset = CellSeg_set(dir=args.data_dir,mode="validation")
    train_loader = DataLoader(trainset,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(valset,batch_size=1,shuffle=False)
    logs_dir = args.logs_dir
    patience = args.lr_patience
    checkpoint_dir = args.ckps_dir
    scale = args.resample
    weight = args.weight_rate
    resume = args.resume
    resume_path = args.resume_path
    tolerate_shape = args.tolerate_shape

    log_args(args, log)

    solver = Solver(gpus=gpus,model=model,criterion=criterion,metric=metric,batch_size=batch_size,
                    epochs=epochs,lr=lr,trainset=trainset,valset=valset,train_loader=train_loader,
                    val_loader=val_loader,logs_dir=logs_dir,patience=patience,
                    checkpoint_dir=checkpoint_dir,scale=scale,weight=weight, resume=resume,resume_path=resume_path,
                    log=log, tolerate_shape=tolerate_shape)

    solver.train()










