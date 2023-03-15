# -*- coding: utf-8 -*-
# @Time    : 2020/5/16 12:54
# @Author  : Bo Hu
# @Email   : hubosist@mail.ustc.edu.cn
# @Software: PyCharm

from pathlib import Path
import SimpleITK as sitk

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def log_args(args,log):
    args_info = "\n##############\n"
    for key in args.__dict__:
        args_info = args_info+(key+":").ljust(25)+str(args.__dict__[key])+"\n"
    args_info += "##############"
    log.info(args_info)

def save_nii(img,save_name):
    nii_image = sitk.GetImageFromArray(img)
    name = str(save_name).split("/")
    sitk.WriteImage(nii_image,str(save_name))
    print(name[-1]+" saving finished!")




