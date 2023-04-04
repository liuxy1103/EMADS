# -*- coding: utf-8 -*-
# @Time    : 2020/8/24 9:22
# @Author  : Bo Hu
# @Email   : hubosist@mail.ustc.edu.cn
# @File    : test_merge.py
# @Software: PyCharm

import os
import cupy as cp
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np
import time
from skimage import morphology
from merge_func1 import merge_func

block_name = "z-3500_y-21034_x-24866_1"
root_folder = os.path.join("/braindat/lab/hubo/DATASET/FAFB_data/test_set/testset", block_name)
id_save_root_folder = os.path.join(root_folder, "Cupy_Merged")
seg_file = os.path.join(root_folder, "Test_Results", block_name + "_seg.nii.gz")
mask_file = os.path.join(root_folder, "Step3_result.nii.gz")

def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def load_nii(path):
    nii = sitk.ReadImage(path)
    nii = sitk.GetArrayFromImage(nii)
    print(path.split("\\")[-1], "loaded!")
    return nii

def save_nii(img,path):
    print(img.shape)
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img, path)
    print(path.split("\\")[-1], "saving succeed!")

if __name__ == '__main__':
    check_dir(id_save_root_folder)

    seg = load_nii(seg_file)
    mask = load_nii(mask_file)

    merged = merge_func.main_merge(mask, seg)

    save_nii(merged, "/braindat/lab/hubo/TEMP/test_merged.nii.gz")

