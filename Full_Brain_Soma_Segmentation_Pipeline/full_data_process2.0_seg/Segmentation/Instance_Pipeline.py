# -*- coding: utf-8 -*-
# @Time    : 2020/7/23 0:46
# @Author  : Bo Hu
# @Email   : hubosist@mail.ustc.edu.cn
# @File    : Instance_Pipeline.py
# @Software: PyCharm

import os
import time
import numpy as np
import SimpleITK as sitk
from skimage import morphology, measure

def save_nii(img, path):
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img, path)
    print(path.split('\\')[-1], "saving succeed!")

def remove_holes_2D(mask):
    mask = mask.astype(np.bool_)
    z_shape = mask.shape[0]
    for i in range(z_shape):
        mask[i] = morphology.remove_small_holes(mask[i], area_threshold=2048, connectivity=1, in_place=False)
    new_mask = (mask.astype(np.uint8) * 255).astype(np.uint8)
    return new_mask

def remove_smalls_2D(mask):
    for i in range(mask.shape[0]):
        new_mask = morphology.remove_small_objects(mask[i].astype(np.bool_), min_size=1024, connectivity=1, in_place=False)
        mask[i] = (new_mask.astype(np.uint8) * 255).astype(np.uint8)
    return mask

def remove_smalls(mask, min_size=20480):
    mask = mask.astype(np.bool_)
    new_mask = morphology.remove_small_objects(mask, min_size=min_size, connectivity=1, in_place=False)
    new_mask = (new_mask.astype(np.uint8) * 255).astype(np.uint8)
    return new_mask

def instance(semantic_output):
    # Step1 #
    # start_time = time.time()

    semantic_output[semantic_output > 0.97] = 255
    semantic_output[semantic_output <= 0.97] = 0
    thre_mask = remove_holes_2D(semantic_output.astype(np.uint8)) ####Not sure whether it should be here
    thre_mask = thre_mask.astype(np.uint8)
    # s1_time = time.time()
    # print("Step 1 uses:", s1_time - start_time, "s.\n")
    # Step1 #

    # Step2 #
    removed_both2Dand3D_mask = remove_smalls_2D(thre_mask)
    removed_both2Dand3D_mask = remove_smalls(removed_both2Dand3D_mask, min_size=102400)
    # s2_time = time.time()
    # print("Step 2 uses:", s2_time - s1_time, "s.\n")
    # Step2 #

    # Step3 #
    result = measure.label(removed_both2Dand3D_mask.astype(bool), connectivity=1)
    result = result.astype(np.uint8)
    # result_path = os.path.join(root_path, block_name, "Step3_result.nii.gz")
    # if not os.path.exists(result_path):
    #     os.makedirs(result_path)
    # save_nii(result.astype(np.uint8), result_path)
    # s3_time = time.time()
    # print("Step 3 uses:", s3_time - s2_time, "s.\n")
    # Step3 #
    return result


