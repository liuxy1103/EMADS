# -*- coding: utf-8 -*-
# @Time    : 2020/8/23 22:40
# @Author  : Bo Hu
# @Email   : hubosist@mail.ustc.edu.cn
# @File    : merge_func.py
# @Software: PyCharm

import os
import cupy as cp
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np
import time
from skimage import morphology

def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def load_nii(path):
    nii = sitk.ReadImage(path)
    nii = sitk.GetArrayFromImage(nii)
    print(path.split("\\")[-1], "loaded!")
    return nii

def area_of_seg(mask, seg):
    intersection = mask*seg
    iou = cp.sum(intersection)/cp.sum(seg)
    return iou

def remove_holes_2D(mask):
    mask = mask.astype(cp.bool_)
    z_shape = mask.shape[0]
    for i in range(z_shape):
        mask[i] = morphology.remove_small_holes(mask[i], area_threshold=2048, connectivity=1, in_place=False)
    new_mask = (mask.astype(np.uint8) * 255).astype(cp.uint8)
    return new_mask

def merge_cell(cell, cell_list, cell_patch, seg_patch, seg_patch_ids):
    merged_patch = cp.zeros(shape=seg_patch.shape, dtype=cp.uint8)

    id_list = seg_patch_ids
    if len(cell_list) > 2048:
        raise Exception("2048 should be set larger!")

    time0 = time.time()
    for id in tqdm(id_list):
        seg_id = cp.zeros(shape=seg_patch.shape, dtype=cp.uint8)
        seg_id_pos = cp.where(seg_patch == id)
        # seg_id_pos_start = seg_id_pos[0][0]
        # id_start_pos_rate = seg_id_pos_start/seg_patch.shape[0]
        # if id_start_pos_rate < 0.35 or id_start_pos_rate > 0.65:
        #     start_end = True
        # else:
        #     start_end = False
        seg_id[seg_id_pos] = 1
        id_rate = cp.sum(seg_id)/(26*seg_id.shape[1]*seg_id.shape[2])

        area_rate = area_of_seg(cell_patch / 255, seg_id)
        if area_rate > 0.4:  # Don't set too large or the condition will be too hard
            merged_patch[seg_id_pos] = cell

        elif (id_rate > 0.1) and (area_rate > 0.1):  #it is based on an observation that  our prediction is always incomplete at start and end slices and the prediction is basically in the cell boundary. So I loose the condition
            merged_patch[seg_id_pos] = cell

        else:
            pass

    time1 = time.time()
    use_time = time1 - time0
    print("Cell {} finished. It used {} s.".format(cell, use_time))

    return merged_patch

def main_merge(mask, seg):


    z, h, w = mask.shape
    cell_list = np.unique(mask)
    merged_block = cp.zeros(shape=(z, h, w), dtype=np.uint8)
    if len(cell_list) > 5:
        for cell in cell_list[1:]:
            print("\nCurrent cell:", cell)
            time_pre = time.time()
            pos_cell = np.where(mask == cell)

            z_real_start = np.min(pos_cell[0])
            h_real_start = np.min(pos_cell[1])
            w_real_start = np.min(pos_cell[2])

            z_real_end = np.max(pos_cell[0])
            h_real_end = np.max(pos_cell[1])
            w_real_end = np.max(pos_cell[2])

            z_start = z_real_start - 20 if z_real_start > 20 else 0
            h_start = h_real_start - 40 if h_real_start > 40 else 0
            w_start = w_real_start - 40 if w_real_start > 40 else 0

            z_end = z_real_end + 21 if z_real_end < z else z
            h_end = h_real_end + 41 if h_real_end < h else h
            w_end = w_real_end + 41 if w_real_end < w else w

            mask_patch = mask[z_start:z_end, h_start:h_end, w_start:w_end]
            cell_in_mask_patch = np.where(mask_patch == cell)
            cell_patch = np.zeros(shape=mask_patch.shape, dtype=cp.uint8)  # 0~255
            cell_patch[cell_in_mask_patch] = 255  # the current cell only has one value
            cell_patch = cp.asarray(cell_patch)

            seg_patch = seg[z_start:z_end, h_start:h_end, w_start:w_end]
            seg_patch = cp.asarray(seg_patch)
            seg_patch_ids = cp.unique(seg_patch)[1:]

            time_pre_end = time.time()
            print("Iter pre used", time_pre_end - time_pre, "s.")
            merged_patch = merge_cell(cell=cell, cell_list=cell_list, cell_patch=cell_patch, seg_patch=seg_patch,
                                      seg_patch_ids=seg_patch_ids)
            merged_patch_pos = cp.where(merged_patch != 0)
            merged_block[
                merged_patch_pos[0] + z_start, merged_patch_pos[1] + h_start, merged_patch_pos[2] + w_start] = cell

        merged_block = cp.asnumpy(merged_block)
        merged_block = remove_holes_2D(merged_block)

    else:
        print("NO SOMA!")

    return merged_block