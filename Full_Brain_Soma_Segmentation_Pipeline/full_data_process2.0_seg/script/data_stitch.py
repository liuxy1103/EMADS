# -*- coding: utf-8 -*-
# @Time : 2020/7/24
# @Author : Xiaoyu Liu
# @Email : 1120313281@qq.com
# @File : data_process.py
# @Software: PyCharm
import os
import h5py

import numpy as np
#from tqdm import tqdm
from PIL import Image

def stitch(n,start_z,y,x):

    # special parameters
    shift_z = 14
    shift_xy = 106
    overlap_z = 30
    num_z = 84
    num_block=n #num of the block
    #data path
    base_path_seg = '/braindat/lab/liuxy/soma_seg/reconstruction_v1/seg_results'
    base_path_raw = '/braindat/FAFB/data'
   # begin_z = start_z
#stitch seg
    seg_stitch = np.zeros((186,1836,1836),dtype=np.uint64)
    ite = 0
    end_id_list = []
    for k in range(num_block):
        tmp_z = start_z + k
        block_name = 'raw_without_artifact_' + str(tmp_z) + '_' + str(y) + '_' + str(x) + '.seg.hdf'
        if k == (num_block - 1):
            end = num_z - (shift_z * 2)
        else:
            end = num_z - (shift_z * 2) - overlap_z

        block_path = os.path.join(base_path_seg,str(tmp_z),str(y),block_name)
        if not os.path.exists(block_path):
            for i in range(end):
                seg_stitch[ite,:] = 0
                ite += 1
        else:
            f = h5py.File(block_path)
            seg = f['labels'][:]
            f.close()
            seg = seg[:end]
            for i in range(end):
                seg_stitch[ite,:] = seg[i]
                ite += 1
#stitch raw
    raw_stitch = np.zeros((186,1836,1836),dtype=np.uint8)
    ite = 0
    for k in range(num_block):
        tmp_z = start_z + k
        block_name = 'raw_without_artifact_' + str(tmp_z) + '_' + str(y) + '_' + str(x)

        if k == (num_block - 1):
            end = num_z - shift_z
        else:
            end = num_z - shift_z - overlap_z

        block_path = os.path.join(base_path_raw, str(tmp_z), str(y), block_name)
        if not os.path.exists(block_path):
            for i in range(shift_z, end):
                raw_stitch[ite,:] = 0
                ite += 1
        else:
            for i in range(shift_z, end):
                img_name = os.path.join(block_path, str(i).zfill(4) + '.png')
                if not os.path.exists(img_name):
                    img = np.zeros((2048, 2048), dtype=np.uint8)
                else:
                    img = np.asarray(Image.open(img_name))
                img = img[shift_xy:-shift_xy, shift_xy:-shift_xy]
                raw_stitch[ite, :] = img
                ite += 1

    print('Done')

    return raw_stitch,seg_stitch

def stitch_seg(n,start_z,y,x):

    # special parameters
    shift_z = 14
    shift_xy = 106
    overlap_z = 30
    num_z = 84
    num_block=n #num of the block
    #data path
    base_path_seg = '/braindat/FAFB/reconstruction_v2/seg_results'
   # begin_z = start_z
#stitch seg
    seg_stitch = np.zeros((186,1836,1836),dtype=np.uint64)
    ite = 0
    end_id_list = []
    for k in range(num_block):
        tmp_z = start_z + k
        block_name = 'raw_without_artifact_' + str(tmp_z) + '_' + str(y) + '_' + str(x) + '_relabel.hdf'
        if k == (num_block - 1):
            end = num_z - (shift_z * 2)
        else:
            end = num_z - (shift_z * 2) - overlap_z

        block_path = os.path.join(base_path_seg,str(tmp_z),str(y),block_name)
        if not os.path.exists(block_path):
            for i in range(end):
                seg_stitch[ite,:] = 0
                ite += 1
        else:
            print('have block')
            f = h5py.File(block_path)
            seg = f['labels'][:]
            f.close()
            seg = seg[:end]
            for i in range(end):
                seg_stitch[ite,:] = seg[i]
                ite += 1
    return seg_stitch

def stitch_raw(n,start_z,y,x):

    # special parameters
    shift_z = 14
    shift_xy = 106
    overlap_z = 30
    num_z = 84
    num_block=n #num of the block
    #data path
    base_path_raw = '/braindat/FAFB/data'
   # begin_z = start_z

#stitch raw
    raw_stitch = np.zeros((186,1836,1836),dtype=np.uint8)
    ite = 0
    for k in range(num_block):
        tmp_z = start_z + k
        block_name = 'raw_without_artifact_' + str(tmp_z) + '_' + str(y) + '_' + str(x)

        if k == (num_block - 1):
            end = num_z - shift_z
        else:
            end = num_z - shift_z - overlap_z

        block_path = os.path.join(base_path_raw, str(tmp_z), str(y), block_name)
        if not os.path.exists(block_path):
            for i in range(shift_z, end):
                raw_stitch[ite,:] = 0
                ite += 1
        else:
            for i in range(shift_z, end):
                img_name = os.path.join(block_path, str(i).zfill(4) + '.png')
                if not os.path.exists(img_name):
                    img = np.zeros((2048, 2048), dtype=np.uint8)
                else:
                    img = np.asarray(Image.open(img_name))
                img = img[shift_xy:-shift_xy, shift_xy:-shift_xy]
                raw_stitch[ite, :] = img
                ite += 1

    return raw_stitch
