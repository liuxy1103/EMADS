# FUNCTION: Adjust the value of every sub-blocks
# Written by Wei Huang(weih527@mail.ustc.edu.cn)
# 2019/07/02

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import h5py
import argparse
import os
import sys
import time


def add_value(block, data_path, dst_folder, value, record_path):
    iz = block.split('_')[-3]
    iy = block.split('_')[-2]
    inblock_path = os.path.join(data_path, iz, iy, block + '.hdf')
    f = h5py.File(inblock_path, 'r')
    seg = f['soma'][:]
    seg = seg.astype(np.uint64)
    f.close()

    seg += value
    if seg.min()<value:
        print('MayBe error',seg.min(),value)
    seg[seg == value] = 0
    outblock_path = os.path.join(dst_folder, iz, iy, block + '_relabel.hdf')
    f2 = h5py.File(outblock_path, 'w')
    f2.create_dataset('labels', data=seg, dtype=np.uint64, compression='gzip')
    f2.close()

    f_re = open(os.path.join(record_path, block + '.txt'), 'w')
    f_re.write('good')
    f_re.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--input_number', type=int, default=None)
    parser.add_argument('-bp', '--base_path', type=str, default=None)
    args = parser.parse_args()

    time1 = time.time()
    # data_path = os.path.join(args.base_path, 'seg_results')
    src_folder = os.path.join(args.base_path, 'soma_out')
    dst_folder = os.path.join(args.base_path, 'seg_results')
    record_path = os.path.join(args.base_path, 'record', 'add_ids')
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    if not os.path.exists(record_path):
        os.makedirs(record_path)

    # input TXT
    submit_path = os.path.join(args.base_path, 'submit', 'add_ids')
    f_block_list = open(os.path.join(submit_path, str(args.input_number) + '.txt'), 'r')

    f_lines = f_block_list.readlines()
    block_list = []
    for k in f_lines:
        k = k.strip('\n')
        block_list.append(k)
    f_block_list.close()

    # print(block_list)
    print('The number of blocks:', len(block_list), flush=True)

    add_val = os.path.join(args.base_path, 'record', 'used')
    addval_path = os.path.join(add_val, 'addval_list.txt')
    f_add = open(addval_path, 'r')
    f_add_lines = f_add.readlines()
    block_name_list = []
    addval_list = []
    for k in f_add_lines:
        k = k.strip('\n')
        block_name_list.append(k.split(' ')[0])
        addval_list.append(int(k.split(' ')[1]))
    f_add.close()
    addval_dict = dict(zip(block_name_list, addval_list))

    for block in block_list:
        # print(addval_dict)
        # print(addval_dict['soma_detect_0_10_23'])
        if os.path.exists(os.path.join(record_path, block + '.txt')):
            print('add_ids ' + block + ' is done!', flush=True)
        else:
            print('*' * 40 + ' add_ids ' + block + ' ' + '*' * 40, flush=True)
            value = addval_dict[block]
            add_value(block, src_folder, dst_folder, value, record_path)
    time2 = time.time()
    print('COST TIME:', (time2 - time1), flush=True)
