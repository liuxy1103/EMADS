# Function: concatenate ids of all sub-blocks
# written by Wei Huang
# 2019/03/27

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import h5py
import argparse
import os
import time

import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--input_number', type=int, default=None)
    parser.add_argument('-bp', '--base_path', type=str, default=None)
    args = parser.parse_args()

    time1 = time.time()
    data_path = os.path.join(args.base_path, 'seg_results')
    used_path = os.path.join(args.base_path, 'record', 'used')

    # input TXT
    submit_path = os.path.join(args.base_path, 'submit', 'concat')
    f_block_list = open(os.path.join(submit_path, str(args.input_number) + '.txt'), 'r')

    f_lines = f_block_list.readlines()
    block_list = []
    for k in f_lines:
        k = k.strip('\n')
        block_list.append(k)
    f_block_list.close()

    # print(block_list)
    print('The number of blocks:', len(block_list), flush=True)

    concat_results = os.path.join(args.base_path, 'concat_results')
    if not os.path.exists(concat_results):
        os.makedirs(concat_results)

    outfile = 'concatenate_joins_' + str(args.input_number) + '.hdf'
    outf = h5py.File(os.path.join(concat_results, outfile), 'w')
    outmerges = np.zeros((0, 2), dtype=np.uint64)

    lock = threading.Lock()


    def read_ids(block):
        global outmerges
        if not os.path.exists(block):
            print('There is no', block)
            return 0
        f = h5py.File(block, 'r')
        assert ('merges' in f) or ('labels' in f)
        if 'merges' in f:
            lock.acquire()
            outmerges = np.vstack((outmerges, f['merges'][...].astype(np.uint64)))
            lock.release()
        if 'labels' in f:
            # write an identity map for the labels
            labels = np.unique(f['labels'][...])
            labels = labels[labels > 0]
            labels = labels.reshape((-1, 1))
            lock.acquire()
            outmerges = np.vstack((outmerges, np.hstack((labels, labels)).astype(np.uint64)))
            lock.release()
        f.close()


    thread_pool = ThreadPoolExecutor(5)

    for block in block_list:
        filename = block + '_relabel.hdf'
        iz = block.split('_')[-3]
        iy = block.split('_')[-2]
        block_path = os.path.join(data_path, iz, iy, filename)
        thread_pool.submit(read_ids, block_path)
    thread_pool.shutdown()

    if outmerges.shape[0] > 0:
        outf.create_dataset('merges', outmerges.shape, outmerges.dtype)[...] = outmerges
    outf.close()
    time2 = time.time()
    print('COST TIME:', (time2 - time1), flush=True)
