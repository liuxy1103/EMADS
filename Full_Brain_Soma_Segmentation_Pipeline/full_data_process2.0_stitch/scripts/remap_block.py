# Function: Remap all blocks
# written by Wei Huang
# 2019/06/27

import numpy as np 
import h5py 
import argparse
import os
import time
import sys 

import multiprocessing

def remap_block(iblock, remap, data_path, record_path):
    print('remap', iblock, flush=True)
    block_split = iblock.split('_')
    iy = block_split[-2]
    iz = block_split[-3]
    filename = iblock + '_relabel.hdf' 
    block = os.path.join(data_path, iz, iy, filename)

    blockf = h5py.File(block, 'r')
    blockdata = blockf['labels'][...]
    blockf.close()
    os.remove(block)
    outf = h5py.File(block, 'w')
    blockdata = remap[0, :].searchsorted(blockdata)
    blockdata = remap[1, blockdata]
    l = outf.create_dataset('labels', blockdata.shape, blockdata.dtype, compression='gzip')
    l[:, :, :] = blockdata
    outf.close()

    f_re = open(os.path.join(record_path, iblock + '.txt'), 'w')
    f_re.write('good')
    f_re.close()
    # print 'Successfully wrote', block

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--input_number', type=int, default=None)
    parser.add_argument('-bp', '--base_path', type=str, default=None)
    args = parser.parse_args()

    time1 = time.time()
    data_path = os.path.join(args.base_path, 'seg_results')
    record_path = os.path.join(args.base_path, 'record', 'remap')
    if not os.path.exists(record_path):
        os.makedirs(record_path)
    # input TXT
    submit_path = os.path.join(args.base_path, 'submit', 'remap')
    f_block_list = open(os.path.join(submit_path, str(args.input_number)+'.txt'), 'r')

    f_lines = f_block_list.readlines()
    block_list = []
    for k in f_lines:
        k = k.strip('\n')
        block_list.append(k)
    f_block_list.close()
    print('The number of blocks:', len(block_list), flush=True)

    mapfile = os.path.join(args.base_path, 'create_global_map.hdf')
    mapf = h5py.File(mapfile, 'r')
    remap = mapf['remap'][...]
    mapf.close()

    print('Begining...', flush=True)
    # num_cores = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(processes=num_cores)

    for block in block_list:
        if os.path.exists(os.path.join(record_path, block+'.txt')):
            print(block + ' is done!', flush=True)
        else:
            # pool.apply_async(remap_block, (block, remap, data_path, record_path,))
            print('remap', block, flush=True)
            block_split = block.split('_')
            iy = block_split[-2]
            iz = block_split[-3]
            filename = block + '_relabel.hdf' 
            block_path = os.path.join(data_path, iz, iy, filename)
            temp_block = os.path.join(data_path, iz, iy, 'temp_' + filename)
            if not os.path.exists(block_path):
                print('There is no', block_path)
                continue
            blockf = h5py.File(block_path, 'r')
            blockdata = blockf['labels'][...]
            blockf.close()

            outf = h5py.File(temp_block, 'w')
            blockdata = remap[0, :].searchsorted(blockdata)
            blockdata = remap[1, blockdata]
            l = outf.create_dataset('labels', blockdata.shape, blockdata.dtype, compression='gzip')
            l[:, :, :] = blockdata
            outf.close()

            os.remove(block_path); os.rename(temp_block, block_path)

            f_re = open(os.path.join(record_path, block + '.txt'), 'w')
            f_re.write('good')
            f_re.close()
    # pool.close()
    # pool.join()

    time2 = time.time()
    print('COST TIME:', (time2-time1), flush=True)