# Function: Create global map
# written by Wei Huang
# 2019/03/27

import numpy as np 
import h5py 
import argparse
import os
import time
import multiprocessing


def read_concat(concat_results, number):
    concat_file = 'concatenate_joins_' + str(number) + '.hdf'
    print('read: ' + concat_file, flush=True)
    concat_path = os.path.join(concat_results, concat_file)
    f = h5py.File(concat_path, 'r')
    merges = f['merges'][...]
    f.close()
    return merges

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--number_task', type=int, default=None)
    parser.add_argument('-bp', '--base_path', type=str, default=None)
    args = parser.parse_args()

    time1 = time.time()
    concat_results = os.path.join(args.base_path, 'concat_results')
    outfile = os.path.join(args.base_path, 'create_global_map.hdf')

    # # numltiprocessing
    # # all_merges = {}
    # outmerges = np.zeros((0, 2), dtype=np.uint64)
    # def load_result(results):
    #     outmerges = np.vstack((outmerges, results))
    
    # num_cores = 4
    # pool = multiprocessing.Pool(processes=num_cores)
    # for i in range(args.number_task):
    #     pool.apply_async(read_concat, (concat_results, i,), callback=load_result)
    # pool.close()
    # pool.join()
    
    # # single processing
    outmerges = np.zeros((0, 2), dtype=np.uint64)
    for i in range(args.number_task):
        concat_file = 'concatenate_joins_' + str(i) + '.hdf'
        print('read: ' + concat_file, flush=True)
        concat_path = os.path.join(concat_results, concat_file)
        f = h5py.File(concat_path, 'r')
        merges = f['merges'][...]
        f.close()
        outmerges = np.vstack((outmerges, merges))


    remap = {}
    next_label = 1
    # put every pair in the remap
    print('put every pair in the remap')
    for v1, v2 in outmerges:
        remap.setdefault(v1, v1)
        remap.setdefault(v2, v2)
        while v1 != remap[v1]:
            v1 = remap[v1]
        while v2 != remap[v2]:
            v2 = remap[v2]
        if v1 > v2:
            v1, v2 = v2, v1
        remap[v2] = v1

    # pack values - every value now either maps to itself (and should get its
    # own label), or it maps to some lower value (which will have already been
    # mapped to its final value in this loop).
    print('pack values')
    remap[0] = 0
    for v in sorted(remap.keys()):
        if v == 0:
            continue
        if remap[v] == v:
            remap[v] = next_label
            next_label += 1
        else:
            remap[v] = remap[remap[v]]

    # write to hdf5 - needs to be sorted for remap to use searchsorted()
    print('write to hdf5')
    remap = sorted(remap.items(), key=lambda remap:remap[0])
    remap = np.array(remap, dtype=np.uint64)
    remap = np.transpose(remap)
    outf = h5py.File(outfile, 'w')
    outf.create_dataset('remap', data=remap, dtype=np.uint64)
    # for idx, v in enumerate(sorted(remap.keys())):
    #     ds[:, idx] = [v, remap[v]]
    outf.close()

    # print "Successfully wrote", outfile
    time2 = time.time()
    print('COST TIME:', (time2-time1))