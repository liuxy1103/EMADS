# FUNCTION: Pairwise match for multi machine
# Written by Wei Huang(weih527@mail.ustc.edu.cn)
# 2019/06/19

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import h5py
import argparse
import os
import time
import fast64counter
import copy
import sys
import pickle

import multiprocessing


def stitching_blocks(in_block1, in_block2, input_path, direction, halo_size, record_path):
    ############################
    # auto_join_pixels=20000
    # minoverlap_pixels=2000
    # minoverlap_dual_ratio=0.7
    # minoverlap_single_ratio=0.9

    # for 288 stitching
    # auto_join_pixels=2000000
    # minoverlap_pixels=2000              
    # minoverlap_dual_ratio=0.9         
    # # minoverlap_single_ratio=0.95 

    # change for reconstruction_v8, dowmsample x4
    auto_join_pixels = 400000
    minoverlap_pixels = 500
    minoverlap_dual_ratio = 0.3
    minoverlap_single_ratio = 0.4
    #############################

    iz1 = in_block1.split('_')[-3]
    iy1 = in_block1.split('_')[-2]
    inblock1 = os.path.join(input_path, iz1, iy1, in_block1 + '_relabel.hdf')
    iz2 = in_block2.split('_')[-3]
    iy2 = in_block2.split('_')[-2]
    inblock2 = os.path.join(input_path, iz2, iy2, in_block2 + '_relabel.hdf')
    if not os.path.exists(inblock1):
        print('There is no', inblock1)
        # inblock1 = os.path.join('/braindat/lab/soma_detect/full_data_process2.0_stitch/soma_out',
        #                         iz1, iy1, in_block1 + '.hdf')
        return 0

    if not os.path.exists(inblock2):
        print('There is no', inblock2)
        # inblock2 = os.path.join('/braindat/lab/soma_detect/full_data_process2.0_stitch/soma_out',
        #                         iz2, iy2, in_block2 + '.hdf')
        return 0
    # print 'Stitching ' + inblock1 + '  and  ' + inblock2
    # read inputfile
    bl1f = h5py.File(inblock1, 'r')
    block1 = bl1f['labels'][...]
    if 'merges' in bl1f:
        previous_merges1 = bl1f['merges'][...]
    else:
        previous_merges1 = []
    bl1f.close()

    bl2f = h5py.File(inblock2, 'r')
    block2 = bl2f['labels'][...]
    if 'merges' in bl2f:
        previous_merges2 = bl2f['merges'][...]
    else:
        previous_merges2 = []
    bl2f.close()

    assert block1.size == block2.size

    #################################
    # block1 = block1 + addval_dict[in_block1]
    # block2 = block2 + addval_dict[in_block2]
    #################################

    stacked = np.vstack((block1, block2))
    # print(np.unique(block1),np.unique(block2))
    inverse, packed = np.unique(stacked, return_inverse=True)
    packed = packed.reshape(stacked.shape)
    packed_block1 = packed[:block1.shape[0], :, :]
    packed_block2 = packed[block1.shape[0]:, :, :]

    lo_block1 = [0, 0, 0]
    hi_block1 = [None, None, None]
    lo_block2 = [0, 0, 0]
    hi_block2 = [None, None, None]

    direction = direction - 1
    lo_block1[direction] = - 2 * halo_size
    hi_block2[direction] = 2 * halo_size

    block1_slice = tuple(slice(l, h) for l, h in zip(lo_block1, hi_block1))
    block2_slice = tuple(slice(l, h) for l, h in zip(lo_block2, hi_block2))
    packed_overlap1 = packed_block1[block1_slice]
    packed_overlap2 = packed_block2[block2_slice]
    # print "block1", block1_slice, packed_overlap1.shape
    # print "block2", block2_slice, packed_overlap2.shape

    counter = fast64counter.ValueCountInt64()
    counter.add_values_pair32(packed_overlap1.astype(np.int32).ravel(), packed_overlap2.astype(np.int32).ravel())
    overlap_labels1, overlap_labels2, overlap_areas = counter.get_counts_pair32()
    # print('overlap_labels1', overlap_labels1)
    # print('overlap_labels2', overlap_labels2)
    # print('overlap_areas', overlap_areas)
    areacounter = fast64counter.ValueCountInt64()
    areacounter.add_values(packed_overlap1.ravel())
    areacounter.add_values(packed_overlap2.ravel())
    areas = dict(zip(*areacounter.get_counts()))

    to_merge = []
    to_steal = []
    for l1, l2, overlap_area in zip(overlap_labels1, overlap_labels2, overlap_areas):
        # print(l1,l2,overlap_area)
        if l1 == 0 or l2 == 0:
            continue
        if ((overlap_area > auto_join_pixels) or
                ((overlap_area > minoverlap_pixels) and
                 ((overlap_area > minoverlap_single_ratio * areas[l1]) or
                  (overlap_area > minoverlap_single_ratio * areas[l2]) or
                  ((overlap_area > minoverlap_dual_ratio * areas[l1]) and
                   (overlap_area > minoverlap_dual_ratio * areas[l2]))))):
            if inverse[l1] != inverse[l2]:
                # print "Merging segments {0} and {1}.".format(inverse[l1], inverse[l2])
                to_merge.append((inverse[l1], inverse[l2]))
        else:
            # print "Stealing segments {0} and {1}.".format(inverse[l1], inverse[l2])
            to_steal.append((overlap_area, l1, l2))
    print('all merge-pairs: ',to_merge)
    ###### change 2019/05/13, delete minoverlap_single_ratio
    # to_merge = []
    # to_steal = []
    # for l1, l2, overlap_area in zip(overlap_labels1, overlap_labels2, overlap_areas):
    #     if l1 == 0 or l2 == 0:
    #         continue
    #     if ((overlap_area > auto_join_pixels) or
    #         ((overlap_area > minoverlap_pixels) and
    #         ((overlap_area > minoverlap_dual_ratio * areas[l1]) and
    #         (overlap_area > minoverlap_dual_ratio * areas[l2])))):
    #         if inverse[l1] != inverse[l2]:
    #             # print "Merging segments {0} and {1}.".format(inverse[l1], inverse[l2])
    #             to_merge.append((inverse[l1], inverse[l2]))
    #     else:
    #         # print "Stealing segments {0} and {1}.".format(inverse[l1], inverse[l2])
    #         to_steal.append((overlap_area, l1, l2))

    merge_map = dict(reversed(sorted(s)) for s in to_merge)
    for idx, val in enumerate(inverse):
        if val in merge_map:
            while val in merge_map:
                val = merge_map[val]
            inverse[idx] = val

    temp_block1 = os.path.join(input_path, iz1, iy1, 'temp-' + in_block1 + '_relabel.hdf')
    temp_block2 = os.path.join(input_path, iz2, iy2, 'temp-' + in_block2 + '_relabel.hdf')
    out1 = h5py.File(temp_block1, 'w')
    out2 = h5py.File(temp_block2, 'w')
    outblock1 = out1.create_dataset('/labels', block1.shape, block1.dtype, compression='gzip')
    outblock2 = out2.create_dataset('/labels', block2.shape, block2.dtype, compression='gzip')
    outblock1[...] = inverse[packed_block1]
    outblock2[...] = inverse[packed_block2]

    # copy any previous merge tables from block 1 to the new output and merge
    if len(previous_merges1) > 0:
        if len(to_merge):
            merges1 = np.vstack((previous_merges1, to_merge))
        else:
            merges1 = previous_merges1
    else:
        merges1 = np.array(to_merge).astype(np.uint64)

    if merges1.size > 0:
        out1.create_dataset('/merges', merges1.shape, merges1.dtype)[...] = merges1

    # copy any previous merge tables from block 2 to the new output
    if len(previous_merges2) > 0:
        out2.create_dataset('/merges', previous_merges2.shape, previous_merges2.dtype)[...] = previous_merges2

    out1.close()
    out2.close()

    os.remove(inblock1);
    os.rename(temp_block1, inblock1)
    os.remove(inblock2);
    os.rename(temp_block2, inblock2)

    f_re = open(os.path.join(record_path, in_block1 + '.txt'), 'w')
    f_re.write('good')
    f_re.close()
    # print "Successfully wrote", inblock1, 'and', inblock2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--input_number', type=int, default=None)
    parser.add_argument('-bp', '--base_path', type=str, default=None)
    args = parser.parse_args()

    time1 = time.time()
    data_path = os.path.join(args.base_path, 'seg_results')
    record_path = os.path.join(args.base_path, 'record', 'stitching')
    # input TXT
    submit_path = os.path.join(args.base_path, 'submit', 'stitching')
    f_block_list = open(os.path.join(submit_path, str(args.input_number) + '.txt'), 'r')

    f_lines = f_block_list.readlines()
    block_list = []
    for k in f_lines:
        k = k.strip('\n')
        block_list.append(k)
    f_block_list.close()

    # print(block_list)
    print('The number of blocks:', len(block_list))
    sys.stdout.flush()

    for block in block_list:
        split_name = block.split(' ')
        block1 = split_name[0]
        block2 = split_name[1]
        direction = int(split_name[2])
        halo_size = int(split_name[3])
        if os.path.exists(os.path.join(record_path, block1 + '.txt')):
            print('stitching ' + block1 + ' and ' + block2 + ' is done!')
            sys.stdout.flush()
        else:
            print('*' * 40 + 'Stitching ' + block1 + ' and ' + block2 + '*' * 40)
            sys.stdout.flush()
            stitching_blocks(block1, block2, data_path, direction, halo_size, record_path)

    time2 = time.time()
    print('COST TIME:', (time2 - time1))
    sys.stdout.flush()
