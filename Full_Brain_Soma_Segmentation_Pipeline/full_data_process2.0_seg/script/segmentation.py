import sys
sys.path.insert(1, '/braindat/lab/liuxy/soma_seg/Code_Full_Brain_V2_seed/full_data_process2.0_seg')
import torch
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from data_stitch import stitch_raw, stitch_seg
import time
import os
import h5py
from Segmentation.predict_sementic import *
from Segmentation.Instance_Pipeline import *
from Localization.predict_anchors import *

model_state_file = "/braindat/lab/liuxy/soma_seg/Code_Full_Brain_V2_seed/full_data_process2.0_seg/patch_model_v2.pth"
ckps = '/braindat/lab/liuxy/soma_seg/Code_Full_Brain_V2_seed/full_data_process2.0_seg/seed_model_v4.7.pth'
config_path = "/braindat/lab/hubo/CODE/Cell_Detection_Code/Mingxing_Code/Config/mysetting.yaml"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--input_number', type=int, default=0,help='the sequence number of tasks ')
    parser.add_argument('-bp', '--base_path', type=str, default='/braindat/lab/liuxy/soma_seg/Code_Full_Brain_V2_seed/full_data_process2.0_seg')
    #parser.add_argument('-nz', '--num_z', type=int, default=None)
    parser.add_argument('-ny', '--num_y', type=int, default=22)
    parser.add_argument('-nx', '--num_x', type=int, default=41)
    # parser.add_argument('-nu', '--num', type=int, default=84)
    args = parser.parse_args()
    time1 = time.time()

    #param
    #num_z = args.num_z
    num_y = args.num_y
    num_x = args.num_x

    submit_path = os.path.join(args.base_path, 'submit')
    f_block_list = open(os.path.join(submit_path, str(args.input_number) + '.txt'), 'r')
    f_block_lines = f_block_list.readlines()
    block_list = [f_block_lines[i].strip('\n').split(' ')[0] for i in range(len(f_block_lines))]
    block_list_flag = {f_block_lines[i].strip('\n').split(' ')[0]:f_block_lines[i].strip('\n').split(' ')[1] for i in range(len(f_block_lines))}


    seg_flag=0 # judge if there is an activate block in successive six blocks
    z_range = list(set([int(block_list[i].split('_')[-3]) for i in range(len(block_list))]))
    first_z = min(z_range)
    last_z = max(z_range)
    range_z = np.arange(first_z,last_z+1,6) #the last one is not up to 6
    #range_z = np.append(range_z,range_z[-1]+6)
    flag_edge = 0

    record_soma_path = os.path.join(args.base_path, 'soma_record')
    if not os.path.exists(record_soma_path):
        os.makedirs(record_soma_path)
    #f_st = open(os.path.join(record_soma_path, str(args.input_number) + '.txt'), 'w')
    for start_z in range_z:
        if start_z == range_z[-1]:
            flag_edge =1

        if flag_edge == 1:
            n = last_z - range_z[-1]+1
        else:
            n = 6

        for iy in range(num_y):
            for ix in range(num_x):
                for iz in range(n):
                    iz = iz + start_z
                    block_name = 'raw_without_artifact_' + str(iz) + '_' + str(iy) + '_' + str(ix)
                    if block_list_flag[block_name] == '1':
                        # the number of blocks is maybe more than 6
                        raw_stitch = stitch_raw(n, start_z, iy, ix)
                        # print('saving raw_stitch')
                        # out_path = os.path.join('./stitch_raw_out', str(start_z), str(iy))
                        # if not os.path.exists(out_path):
                        #     os.makedirs(out_path)
                        # out = h5py.File(os.path.join(out_path, 'stitch_raw_'+str(start_z) + '_' + str(iy) + '_' + str(ix)+'.hdf'), 'w')
                        # out.create_dataset('soma', shape=raw_stitch.shape, dtype=raw_stitch.dtype, compression="gzip")[...] = raw_stitch
                        # out.close()

                        f_st = open(os.path.join(record_soma_path, str(args.input_number) + '.txt'), 'a')
                        f_st.write('stitch_raw_'+str(start_z) + '_' + str(iy) + '_' + str(ix))
                        f_st.write('\t')  
                        #f_st.write('\n')
                        #f_st.close()
                        time0=time.time()
                        seed_map = anchors(ckps, raw_stitch) 
                        print('finished seeds')
                        # out_path = os.path.join('./seed_out', str(start_z), str(iy))
                        # if not os.path.exists(out_path):
                        #     os.makedirs(out_path)
                        # out = h5py.File(os.path.join(out_path, 'soma_detect_'+str(start_z) + '_' + str(iy) + '_' + str(ix)+'.hdf'), 'w')
                        # out.create_dataset('soma', shape=seed_map.shape, dtype=seed_map.dtype, compression="gzip")[...] = seed_map
                        # out.close()
                        sementic_out = main_predict(config_path=config_path, block=raw_stitch,seed_map = seed_map, model_state_file=model_state_file)
                        instance_output = instance(semantic_output=sementic_out)
                        print('finish:','stitch_raw_'+str(start_z) + '_' + str(iy) + '_' + str(ix),time.time()-time0)
                        if not (instance_output==0).all():
                            f_st.write('1')
                            print('there are cells','stitch_raw_'+str(start_z) + '_' + str(iy) + '_' + str(ix))
                        else:
                            print('there are no  cells','stitch_raw_'+str(start_z) + '_' + str(iy) + '_' + str(ix))
                            f_st.write('0')
                        f_st.write('\n')
                        f_st.close()
                        out_path = os.path.join('./soma_out', str(start_z), str(iy))
                        if not os.path.exists(out_path):
                            os.makedirs(out_path)
                        out = h5py.File(os.path.join(out_path, 'soma_detect_'+str(start_z) + '_' + str(iy) + '_' + str(ix)+'.hdf'), 'w')
                        out.create_dataset('soma', shape=instance_output.shape, dtype=instance_output.dtype, compression="gzip")[...] = instance_output
                        out.close()
                        break
    #f_st.close()
    time2 = time.time()
    print('COST TIME:', (time2 - time1))