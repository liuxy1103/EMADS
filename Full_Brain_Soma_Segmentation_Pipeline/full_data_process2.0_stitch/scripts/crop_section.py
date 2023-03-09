# FUNCTION: Crop raw data in single section with downsampling 4 times for 84X2048X2048 sub-blocks 
# with overlap 100 pixel in x,y direction and 30 pixel in z direction


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os 
from skimage import io 
import numpy as np 
import argparse
import time 
import cv2 
import multiprocessing
import sys 


def find_folder(number, folder_list):
    for folder in folder_list:
        name = folder.split('_')
        if name[0] == 'temca2':
            if number >= int(name[1]) and number <= int(name[2].split('.')[0]):
                return folder
            else:
                continue
    return None


def read_image(file_path, img):
    name = img.split('.')
    if name[-1] == 'png':
        iy = int(name[2])
        ix = int(name[3])
        data = io.imread(os.path.join(file_path, img))
        return [iy, ix, data]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--input_number', type=int, default=None)
    parser.add_argument('-fp', '--fafb_path', type=str, default='/braindat/original')
    parser.add_argument('-bp', '--base_path', type=str, default='/braindat/seg_fafb_2/reconstruction_v2')
    args = parser.parse_args()

    time1 = time.time()
    base = 1
    num_cores = multiprocessing.cpu_count()

    output_path = os.path.join(args.base_path, 'data')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    section_full = os.path.join(args.base_path, 'section_full')
    if not os.path.exists(section_full):
        os.mkdir(section_full)
    record_path = os.path.join(args.base_path, 'record', 'crop_section')
    submit_path = os.path.join(args.base_path, 'submit', 'crop_section')

    # input TXT
    f_block_list = open(os.path.join(submit_path, str(args.input_number)+'.txt'), 'r')
    line = f_block_list.readlines()[0].strip('\n').split(' ')
    start = int(line[0])
    num = int(line[1])
    
    # define block
    # num_block_x = 26   # (pixel_x // times - 3072) / 2560
    # num_block_y = 14   # (pixel_y // times - 3072) / 2560

    # used for block size: 252X3072X3072
    # step_x = 2560      # 2560 = 3072 - (106*2) - (100*2) - 100(overlap)
    # step_y = 2560
    # step_z = 194       # 194 = 252 - (14*2) - 30(overlap)
    # length_x = 3072
    # length_y = 3072
    # length_z = 252     # 252 = (56 * 4) + 28
    # overlap_z = 58


    # used for block size: 84X2048X2048
    step_x = 1736      # 2560 = 2048 - (106*2) - 100(overlap)
    step_y = 1736
    step_z = 26       #  = 84 - (14*2) - 30(overlap)
    length_x = 2048
    length_y = 2048
    length_z = 84     # 84 = (56 * 1) + 28
    overlap_3z = 6    # 6 = 84 - 26 * 3

    # difine the size of canvas
    times = 4
    num_y = 19
    num_x = 35
    pixel_unit = 8192
    pixel_y = num_y * pixel_unit 
    pixel_x = num_x * pixel_unit 
    down_pixel_unit = pixel_unit // times 

    num_block_y = (pixel_y // times - length_y) // step_y       # num_block_y = 14
    num_block_x = (pixel_x // times - length_x) // step_x       # num_block_x = 26

    if (pixel_y // times - length_y) % step_y != 0:
        num_block_y += 1
        pixel_y = (num_block_y * step_y + length_y) * times 
    if (pixel_x // times - length_x) % step_x != 0:
        num_block_x += 1                                       # num_block_x = 27
        pixel_x = (num_block_x * step_x + length_x) * times 
    # section = np.zeros((pixel_y, pixel_x), dtype=np.uint8)     # shape: (155648, 288768)

    down_x = pixel_x // times
    down_y = pixel_y // times
    down_size = 1.0 / times 
    down_section = np.zeros((down_y, down_x), dtype=np.uint8)

    spcial_folder = 'temca2.13.0.intensity_orderfixed'

    path_file = os.listdir(args.fafb_path)
    folder = []
    for i in path_file:
        if os.path.isdir(os.path.join(args.fafb_path, i)):
            folder.append(i)
    folder.sort()
    print('folder:', folder)
    sys.stdout.flush()
    
    def _save_image(bz, iby, ibx, new_start):
        img = down_section[step_y*iby:step_y*iby+length_y, step_x*ibx:step_x*ibx+length_x]
        if np.max(img) != 0:
            block_name = 'raw_without_artifact_' + str(bz) + '_' + str(iby) + '_' + str(ibx)
            block_path = os.path.join(output_path, str(bz), str(iby), block_name)
            io.imsave(os.path.join(block_path, str(new_start % step_z).zfill(4)+'.png'), img)
            if bz > 0:
                block_name_overlap = 'raw_without_artifact_' + str(bz-1) + '_' + str(iby) + '_' + str(ibx)
                block_overlap_path = os.path.join(output_path, str(bz-1), str(iby), block_name_overlap)
                io.imsave(os.path.join(block_overlap_path, str(new_start % step_z + step_z).zfill(4)+'.png'), img)
            if bz > 1:
                block_name_overlap = 'raw_without_artifact_' + str(bz-2) + '_' + str(iby) + '_' + str(ibx)
                block_overlap_path = os.path.join(output_path, str(bz-2), str(iby), block_name_overlap)
                io.imsave(os.path.join(block_overlap_path, str(new_start % step_z + step_z*2).zfill(4)+'.png'), img)
            if bz > 2 and new_start % step_z < overlap_3z:
                block_name_overlap = 'raw_without_artifact_' + str(bz-3) + '_' + str(iby) + '_' + str(ibx)
                block_overlap_path = os.path.join(output_path, str(bz-3), str(iby), block_name_overlap)
                io.imsave(os.path.join(block_overlap_path, str(new_start % step_z + step_z*3).zfill(4)+'.png'), img)

    def load_section(result):
        iy = result[0]
        ix = result[1]
        # section[iy*pixel_unit:(iy+1)*pixel_unit, ix*pixel_unit:(ix+1)*pixel_unit] = result[-1]
        down_image = cv2.resize(result[-1], (down_pixel_unit, down_pixel_unit), interpolation = cv2.INTER_CUBIC)
        down_section[iy*down_pixel_unit:(iy+1)*down_pixel_unit, ix*down_pixel_unit:(ix+1)*down_pixel_unit] = down_image

    record_file = str(start) + '.txt'
    f_re = open(os.path.join(record_path, record_file), 'w')
    for ite in range(start, start + num):
        folder_name = find_folder(ite, folder)
        if folder_name is not None:
            if os.path.exists(os.path.join(args.fafb_path, folder_name, str(ite))):
                file_path = os.path.join(args.fafb_path, folder_name, str(ite))
            elif os.path.exists(os.path.join(args.fafb_path, spcial_folder, str(ite))):
                file_path = os.path.join(args.fafb_path, spcial_folder, str(ite))
            else:
                file_path = None
        else:
            raise AttributeError('{} does not exsit'.format(ite))
        
        print('file_path:', file_path)
        sys.stdout.flush()
        # section[:,:] = 0
        
        # Read images
        # print('read images...')  
        pool = multiprocessing.Pool(processes=num_cores)
        time3 = time.time()
        
        if file_path is not None:
            files = os.listdir(file_path)
            files.sort()
            for file in files:
                pool.apply_async(read_image, (file_path, file,), callback=load_section)
        pool.close()
        pool.join()

        time4 = time.time()
        # print('pool reading time:', (time4-time3))
        # print('end reading...')
        
        section2 = cv2.resize(down_section, (0,0), fx=0.1, fy=0.1, interpolation = cv2.INTER_CUBIC)
        io.imsave(os.path.join(section_full, str(ite).zfill(4)+'_full.png'), section2)
        del section2 

        # Crop section by parallelization
        # print('crop section...')
        # judge z 
        new_start = ite - base
        bz = new_start // step_z

        path2 = os.path.join(output_path, str(bz))
        if not os.path.exists(path2):
            os.mkdir(path2)
        for ibyy in range(num_block_y):
            for ibxx in range(num_block_x):
                block_name = 'raw_without_artifact_' + str(bz) + '_' + str(ibyy) + '_' + str(ibxx)
                block_path = os.path.join(path2, str(ibyy), block_name)
                if not os.path.exists(block_path):
                    os.makedirs(block_path)
                if bz > 0:
                    block_name_overlap = 'raw_without_artifact_' + str(bz-1) + '_' + str(ibyy) + '_' + str(ibxx)
                    block_overlap_path = os.path.join(output_path, str(bz-1), str(ibyy), block_name_overlap)
                    if not os.path.exists(block_overlap_path):
                        os.makedirs(block_overlap_path)
                if bz > 1:
                    block_name_overlap = 'raw_without_artifact_' + str(bz-2) + '_' + str(ibyy) + '_' + str(ibxx)
                    block_overlap_path = os.path.join(output_path, str(bz-2), str(ibyy), block_name_overlap)
                    if not os.path.exists(block_overlap_path):
                        os.makedirs(block_overlap_path)
                if bz > 2 and new_start % step_z < overlap_3z:
                    block_name_overlap = 'raw_without_artifact_' + str(bz-3) + '_' + str(ibyy) + '_' + str(ibxx)
                    block_overlap_path = os.path.join(output_path, str(bz-3), str(ibyy), block_name_overlap)
                    if not os.path.exists(block_overlap_path):
                        os.makedirs(block_overlap_path)
        
        pool = multiprocessing.Pool(processes=num_cores)
        for iby in range(num_block_y):
            for ibx in range(num_block_x):
                pool.apply_async(_save_image, (bz, iby, ibx, new_start,))
        pool.close()
        pool.join()
        # print('end croping...')

        del down_section
        down_section = np.zeros((down_y, down_x), dtype=np.uint8)
        f_re.write(str(ite))
        f_re.write('\n')
    f_re.close()

    time2 = time.time()
    print('COST TIME:', (time2 - time1))
    sys.stdout.flush()
