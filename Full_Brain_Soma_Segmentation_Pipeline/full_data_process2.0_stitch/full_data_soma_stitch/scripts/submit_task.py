from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
#from memory_profiler import profile
import numpy as np 
import time 
import os 
import argparse
import shutil
import requests
import re
import logging
import subprocess
import pdb
import h5py


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaiToken:
    #__metaclass__ = Singleton
    def __init__(self, userConfig, tokenServerUrl):
        self.userConfig = userConfig
        self.tokenServerUrl = tokenServerUrl
        self.__token = self.__updateToken(userConfig, tokenServerUrl)

    def __updateToken(self, userConfig, tokenServerUrl):
        result = requests.post(tokenServerUrl, userConfig).json()
        #pdb.set_trace()
        return result['token']

    def updateToken(self):
        self.__token = self.__updateToken(self.userConfig, self.tokenServerUrl)

    def getToken(self):
        return self.__token


def get_token(token_config, server_url):
    job_server_url = server_url + '/rest-server/api/v1/jobs'
    token_server_url = server_url + '/rest-server/api/v1/token'
    logger.info("get_jobs_list {} {}".format(job_server_url, token_server_url))
    paiToken = PaiToken(token_config, token_server_url)
    return paiToken.getToken()


def task_run(task_name, lines, time_stamp, base_path, json_path, log_path, num_tasks, depth):
    pwd = os.getcwd()
    token = get_token(token_config, server_url)
    ite = 0
    for i in range(0, num_tasks):
        name = task_name + '_' + str(i) + '_' + time_stamp
        fname = os.path.join(json_path, name + '.json')
        temp = lines
        row1 = temp[1].split('"')
        row1[3] = name
        temp[1] = '"'.join(row1)

        row15 = temp[15].split('"')
        row15[3] = 'cd ' + pwd + ';./tasks.sh ' + \
                    task_name + ' ' + \
                    str(i) + ' ' + \
                    base_path + ' ' + \
                    str(depth) + \
                    ' 2>&1 | tee ' + log_path + '/' + \
                    task_name + '_' + str(i) + '.txt'
        temp[15] = '"'.join(row15)

        with open(fname, 'w') as json:
            for line in temp:
                json.write(line)

        subprocess.check_call('sh ./submit.sh {} {}'.format(token, fname), shell=True)
        print('Submit task:', name)
        if ite % 10 == 0:
            time.sleep(120)
        else:
            time.sleep(10)
        ite += 1


def task_run_2(task_name, lines, time_stamp, base_path, json_path, log_path, num_tasks, depth):
    pwd = os.getcwd()
    token = get_token(token_config, server_url)

    name = task_name + '_' + time_stamp
    fname = os.path.join(json_path, name + '.json')
    temp = lines
    row1 = temp[1].split('"')
    row1[3] = name
    temp[1] = '"'.join(row1)

    row15 = temp[15].split('"')
    row15[3] = 'cd ' + pwd + ';./tasks.sh ' + \
                task_name + ' ' + \
                str(num_tasks) + ' ' + \
                base_path + ' ' + \
                str(depth) + \
                ' 2>&1 | tee ' + log_path + '/' + \
                task_name + '.txt'
    temp[15] = '"'.join(row15)

    with open(fname, 'w') as json:
        for line in temp:
            json.write(line)

    subprocess.check_call('sh ./submit.sh {} {}'.format(token, fname), shell=True)
    print('Submit task:', name)

def count_active_block(num_tasks, active_path, used_path):
    time1 = time.time()
    all_blocks = os.path.join(used_path, 'all.txt')
    used_blocks = os.path.join(used_path, 'used.txt')
    f_all = open(all_blocks, 'w')
    f_used = open(used_blocks, 'w')

    number_used = 0
    for i in range(num_tasks):
        filename = os.path.join(active_path, str(i)+'.txt')
        if os.path.exists(filename):
            f = open(filename, 'r')
            lines = f.readlines()
            for k in lines:
                k = k.strip('\n')
                k_split = k.split(' ')
                flag = k_split[1]
                if flag == '1':
                    number_used += 1
                    f_used.write(k_split[0])
                    f_used.write('\n')
                f_all.write(k)
                f_all.write('\n')
    f_all.close() 
    f_used.close() 
    print('the number of used blocks is %d' % number_used)
    time2 = time.time()
    print('COST TIME', (time2 - time1))

#@profile
def sort_ids(record_path, used_path, txt_name='addval_list.txt'):
    time1 = time.time()
    # f_re = open(os.path.join(used_path, 'used.txt'), 'r')
    # block_list = f_re.read().splitlines()
    # f_re.close()
    f_re = open(os.path.join(used_path, 'used.txt'), 'r')
    block_list_all = f_re.readlines()
    f_re.close()
    block_list = []
    for k in block_list_all:
        k = k.split('\t')
        block_list.append(k[0])

    maxval_list = []
    for block in block_list:
        block_name = block.split('_')
        start_z = block_name[2]
        iy = block_name[3]
        block_path = os.path.join('/braindat/lab/liuxy/soma_seg/Code_Full_Brain_V2_seed/full_data_process2.0_stitch/soma_out', start_z, iy)
        # block_path = os.path.join(record_path, block+'.txt')
        data_block_open = h5py.File(os.path.join(block_path ,
                                            block+'.hdf'),
                               'r')
        data_block = data_block_open['soma'][:]
        data_block_open.close()
        # val_record = data_block.max()
        val_record = len(np.unique(data_block))-1
        del data_block
        print('current block:{}, max id: {}'.format(block_name,val_record))
        # f_record = open(block_path, 'r')
        # val_record = f_record.readlines()[0]
        # val_record = int(val_record.strip(' \n'))
        # f_record.close()
        maxval_list.append(val_record)

    addval_list = np.zeros(len(maxval_list), dtype=np.uint64)
    for i in range(1, len(maxval_list)):
        addval_list[i] = addval_list[i-1] + maxval_list[i-1] + 1
    out_path = os.path.join(used_path, txt_name)
    f_add = open(out_path, 'w')
    for i in range(len(addval_list)):
        f_add.write(block_list[i] + ' ' + str(addval_list[i]))
        f_add.write('\n')
    f_add.close()
    time2 = time.time()
    print('COST TIME', (time2 - time1))

#@profile
def count_stitching(stitching_dir, x, y, z, used):
    numX = x // 2
    numY = y // 2
    numZ = z // 2
    blocks_list = []

    if stitching_dir == 'x0':
        ite = 0
        for ix in range(numX):
            for iy in range(y):
                for iz in range(z):
                    iz = iz*6
                    block1 = 'soma_detect_' + str(iz) + '_' + str(iy) + '_' + str(ite)
                    block2 = 'soma_detect_' + str(iz) + '_' + str(iy) + '_' + str(ite+1)
                    if used[block1] == 1 and used[block2] == 1:
                        blocks_list.append(block1 + ' ' + block2 + ' ' + str(3) + ' ' + str(50))
            ite += 2
        print('The number of stitching in x0:', len(blocks_list))

    elif stitching_dir == 'x1':
        ite = 1
        if (x % 2 == 1):
            numX2 = numX
        else:
            numX2 = numX - 1
        for ix in range(numX2):
            for iy in range(y):
                for iz in range(z):
                    iz = iz * 6
                    block1 = 'soma_detect_' + str(iz) + '_' + str(iy) + '_' + str(ite)
                    block2 = 'soma_detect_' + str(iz) + '_' + str(iy) + '_' + str(ite+1)
                    if used[block1] == 1 and used[block2] == 1:
                        blocks_list.append(block1 + ' ' + block2 + ' ' + str(3) + ' ' + str(50))
            ite += 2
        print('The number of stitching in x1:', len(blocks_list))

    elif stitching_dir == 'y0':
        ite = 0
        for iy in range(numY):
            for ix in range(x):
                for iz in range(z):
                    iz = iz * 6
                    block1 = 'soma_detect_' + str(iz) + '_' + str(ite) + '_' + str(ix)
                    block2 = 'soma_detect_' + str(iz) + '_' + str(ite+1) + '_' + str(ix)
                    if used[block1] == 1 and used[block2] == 1:
                        blocks_list.append(block1 + ' ' + block2 + ' ' + str(2) + ' ' + str(50))
            ite += 2
        print('The number of stitching in y0:', len(blocks_list))

    elif stitching_dir == 'y1':
        ite = 1
        if (y % 2 == 1):
            numY2 = numY 
        else:
            numY2 = numY - 1
        for iy in range(numY2):
            for ix in range(x):
                for iz in range(z):
                    iz = iz * 6
                    block1 = 'soma_detect_' + str(iz) + '_' + str(ite) + '_' + str(ix)
                    block2 = 'soma_detect_' + str(iz) + '_' + str(ite+1) + '_' + str(ix)
                    if used[block1] == 1 and used[block2] == 1:
                        blocks_list.append(block1 + ' ' + block2 + ' ' + str(2) + ' ' + str(50))
            ite += 2
        print('The number of stitching in y1:', len(blocks_list))

    elif stitching_dir == 'z0':
        if z > 1:
            ite = 0
            for iz in range(numZ):
                for iy in range(y):
                    for ix in range(x):
                        block1 = 'soma_detect_' + str(ite*6) + '_' + str(iy) + '_' + str(ix)
                        block2 = 'soma_detect_' + str((ite+1)*6) + '_' + str(iy) + '_' + str(ix)
                        if used[block1] == 1 and used[block2] == 1: 
                            blocks_list.append(block1 + ' ' + block2 + ' ' + str(1) + ' ' + str(15))
                ite += 2
            print('The number of stitching in z0:', len(blocks_list))
        else:
            print('The number of sub-blocks in z direction is 1, it donot need to stitch.')

    elif stitching_dir == 'z1':
        if z > 1:
            ite = 1
            if (z % 2 == 1):
                numZ2 = numZ 
            else:
                numZ2 = numZ - 1
            for iz in range(numZ2):
                for iy in range(y):
                    for ix in range(x):
                        block1 = 'soma_detect_' + str(ite*6) + '_' + str(iy) + '_' + str(ix)
                        block2 = 'soma_detect_' + str((ite+1)*6) + '_' + str(iy) + '_' + str(ix)
                        if used[block1] == 1 and used[block2] == 1:
                            blocks_list.append(block1 + ' ' + block2 + ' ' + str(1) + ' ' + str(15))
                ite += 2
            print('The number of stitching in z1:', len(blocks_list))
        else:
            print('The number of sub-blocks in z direction is 1, it donot need to stitch.')

    else:
        raise AttributeError('No matching choice')

    return blocks_list



if __name__ == "__main__":

    ##############################################################################
    token_config = dict()
    server_url = ''
    ##############################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('-tn', '--task_name', type=str, default=None)
    parser.add_argument('-sd', '--stitching_dir', type=str, default=None)
    parser.add_argument('-rr', '--remove_record', action='store_false', default=True)
    parser.add_argument('-bp', '--base_path', type=str, default='/braindat/lab/liuxy/soma_seg/Code_Full_Brain_V2_seed/full_data_process2.0_stitch')
    parser.add_argument('-dp', '--depth', type=int, default=186)
    parser.add_argument('-nz', '--num_z', type=int, default=45)
    parser.add_argument('-ny', '--num_y', type=int, default=22)
    parser.add_argument('-nx', '--num_x', type=int, default=41)
    parser.add_argument('-st', '--start', type=int, default=1)
    parser.add_argument('-ed', '--end', type=int, default=7062)
    parser.add_argument('-task', '--num_tasks', type=int, default=16)
    args = parser.parse_args()

    # init
    task_name = args.task_name
    num_tasks = args.num_tasks
    block_number = 0
    # all_tasks = ['crop_section', 'active_block', 'count_active_block', 'detect_artifact', 'interp', 'inference', 'waterz', 
    #              'sort_ids', 'copy_seg', 'add_ids', 'stitching', 'concat1', 'concat2', 'global', 'remap']
    all_tasks = ['crop_section', 'active_block', 'count_active_block', 'segmentation', 
                 'sort_ids', 'add_ids', 'stitching', 'concat', 'global', 'remap', 
                 'move_and_zeros', 'sort_ids_2', 'add_ids_2']
    # all_tasks = [    0,              1,                 2,                 3,      
    #                 4,        5,           6,          7,       8,        9,       
    #                 10,              11,           12]
    #stitching pipeline: 4,5,6,7,8,9
    # check task_name
    if task_name == None:
        raise AttributeError('Mast have a task name!')
    # Adjust whether the task_name is in all_tasks or not.
    for i in range(len(all_tasks)+1):
        if i == len(all_tasks):
            print('Task: ' + task_name + ' is not in all_tasks')
            raise AttributeError('Task must be one of', all_tasks)
        if task_name == all_tasks[i]:
            print('This task is ' + task_name)
            break

    # time stamp
    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d_%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)

    # compute the number of blocks or sections in every tasks
    used_path = os.path.join(args.base_path, 'record', 'used')
    if task_name == all_tasks[0]:
        block_number = args.end - args.start + 1
    elif task_name == all_tasks[1]:
        block_list = []
        for iz in range(args.num_z):
            for iy in range(args.num_y):
                for ix in range(args.num_x):
                    block_name = 'soma_detect_' + str(iz) + '_' + str(iy) + '_' + str(ix)
                    block_list.append(block_name)
        block_number = len(block_list)
    elif task_name == all_tasks[2]:
        active_path = os.path.join(args.base_path, 'record', all_tasks[1])
        if not os.path.exists(used_path):
            os.makedirs(used_path)
        count_active_block(num_tasks, active_path, used_path)
        exit(0)
    elif task_name == all_tasks[4]:
        waterz_path = os.path.join(args.base_path, 'soma_record')
        sort_ids(waterz_path, used_path)
        exit(0)
    elif task_name == all_tasks[11]:
        waterz_path = os.path.join(args.base_path, 'record', all_tasks[10])
        sort_ids(waterz_path, used_path, txt_name='addval_list_2.txt')
        exit(0)
    elif task_name == all_tasks[6]:
        used = {}
        f_re = open(os.path.join(used_path, 'all.txt'), 'r')
        lines_re = f_re.readlines()
        for line in lines_re:
            # print(line)
            line = line.split('\t')
            # name = line.split(' ')
            # print(line[1])
            name1 = line[0]
            name2 = int(line[1].strip('\n'))
            used.setdefault(name1, name2)
        f_re.close()
        block_list = count_stitching(args.stitching_dir, args.num_x, args.num_y, args.num_z, used)
        record_stitching_path = os.path.join(args.base_path, 'record/used')
        f_st = open(os.path.join(record_stitching_path, args.stitching_dir+'.txt'), 'w')
        for k in block_list:
            f_st.write(k)
            f_st.write('\n')
        f_st.close()
        block_number = len(block_list)
        print('In ' + args.stitching_dir + ', the number of used blocks:', block_number)
    elif task_name in (all_tasks[3], all_tasks[5], all_tasks[7], all_tasks[9], all_tasks[10], all_tasks[12]):
        f_re = open(os.path.join(used_path, 'used.txt'), 'r')
        block_list_all = f_re.readlines()
        f_re.close()
        block_list = []
        for k in block_list_all:
            k = k.split('\t')
            # k_name = k.split(' ')
            block_list.append(k[0])
        block_number = len(block_list)
        print('The number of used blocks:', block_number)
    else:
        print('do nothing.')
        pass

    # create folder for add_ids task
    if task_name == all_tasks[5] or task_name == all_tasks[10]:
        print('Create folder for seg_results...')
        seg_path = os.path.join(args.base_path, 'seg_results')
        if not os.path.exists(seg_path):
            os.mkdir(seg_path)
        for block in block_list:
            iz = block.split('_')[-3]
            iy = block.split('_')[-2]
            block_path = os.path.join(seg_path, iz, iy)
            if not os.path.exists(block_path):
                os.makedirs(block_path)

    # check path
    # json path
    json_path = os.path.join(args.base_path, 'json', task_name)
    if os.path.exists(json_path):
        shutil.rmtree(json_path)
    os.makedirs(json_path)
    # log path
    log_path = os.path.join(args.base_path, 'log', task_name)
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    # record path
    record_path = os.path.join(args.base_path, 'record', task_name)
    if args.remove_record == True:
        print('clean record_path...')
        if os.path.exists(record_path):
            shutil.rmtree(record_path)
        os.makedirs(record_path)
    # submit path
    submit_path = os.path.join(args.base_path, 'submit', task_name)
    if os.path.exists(submit_path):
        shutil.rmtree(submit_path)
    os.makedirs(submit_path)

    block_number_of_each_task = block_number // num_tasks
    remainder = block_number % num_tasks
    if remainder != 0:
        block_number_of_each_task = block_number_of_each_task + 1
    if task_name == all_tasks[0]:
        for i in range(num_tasks):
            sec_id = i * block_number_of_each_task + 1
            if remainder != 0 and i == num_tasks - 1:
                num = args.end + 1 - sec_id
            else:
                num = block_number_of_each_task
            block_list_name = os.path.join(submit_path, str(i)+'.txt')
            f_block_list = open(block_list_name, 'w')
            f_block_list.write(str(sec_id) + ' ' + str(num))
            f_block_list.close()
    elif task_name in (all_tasks[1], all_tasks[3], all_tasks[5], all_tasks[6], all_tasks[7], all_tasks[9], all_tasks[10], all_tasks[12]):
        for i in range(num_tasks):
            if remainder != 0 and i == num_tasks - 1:
                temp_block = block_list[i*block_number_of_each_task:]
            else:
                temp_block = block_list[i*block_number_of_each_task : (i+1)*block_number_of_each_task]
            block_list_name = os.path.join(submit_path, str(i)+'.txt')
            f_block_list = open(block_list_name, 'w')
            for k in temp_block:
                f_block_list.write(k)
                f_block_list.write('\n')
            f_block_list.close()
    else:
        print('do nothing.')
        pass

    # open example json
    f = open('./example_json/json-' + task_name + '.json', 'r')
    lines = f.readlines()
    f.close()

    # submit task
    if task_name in (all_tasks[0], all_tasks[1], all_tasks[3], \
                    all_tasks[5], all_tasks[6], all_tasks[7], all_tasks[9], all_tasks[10], all_tasks[12]):
        print('submit task 1...')
        task_run(task_name, lines, time_stamp, args.base_path, json_path, log_path, num_tasks, args.depth)
    elif task_name == all_tasks[8]:
        print('submit task 2...')
        task_run_2(task_name, lines, time_stamp, args.base_path, json_path, log_path, num_tasks, args.depth)
    else:
        print('no submit.')
        pass 
