# -*- coding: utf-8 -*-
# @Time : 2022/4/17
# @Author : Xiaoyu Liu
# @Email : liuxyu@mail.ustc.edu.cn
# @File : submit_test.py
# @Software: PyCharm
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
import time
import os
import argparse
import shutil
import requests
import logging
import subprocess

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
                    base_path + \
                    ' 2>&1 | tee ' + log_path + '/' + \
                    task_name + '_' + str(i) + '.txt'
        temp[15] = '"'.join(row15)

        with open(fname, 'w') as json:
            for line in temp:
                json.write(line)

        subprocess.check_call('sh ./submit.sh {} {}'.format(token, fname), shell=True)
        print('Submit task:', name)
        if ite % 10 == 0:
            time.sleep(100)
        else:
            time.sleep(20)
        ite += 1

if __name__ == "__main__":

    ##############################################################################
    token_config = dict()
    server_url = ''
    ##############################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('-tn', '--task_name', type=str, default=None)
    parser.add_argument('-bp', '--base_path', type=str, default='/braindat/lab/liuxy/soma_seg/Code_Full_Brain_V2_seed/full_data_process2.0_seg')
    parser.add_argument('-dp', '--depth', type=int, default=84)
    parser.add_argument('-nz', '--num_z', type=int, default=270)
    parser.add_argument('-ny', '--num_y', type=int, default=22)
    parser.add_argument('-nx', '--num_x', type=int, default=41)
    parser.add_argument('-task', '--num_tasks', type=int, default=20)
    args = parser.parse_args()

    # time stamp
    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d_%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)

    # init
    task_name = args.task_name
    num_tasks = args.num_tasks
    num_z = args.num_z
    num_y = args.num_y
    num_x = args.num_x

#_____________________select a task_________________________________#
    all_tasks = ['divide_block','segmentation','merge']

    assert task_name in all_tasks
# __________________________________________________________________#

    if task_name==all_tasks[0]:
        submit_path = os.path.join(args.base_path, 'submit')
        if os.path.exists(submit_path):
            shutil.rmtree(submit_path)
        os.makedirs(submit_path)
        f_all = open('./all.txt','r')
        f_all_lines = f_all.readlines()
        f_active = open('./used.txt','r')
        f_active_lines = f_active.readlines()
        block_list = []
        for line in f_all_lines:
            line = line.strip('\n')
            # block_name = line.split(' ')
            block_list.append(line)
        block_number = len(block_list)
        print('the number of all block: ',block_number)


        #for i in range(num_tasks):
        xy_num_each_z = num_x*num_y
        z_number_of_each_task = num_z // num_tasks
        remainder = num_z % num_tasks
        if remainder != 0:
            z_number_of_each_task = z_number_of_each_task + 1

        for i in range(num_tasks):
            temp_block = []
            if remainder != 0 and i == num_tasks - 1:
                temp_block = block_list[i * z_number_of_each_task*xy_num_each_z:]

            else:
                temp_block = block_list[i * z_number_of_each_task*xy_num_each_z: (i + 1) * z_number_of_each_task*xy_num_each_z]
            block_list_name = os.path.join(submit_path, str(i) + '.txt')
            f_block_list = open(block_list_name, 'w')
            for k in temp_block:
                f_block_list.write(k)
                f_block_list.write('\n')
            f_block_list.close()

    if task_name in(all_tasks[1],all_tasks[2]):
        # json path   # write all configure information for xp( adding command)
        json_path = os.path.join(args.base_path, 'json', task_name)
        if os.path.exists(json_path):
            shutil.rmtree(json_path)
        os.makedirs(json_path)
        # log path   # log of a task in xp
        log_path = os.path.join(args.base_path, 'log', task_name)
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
        os.makedirs(log_path)

        # open example json   #Configuring computing resources
        f = open('./example_json/json-' + task_name + '.json', 'r')
        lines = f.readlines()
        f.close()

        task_run(task_name, lines, time_stamp, args.base_path, json_path, log_path, num_tasks, args.depth)
