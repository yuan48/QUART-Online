# 弃用
import sys
QUART_path = '/dingpengxiang/Pengxiang/Quart++'
sys.path.append(QUART_path)

from init_path import SIM_INSTRUCTION_DICT, REAL_INSTRUCTION_DICT

import os
import math
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch



def merged_single_json(sim_json_path, sim_instruction_dict, data_type, json_save_name):

    sim_task_list = sim_instruction_dict.keys()
    # data_type = 'sim'

    all_list = []
    all_json_path = os.path.join(sim_json_path,json_save_name)  #无ahead默认/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_json_path/sim.json
    for task in tqdm(sim_task_list):
        json_name = '{}/{}.json'.format(data_type,task)
        json_load_path =  os.path.join(sim_json_path, json_name)  #例如/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_json_path/sim/crawl_red_gate1.json
        with open(json_load_path, 'r') as f:
            cur_list = json.load(f)
            all_list = all_list + cur_list

    with open(all_json_path, 'w') as f:
        json.dump(all_list, f)
    print("save to path:",all_json_path)


if __name__ == "__main__":
    AHEAD=False #是否提前预测n步
    SEQUENCE=True

    if SEQUENCE==True:
        ahead_step=10  #提前预测的step数
        instructions_key = 'Full' 
        ROOT_PATH='/dingpengxiang/Pengxiang/Quart++'
        json_save_name='sim_ahead_'+str(ahead_step)+'_seq.json'
        data_type= 'sim_vq_ahead_'+str(ahead_step)+'_seq'  #jsonpath的下述类型/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_json_path/sim_vq_ahead_3

        sim_instruction_dict = SIM_INSTRUCTION_DICT[instructions_key]
        sim_json_path = os.path.join(ROOT_PATH, 'datasets', instructions_key, 'sim_json_path') #'/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_json_path'

        merged_single_json(sim_json_path, sim_instruction_dict, data_type, json_save_name)  #单独把sim的json融合子任务为sim.json

    elif AHEAD==True:
        ahead_step=5  #提前预测的step数
        instructions_key = 'Full' 
        ROOT_PATH='/dingpengxiang/Pengxiang/Quart++'
        json_save_name='sim_ahead_'+str(ahead_step)+'.json'
        data_type= 'sim_vq_ahead_'+str(ahead_step)  #jsonpath的下述类型/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_json_path/sim_vq_ahead_3

        sim_instruction_dict = SIM_INSTRUCTION_DICT[instructions_key]
        sim_json_path = os.path.join(ROOT_PATH, 'datasets', instructions_key, 'sim_json_path') #'/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_json_path'

        merged_single_json(sim_json_path, sim_instruction_dict, data_type, json_save_name)  #单独把sim的json融合子任务为sim.json
    else:
        ahead_step=3  #提前预测的step数
        instructions_key = 'Full' 
        ROOT_PATH='/dingpengxiang/Pengxiang/Quart'
        data_type='sim'
        json_save_name='sim.json'  #./Quart++/datasets/Full/sim_json_path/sim.json

        # data_type='han_quart'
        # json_save_name='han_quart.json'
        sim_instruction_dict = SIM_INSTRUCTION_DICT[instructions_key]
        sim_json_path = os.path.join(ROOT_PATH, 'datasets', instructions_key, 'sim_json_path') #'/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_json_path'

        merged_single_json(sim_json_path, sim_instruction_dict, data_type, json_save_name)  #just combine all sim tasks into sim.json
