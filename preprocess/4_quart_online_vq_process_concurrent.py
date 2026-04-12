#This file is for processing the vq json code for quart-online training using trained .pt model, concurrently

import sys
import os

# Use relative paths for cross-platform compatibility
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_PATH)

from preprocess.init_path import SIM_INSTRUCTION_DICT, REAL_INSTRUCTION_DICT

import math
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
import concurrent

from models.RVQ.vq_Sequence import RVQ_Seq, RVQ_Seq_10
from models.RVQ.residual_vq import RVQ

input_dim=12
n_step=10

if n_step==5:
    vq_path = os.path.join(ROOT_PATH, 'vq_state_dict', 'VQ', 'Sequence_vq_5_each_conv.pt')
    VQ_model=RVQ_Seq(layers_hidden=[512, 512, 512, 512], input_dim=input_dim, K=512, num_quantizers=2, output_act=False)

elif n_step==10:
    vq_path = os.path.join(ROOT_PATH, 'vq_state_dict', 'VQ', 'Sequence_vq_10_each_conv.pt')
    VQ_model=RVQ_Seq_10(layers_hidden=[512, 512, 512, 512], input_dim=input_dim, K=512, num_quantizers=2, output_act=False)

VQ_model.load_state_dict(torch.load(vq_path, map_location='cpu'))
print("Loaded VQ model from:", vq_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VQ_model = VQ_model.to(device)



def make_vq_json(sim_instruction_dict, ranges_info_path, commands_info_path, json_path, sim_path, sample_rate, ahead_step):
    task_list = list(sim_instruction_dict.keys())
    range_dict = {}
    image_id = 0
    ahead_name = 'sim_vq_ahead_' + str(ahead_step) + '_seq'
    json_saved_path = os.path.join(json_path, ahead_name)
    os.makedirs(json_saved_path, exist_ok=True)

    with ThreadPoolExecutor(max_workers=4) as executor:  
        futures = []
        for task in task_list:
            futures.append(executor.submit(single_task_json_vq, sim_path, commands_info_path, task, range_dict, image_id, sample_rate, ahead_step, json_saved_path))
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            task_name = future.result()[0]

def single_task_json_vq(root_path, info_path, task, range_dict, image_id, sample_rate, ahead_step, json_saved_path):
    json_list = []
    json_list = single_task_json_vq_core(root_path, info_path, task, range_dict, json_list, image_id, sample_rate, ahead_step, json_saved_path)
    return task, json_saved_path, json_list



def single_task_json_vq_core(root_path, info_path, task, range_dict, json_list, image_id, sample_rate, ahead_step, json_saved_path):
    all_dict_path = os.path.join(info_path, "commands.npy")
    all_dict = np.load(all_dict_path, allow_pickle=True).item()
    
    vq_oneList=[]
    vq_DataList=[]

    task_path = os.path.join(root_path, task)

    print("processing task_path:",task_path)

    step_list=[]

    if os.path.exists(task_path):
        episode_list = os.listdir(task_path) #['000672','000178',...]
        for episode in tqdm(episode_list):
            episode_path = os.path.join(task_path, episode)
            episode_command_path = os.path.join(episode_path, "command")
            if os.path.exists(episode_path): 
                
                if episode_path in all_dict.keys():
                    commands_dict = all_dict[episode_path]
                    episode_length = len(commands_dict['dx'])
                    for i in range(episode_length): 
                        dict_json, vq_input_list, img_idx = get_fill_vq(i,commands_dict,episode_length,image_id,sample_rate,episode_path)
                        step_list.append(vq_input_list)

                        if i>=ahead_step-1: #cut sequentially
                            merged_list = []
                            # result_value = '<0x04> ' + ' '.join(merged_list)

                            vq_list=[]

                            # flatten tensor and turn into string
                            vq_input = np.array(step_list[-ahead_step:])
                            vq_input = torch.from_numpy(vq_input).to(torch.float32).to(device)
                            if "Seq" in vq_path:
                                vq_input=torch.unsqueeze(vq_input,dim=0)
                            vq_output = VQ_model.tokenize(vq_input)

                            vq_flattened = vq_output.flatten()
                            vq_list = list(map(str, vq_flattened.cpu().numpy()))  #

                            vq_value = '<0x04> ' + ' '.join(vq_list)
                            # [[ 13, 320],[ 16, 276]] → [ 13, 320,  16, 276] → '<0x04> 13 320 16 276'


                            dict_json['vq'] = vq_value   
                            dict_json['conversations'][1]['value'] = '<0x04> ' 
                            pre_img_idx = int(img_idx)-(ahead_step-1)*sample_rate    
                            # import pdb; pdb.set_trace()
                            pre_img_idx="{:03d}".format(pre_img_idx)
                            dict_json['image'] = os.path.join(episode_path, f"image/{pre_img_idx}.png")
                            json_list.append(dict_json) 

                            image_id += 1




    with open(json_saved_path + '/{}.json'.format(task), 'w') as f:
        json.dump(json_list, f)
    print("one json file saved to:",json_saved_path,'/{}.json'.format(task))
    return json_list

# gpt_value = source['conversations'][1]['value']


def get_fill_vq(i,commands_dict,episode_length,image_id,sample_rate,episode_path):
    true_idx = i * sample_rate
    img_idx = "{:03d}".format(true_idx)
    img_path = os.path.join(episode_path, f"image/{img_idx}.png")
    if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
        img_idx = "{:03d}".format(true_idx - 1)
        img_path = os.path.join(episode_path, f"image/{img_idx}.png")
    dx = commands_dict['dx'][true_idx]
    dy = commands_dict['dy'][true_idx]
    dyaw = commands_dict['dyaw'][true_idx]
    body_height = commands_dict['body_height'][true_idx]
    step_frequency = commands_dict['step_frequency'][true_idx]
    gait_0 = commands_dict['gait_0'][true_idx]
    gait_1 = commands_dict['gait_1'][true_idx]
    gait_2 = commands_dict['gait_2'][true_idx]
    footswing_height = commands_dict['footswing_height'][true_idx]
    pitch = commands_dict['pitch'][true_idx]
    stance_width = commands_dict['stance_width'][true_idx]                    
    instruction = commands_dict['instruction']

    terminate = int(i == episode_length - 1)
    dict_json = {}
    dict_json['id'] = str(image_id).rjust(12,'0')  
    dict_json['image'] = img_path

    dict_json['conversations'] = []
    human = {
        'from': 'human',
        'value': 'What action should the legged robot take to {}?\n'.format(instruction),
        'type': 'sim'
    }
    gpt = {
        'from': 'gpt',
        # 'value': '<0x04> {} {} {} {} {} {} {} {} {} {} {} {}'.format(terminate, dx_token,dy_token,dyaw_token,body_token,step_frequency_token,gait_0_token,gait_1_token,gait_2_token,footswing_height_token,pitch_token,stance_width_token)
        'value': '<0x04> {} {} {} {} {} {} {} {} {} {} {} {}'.format(terminate, dx,dy,dyaw,body_height,step_frequency,gait_0,gait_1,gait_2,footswing_height,pitch,stance_width)
    }
    dict_json['conversations'].append(human)
    dict_json['conversations'].append(gpt)
    vq_input_list=[terminate, dx,dy,dyaw,body_height,step_frequency,gait_0,gait_1,gait_2,footswing_height,pitch,stance_width]
    return dict_json, vq_input_list, img_idx

    



if __name__ == "__main__":
    instructions_key = 'Full' 
    
    # Use relative paths - users should set RAW_DATA_PATH to their data location
    RAW_DATA_PATH = '/path/to/your/Datasets'  # TODO: Change this to your data path
    
    proprioception_keys = ['joint_pos', 'joint_vel', 'body_linear_vel', 'body_angular_vel', 'contact_states', 'body_pos', 'body_quat']
    sim_sample_rate = 10  
    ahead_step = 10 

    # Input path - raw simulation data (v1 and unload data should be merged into one folder)
    sim_path = os.path.join(RAW_DATA_PATH, 'sim_quadruped_data')
    
    # Output paths - processed data
    sim_info_path = os.path.join(ROOT_PATH, 'datasets', instructions_key, 'sim_quadruped_data_info')
    sim_json_path = os.path.join(ROOT_PATH, 'datasets', instructions_key, 'sim_json_path')
    
    sim_instruction_dict = SIM_INSTRUCTION_DICT[instructions_key]

    make_vq_json(sim_instruction_dict, sim_info_path, sim_info_path, sim_json_path, sim_path, sim_sample_rate, ahead_step)

