import sys
from init_path import SIM_INSTRUCTION_DICT, REAL_INSTRUCTION_DICT

import os
import math
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_sim_proprioception(sample_rate, sim_instruction_dict, data_path, data_path_unload, info_path, proprioception_keys):

    # tasks in raw datasets
    tasks = next(os.walk(data_path))[1]
    tasks_unload = next(os.walk(data_path_unload))[1]
    tasks = tasks + tasks_unload

    # remove some tasks according to the key of instruction dict
    tasks = [i for i in tasks if i in list(sim_instruction_dict.keys())]
    if not os.path.exists(info_path):
        os.makedirs(info_path,exist_ok=True)

    # init the dict to save all proprioceptions
    all_proprioceptions_dict = {}

    # loop over all tasks
    for task in tqdm(tasks):
        # get the path of each task
        task_path = os.path.join(data_path, task)
        if not os.path.exists(task_path):
            task_path = os.path.join(data_path_unload, task)

        # get all episodes of current task
        episode_files = os.listdir(task_path)
        episode_files = [i for i in episode_files if '.' not in i]
        num_episodes = len(episode_files)
        print("task: {} {}".format(task, num_episodes))
        
        for episode in episode_files:
            episode_path = os.path.join(task_path, episode)
            episode_proprioception_path = os.path.join(episode_path, "action")
            episode_instruction_path = os.path.join(episode_path, "info.csv")
            df = pd.read_csv(episode_instruction_path)

            # get the proprioceptions of each episode
            if os.path.exists(episode_proprioception_path):   
                episode_proprioceptions = os.listdir(episode_proprioception_path)
                episode_length = len(episode_proprioceptions)
                if episode_length > 5 * sample_rate:

                    # print("episode: {} {}".format(episode, episode_length))
                    episode_dict = {}
                    joint_pos_dict = {}
                    joint_vel_dict = {}
                    body_linear_vel_dict = {}
                    body_angular_vel_dict = {}
                    contact_states_dict = {}
                    body_pos_dict = {}
                    body_quat_dict = {}

                    
                    # loop over all proprioceptions of each episode
                    for i in range(0, episode_length, sample_rate):
                        proprioception_data_path = os.path.join(episode_proprioception_path, '{:03d}.npy'.format(i))
                        if os.path.getsize(proprioception_data_path) == 0:
                            proprioception_data_path = os.path.join(episode_proprioception_path, '{:03d}.npy'.format(i - 1))
                        data = np.load(proprioception_data_path, allow_pickle=True)
                        # import pdb; pdb.set_trace()
                        joint_pos = data.item()['joint_pos']
                        joint_vel = data.item()['joint_vel']
                        body_linear_vel = data.item()['body_linear_vel']
                        body_angular_vel = data.item()['body_angular_vel']
                        contact_states = data.item()['contact_states']
                        body_pos = data.item()['body_pos']
                        body_quat = data.item()['body_quat']
                        
                        joint_pos_dict[i] = joint_pos
                        joint_vel_dict[i] = joint_vel
                        body_linear_vel_dict[i] = body_linear_vel
                        body_angular_vel_dict[i] = body_angular_vel
                        contact_states_dict[i] = contact_states
                        body_pos_dict[i] = body_pos
                        body_quat_dict[i] = body_quat
                        
                            
                    episode_dict['joint_pos'] = joint_pos_dict
                    episode_dict['joint_vel'] = joint_vel_dict
                    episode_dict['body_linear_vel'] = body_linear_vel_dict
                    episode_dict['body_angular_vel'] = body_angular_vel_dict
                    episode_dict['contact_states'] = contact_states_dict
                    episode_dict['body_pos'] = body_pos_dict
                    episode_dict['body_quat'] = body_quat_dict


                    save_episode_path =  episode_path
                    all_proprioceptions_dict[save_episode_path] = episode_dict

    dict_path = os.path.join(info_path, "proprioceptions.npy")
    np.save(dict_path, all_proprioceptions_dict)

def get_sim_ranges(sample_rate, sim_instruction_dict, data_path, data_path_unload, info_path):

    # tasks in raw datasets
    tasks = next(os.walk(data_path))[1]
    tasks_unload = next(os.walk(data_path_unload))[1]
    tasks = tasks + tasks_unload

    # remove some tasks according to the key of instruction dict
    tasks = [i for i in tasks if i in list(sim_instruction_dict.keys())]
    if not os.path.exists(info_path):
        os.makedirs(info_path,exist_ok=True)
    
    # init the max and min of each command
    dx_max, dx_min = -100, 100
    dy_max, dy_min = -100, 100
    dyaw_max, dyaw_min = -100, 100
    body_height_max, body_height_min = -100, 100
    step_frequency_max, step_frequency_min = -100, 100
    gait_max, gait_min = 1.0, 0.0
    footswing_height_max, footswing_height_min = -100, 100
    pitch_max, pitch_min = -100, 100
    roll_max, roll_min = -100, 100
    stance_width_max, stance_width_min = -100, 100

    # init the dict to save all commands
    all_commands_dict = {}

    # init the count of dy != 0 and pitch != 0
    count_dy = 0
    count_pitch = 0
  
    # loop over all tasks
    for task in tqdm(tasks):
        # get the path of each task
        task_path = os.path.join(data_path, task)
        if not os.path.exists(task_path):
            task_path = os.path.join(data_path_unload, task)

        # get all episodes of current task
        episode_files = os.listdir(task_path)
        episode_files = [i for i in episode_files if '.' not in i]
        num_episodes = len(episode_files)
        print("task: {} {}".format(task, num_episodes))
        
        for episode in episode_files:
            episode_path = os.path.join(task_path, episode)
            episode_command_path = os.path.join(episode_path, "command")
            episode_instruction_path = os.path.join(episode_path, "info.csv")
            df = pd.read_csv(episode_instruction_path)

            # get the commands of each episode
            if os.path.exists(episode_command_path):   
                episode_commands = os.listdir(episode_command_path)
                episode_length = len(episode_commands)
                if episode_length > 5 * sample_rate:

                    # print("episode: {} {}".format(episode, episode_length))
                    episode_dict = {}
                    dx_dict = {}
                    dy_dict = {}
                    dyaw_dict = {}
                    body_height_dict = {}
                    step_frequency_dict = {}
                    gait_0_dict = {}
                    gait_1_dict = {}
                    gait_2_dict = {}
                    footswing_height_dict = {}
                    pitch_dict = {}
                    roll_dict = {}
                    stance_width_dict = {}
                    
                    # loop over all commands of each episode
                    for i in range(0, episode_length, sample_rate):
                        command_data_path = os.path.join(episode_command_path, '{:03d}.npy'.format(i))
                        if os.path.getsize(command_data_path) == 0:
                            command_data_path = os.path.join(episode_command_path, '{:03d}.npy'.format(i - 1))
                        data = np.load(command_data_path, allow_pickle=True)
                        dx = float(data.item()['x_vel_cmd'])
                        dy = float(data.item()['y_vel_cmd'])
                        dyaw = float(data.item()['yaw_vel_cmd'])
                        body_height = float(data.item()['body_height_cmd'])
                        step_frequency = float(data.item()['step_frequency_cmd'])
                        gait_0 = float(data.item()['gait '][0])
                        gait_1 = float(data.item()['gait '][1])
                        gait_2 = float(data.item()['gait '][2])
                        footswing_height = float(data.item()['footswing_height_cmd'])
                        pitch = float(data.item()['pitch_cmd'])
                        roll = float(data.item()['roll_cmd'])
                        stance_width = float(data.item()['stance_width_cmd'])

                        if dy != 0:
                            count_dy += 1
                            if i < 400:
                                print("path:", command_data_path, "dy:", dy)
                        if pitch != 0:
                            count_pitch += 1
                            if i < 400:
                                print("path:", command_data_path, "pitch:", pitch)
                        
                        dx_dict[i] = dx 
                        dy_dict[i] = dy
                        dyaw_dict[i] = dyaw
                        body_height_dict[i] = body_height
                        step_frequency_dict[i] = step_frequency
                        gait_0_dict[i] = gait_0
                        gait_1_dict[i] = gait_1
                        gait_2_dict[i] = gait_2
                        footswing_height_dict[i] = footswing_height
                        pitch_dict[i] = pitch
                        roll_dict[i] = roll
                        stance_width_dict[i] = stance_width
                        
                        if dx > dx_max:
                            dx_max = dx
                        if dy > dy_max:
                            dy_max = dy
                        if dyaw > dyaw_max:
                            dyaw_max = dyaw
                        if body_height > body_height_max:
                            body_height_max = body_height
                        if step_frequency > step_frequency_max:
                            step_frequency_max = step_frequency
                        if footswing_height > footswing_height_max:
                            footswing_height_max = footswing_height
                        if pitch > pitch_max:
                            pitch_max = pitch
                        if roll > roll_max:
                            roll_max = roll
                        if stance_width > stance_width_max:
                            stance_width_max = stance_width
                        if dx < dx_min:
                            dx_min = dx
                        if dy < dy_min:
                            dy_min = dy
                        if dyaw < dyaw_min:
                            dyaw_min = dyaw
                        if body_height < body_height_min:
                            body_height_min = body_height
                        if step_frequency < step_frequency_min:
                            step_frequency_min = step_frequency
                        if footswing_height < footswing_height_min:
                            footswing_height_min = footswing_height
                        if pitch < pitch_min:
                            pitch_min = pitch
                        if roll < roll_min:
                            roll_min = roll
                        if stance_width < stance_width_min:
                            stance_width_min = stance_width
                            
                    episode_dict['dx'] = dx_dict
                    episode_dict['dy'] = dy_dict
                    episode_dict['dyaw'] = dyaw_dict
                    episode_dict['body_height'] = body_height_dict
                    episode_dict['step_frequency'] = step_frequency_dict
                    episode_dict['gait_0'] = gait_0_dict
                    episode_dict['gait_1'] = gait_1_dict
                    episode_dict['gait_2'] = gait_2_dict
                    episode_dict['footswing_height'] = footswing_height_dict
                    episode_dict['pitch'] = pitch_dict
                    episode_dict['roll'] = roll_dict
                    episode_dict['stance_width'] = stance_width_dict
                    episode_dict['instruction'] = sim_instruction_dict[task] + df['instruction'].values.tolist()[0].split(',')[1] + ' with a {} gait'.format(df['gait'].values.tolist()[0])
                    # 
                    save_episode_path =  episode_path
                    all_commands_dict[save_episode_path] = episode_dict
 

    dict_path = os.path.join(info_path, "commands.npy")
    np.save(dict_path, all_commands_dict)

    print("dy != 0:", count_dy, "pitch != 0", count_pitch)

    dx_range = dx_max - dx_min
    dy_range = dy_max - dy_min
    dyaw_range = dyaw_max - dyaw_min
    body_height_range = body_height_max - body_height_min
    step_frequency_range = step_frequency_max - step_frequency_min
    footswing_height_range = footswing_height_max - footswing_height_min
    pitch_range = pitch_max - pitch_min
    roll_range = roll_max - roll_min
    stance_width_range = stance_width_max - stance_width_min

    command_space_low = np.array([dx_min, dy_min, dyaw_min, body_height_min, step_frequency_min, gait_min, gait_min, gait_min, footswing_height_min, pitch_min, stance_width_min])
    command_space_high = np.array([dx_max, dy_max, dyaw_max, body_height_max, step_frequency_max, gait_max, gait_max, gait_max, footswing_height_max, pitch_max, stance_width_max])

    range_dict = {}
    range_dict['dx_range'] = dx_range
    range_dict['dy_range'] = dy_range
    range_dict['dyaw_range'] = dyaw_range
    range_dict['body_height_range'] = body_height_range
    range_dict['step_frequency_range'] = step_frequency_range
    range_dict['footswing_height_range'] = footswing_height_range
    range_dict['pitch_range'] = pitch_range
    range_dict['roll_range'] = roll_range
    range_dict['stance_width_range'] = stance_width_range
    range_dict['dx_max'] = dx_max
    range_dict['dx_min'] = dx_min
    range_dict['dy_max'] = dy_max
    range_dict['dy_min'] = dy_min
    range_dict['dyaw_max'] = dyaw_max
    range_dict['dyaw_min'] = dyaw_min
    range_dict['body_height_max'] = body_height_max
    range_dict['body_height_min'] = body_height_min
    range_dict['step_frequency_max'] = step_frequency_max
    range_dict['step_frequency_min'] = step_frequency_min
    range_dict['footswing_height_max'] = footswing_height_max
    range_dict['footswing_height_min'] = footswing_height_min
    range_dict['pitch_max'] = pitch_max
    range_dict['pitch_min'] = pitch_min
    range_dict['roll_max'] = roll_max
    range_dict['roll_min'] = roll_min
    range_dict['stance_width_max'] = stance_width_max
    range_dict['stance_width_min'] = stance_width_min
    range_dict['command_space_low'] = command_space_low
    range_dict['command_space_high'] = command_space_high

    range_dict_path = os.path.join(info_path, "ranges.npy")
    np.save(range_dict_path, range_dict)

def get_real_ranges(sample_rate, real_instruction_dict, data_path, info_path):
    
    # tasks in raw datasets
    tasks = next(os.walk(data_path))[1]
    # remove some tasks according to the key of instruction dict
    tasks = [i for i in tasks if i in list(real_instruction_dict.keys())]
    if not os.path.exists(info_path):
        os.makedirs(info_path,exist_ok=True)

    # init the max and min of each command
    dx_max, dx_min = -100, 100
    dy_max, dy_min = -100, 100
    dyaw_max, dyaw_min = -100, 100
    body_height_max, body_height_min = -100, 100
    step_frequency_max, step_frequency_min = -100, 100
    gait_max, gait_min = 1.0, 0.0
    footswing_height_max, footswing_height_min = -100, 100
    pitch_max, pitch_min = -100, 100
    roll_max, roll_min = -100, 100
    stance_width_max, stance_width_min = -100, 100

    # init the dict to save all commands
    all_dict = {}
    count_dy = 0
    count_pitch = 0
 
    for task in tqdm(tasks):
        task_path = os.path.join(data_path, task)
        task_info_path = os.path.join(info_path, task)

        episode_files = os.listdir(task_path)
        num_episodes = len(episode_files)
        print("task: {} {} ".format(task, num_episodes))
        
        for episode in episode_files:
            episode_path = os.path.join(task_path, episode)
            
            episode_info_path = os.path.join(task_info_path, episode)
            episode_action_path = os.path.join(episode_path, "action")
            episode_image_path = os.path.join(episode_path, "image")
            

            if os.path.exists(episode_action_path):   
                actions = os.listdir(episode_action_path)
                images = os.listdir(episode_image_path)
                if len(images) != len(actions):
                    print(episode_action_path)
                if len(images) == len(actions):
                    episode_length = len(images)
                    if episode_length <30:
                        rate = 'quickly'
                    elif episode_length <60:
                        rate = 'normally'
                    else:
                        rate = 'slowly'
                    # print(episode_action_path)
                    if episode_length > 5 * sample_rate:
                        # print(episode_length)
                        # print("episode: {} {}".format(episode, episode_length))
                    
                        episode_dict = {}
                        dx_dict = {}
                        dy_dict = {}
                        dyaw_dict = {}
                        body_height_dict = {}
                        step_frequency_dict = {}
                        gait_0_dict = {}
                        gait_1_dict = {}
                        gait_2_dict = {}
                        footswing_height_dict = {}
                        pitch_dict = {}
                        roll_dict = {}
                        stance_width_dict = {}
                        
                        for i in range(0, episode_length, sample_rate):
                            command_data_path = os.path.join(episode_action_path, '{}.npy'.format(i))
                            if os.path.getsize(command_data_path) == 0:
                                command_data_path = os.path.join(episode_action_path, '{}.npy'.format(i - 1))
                            data = np.load(command_data_path, allow_pickle=True)

                            
                            if 'command_v_des' not in data.item().keys():
                                pass
                            else:

                                dx = float(data.item()['command_v_des'].split(',')[0].split('(')[1])
                                dy = float(data.item()['command_v_des'].split(',')[1])
                                # dy = float(0)
                                dyaw = float(data.item()['command_omega_des'].split(',')[2].split(')')[0])
                                body_height = float(0.285)
                                gait_0 = float(0.5)
                                gait_1 = float(0)
                                gait_2 = float(0)
                                # 以下是不知道的量
                                step_frequency = float(4)
                                footswing_height = float(0.02)
                                pitch = float(0)
                                roll = float(0)
                                stance_width = float(0.3)
                                
                                dx_dict[i] = dx 
                                dy_dict[i] = dy
                                dyaw_dict[i] = dyaw
                                body_height_dict[i] = body_height
                                step_frequency_dict[i] = step_frequency
                                gait_0_dict[i] = gait_0
                                gait_1_dict[i] = gait_1
                                gait_2_dict[i] = gait_2
                                footswing_height_dict[i] = footswing_height
                                pitch_dict[i] = pitch
                                roll_dict[i] = roll
                                stance_width_dict[i] = stance_width
                                
                                if dx > dx_max:
                                    dx_max = dx
                                if dy > dy_max:
                                    dy_max = dy
                                if dyaw > dyaw_max:
                                    dyaw_max = dyaw
                                if body_height > body_height_max:
                                    body_height_max = body_height
                                if step_frequency > step_frequency_max:
                                    step_frequency_max = step_frequency
                                if footswing_height > footswing_height_max:
                                    footswing_height_max = footswing_height
                                if pitch > pitch_max:
                                    pitch_max = pitch
                                if roll > roll_max:
                                    roll_max = roll
                                if stance_width > stance_width_max:
                                    stance_width_max = stance_width
                                if dx < dx_min:
                                    dx_min = dx
                                if dy < dy_min:
                                    dy_min = dy
                                if dyaw < dyaw_min:
                                    dyaw_min = dyaw
                                if body_height < body_height_min:
                                    body_height_min = body_height
                                if step_frequency < step_frequency_min:
                                    step_frequency_min = step_frequency
                                if footswing_height < footswing_height_min:
                                    footswing_height_min = footswing_height
                                if pitch < pitch_min:
                                    pitch_min = pitch
                                if roll < roll_min:
                                    roll_min = roll
                                if stance_width < stance_width_min:
                                    stance_width_min = stance_width
    
                        if len(dx_dict) == 0:
                            print('skip blank dict')
                        else: 
                            episode_dict['dx'] = dx_dict
                            episode_dict['dy'] = dy_dict
                            episode_dict['dyaw'] = dyaw_dict
                            episode_dict['body_height'] = body_height_dict
                            episode_dict['step_frequency'] = step_frequency_dict
                            episode_dict['gait_0'] = gait_0_dict
                            episode_dict['gait_1'] = gait_1_dict
                            episode_dict['gait_2'] = gait_2_dict
                            episode_dict['footswing_height'] = footswing_height_dict
                            episode_dict['pitch'] = pitch_dict
                            episode_dict['roll'] = roll_dict
                            episode_dict['stance_width'] = stance_width_dict
                            episode_dict['instruction'] = real_instruction_dict[task] + ' {} with a trotting gait'.format(rate)
                          
                        save_episode_path =  episode_path
                        all_dict[save_episode_path] = episode_dict

    dict_path = os.path.join(info_path, "commands.npy")
    np.save(dict_path, all_dict)

    print("dy != 0:", count_dy, "pitch != 0", count_pitch)

    dx_range = dx_max - dx_min
    dy_range = dy_max - dy_min
    dyaw_range = dyaw_max - dyaw_min
    body_height_range = body_height_max - body_height_min
    step_frequency_range = step_frequency_max - step_frequency_min
    footswing_height_range = footswing_height_max - footswing_height_min
    pitch_range = pitch_max - pitch_min
    roll_range = roll_max - roll_min
    stance_width_range = stance_width_max - stance_width_min

    command_space_low = np.array([dx_min, dy_min, dyaw_min, body_height_min, step_frequency_min, gait_min, gait_min, gait_min, footswing_height_min, pitch_min, stance_width_min])
    command_space_high = np.array([dx_max, dy_max, dyaw_max, body_height_max, step_frequency_max, gait_max, gait_max, gait_max, footswing_height_max, pitch_max, stance_width_max])

    range_dict = {}
    range_dict['dx_range'] = dx_range
    range_dict['dy_range'] = dy_range
    range_dict['dyaw_range'] = dyaw_range
    range_dict['body_height_range'] = body_height_range
    range_dict['step_frequency_range'] = step_frequency_range
    range_dict['footswing_height_range'] = footswing_height_range
    range_dict['pitch_range'] = pitch_range
    range_dict['roll_range'] = roll_range
    range_dict['stance_width_range'] = stance_width_range
    range_dict['dx_max'] = dx_max
    range_dict['dx_min'] = dx_min
    range_dict['dy_max'] = dy_max
    range_dict['dy_min'] = dy_min
    range_dict['dyaw_max'] = dyaw_max
    range_dict['dyaw_min'] = dyaw_min
    range_dict['body_height_max'] = body_height_max
    range_dict['body_height_min'] = body_height_min
    range_dict['step_frequency_max'] = step_frequency_max
    range_dict['step_frequency_min'] = step_frequency_min
    range_dict['footswing_height_max'] = footswing_height_max
    range_dict['footswing_height_min'] = footswing_height_min
    range_dict['pitch_max'] = pitch_max
    range_dict['pitch_min'] = pitch_min
    range_dict['roll_max'] = roll_max
    range_dict['roll_min'] = roll_min
    range_dict['stance_width_max'] = stance_width_max
    range_dict['stance_width_min'] = stance_width_min
    range_dict['command_space_low'] = command_space_low
    range_dict['command_space_high'] = command_space_high

    range_dict_path = os.path.join(info_path, "ranges.npy")
    np.save(range_dict_path, range_dict)
    
def merge_ranges(merged_info_path, sim_info_path, real_info_path):

    sim_ranges_ptah = os.path.join(sim_info_path, 'ranges.npy')
    real_ranges_path = os.path.join(real_info_path, 'ranges.npy')

    sim_ranges_data = np.load(sim_ranges_ptah, allow_pickle=True).item()
    real_ranges_data = np.load(real_ranges_path, allow_pickle=True).item()

    print('sim ranges: {}'.format([str(i) for i in sim_ranges_data['command_space_high'] - sim_ranges_data['command_space_low'].tolist()]))
    print('sim min: {}'.format([str(i) for i in sim_ranges_data['command_space_low'].tolist()]))
    print('sim max: {}'.format([str(i) for i in sim_ranges_data['command_space_high'].tolist()]))

    print('real ranges: {}'.format([str(i) for i in real_ranges_data['command_space_high'] - real_ranges_data['command_space_low'].tolist()]))
    print('real min: {}'.format([str(i) for i in real_ranges_data['command_space_low'].tolist()]))
    print('real max: {}'.format([str(i) for i in real_ranges_data['command_space_high'].tolist()]))

    full_dict = {}

    # 只考虑dx,dy,dyaw这几个参数进行merge,其他的参数都是默认使用sim的结果
    for i in sim_ranges_data.keys():
        if i in ['dx_max', 'dy_max', 'dyaw_max']:
            full_dict[i] = np.maximum(sim_ranges_data[i], real_ranges_data[i])
        elif i in ['dx_min', 'dy_min','dyaw_min']:
            full_dict[i] = np.minimum(sim_ranges_data[i], real_ranges_data[i])
        else:
            full_dict[i] = sim_ranges_data[i]
            
    # import pdb; pdb.set_trace()     
    # 更改dx,dy,dyaw的ranges的数值
    dx_range = full_dict['dx_max']  - full_dict['dx_min'] 
    dy_range = full_dict['dy_max']  - full_dict['dy_min']
    dyaw_range = full_dict['dyaw_max'] - full_dict['dyaw_min']
    full_dict['dx_range'] = dx_range
    full_dict['dy_range'] = dy_range
    full_dict['dyaw_range'] = dyaw_range

    first3_min = np.minimum( real_ranges_data['command_space_low'][:3], sim_ranges_data['command_space_low'][:3])
    first3_max = np.maximum( real_ranges_data['command_space_high'][:3], sim_ranges_data['command_space_high'][:3])
    full_dict['command_space_low'] =  np.concatenate((first3_min, sim_ranges_data['command_space_low'][3:]))
    full_dict['command_space_high'] = np.concatenate((first3_max, sim_ranges_data['command_space_high'][3:]))

    if not os.path.exists(merged_info_path):
        os.mkdir(merged_info_path)
    all_dict_path = os.path.join(merged_info_path, "ranges.npy")
    np.save(all_dict_path, full_dict)

def make_real_token_and_json(root_path,info_path,task,range_dict,json_list,image_id,sample_rate = 2):
    
    all_dict_path = os.path.join(info_path, "commands.npy")
    all_dict = np.load(all_dict_path, allow_pickle=True).item()

    dxmax = range_dict['dx_max']
    dxmin = range_dict['dx_min']
    dymax = range_dict['dy_max']
    dymin = range_dict['dy_min']
    dyawmax = range_dict['dyaw_max']
    dyawmin = range_dict['dyaw_min']
    body_height_max = range_dict['body_height_max'] 
    body_height_min = range_dict['body_height_min'] 
    step_frequency_max = range_dict['step_frequency_max'] 
    step_frequency_min = range_dict['step_frequency_min'] 
    footswing_height_max = range_dict['footswing_height_max'] 
    footswing_height_min = range_dict['footswing_height_min'] 
    pitch_max = range_dict['pitch_max'] 
    pitch_min = range_dict['pitch_min'] 
    stance_width_max = range_dict['stance_width_max'] 
    stance_width_min = range_dict['stance_width_min']

    task_path = os.path.join(root_path, task)
    if os.path.exists(task_path):
        episode_list = os.listdir(task_path)
        for episode in episode_list:
            episode_path = os.path.join(task_path, episode)
            episode_action_path = os.path.join(episode_path, "action")
            if os.path.exists(episode_path):                
                if episode_path in all_dict.keys():
                    commands_dict = all_dict[episode_path]
                    if len(commands_dict) == 0:
                        pass
                    else:
                        episode_length = len(commands_dict['dx'])
                    
                        for i in range(episode_length):
                            true_idx = i * sample_rate
                            img_idx = "{}".format(true_idx)
                            img_path = os.path.join(episode_path, f"image/{img_idx}.jpg")
                            if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
                                img_idx = "{}".format(true_idx)
                                img_path = os.path.join(episode_path, f"image/{img_idx}.jpg")
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
                            dx_token = int((dx-dxmin)/(dxmax-dxmin)*255) 
                            dy_token = int((dy-dymin)/(dymax-dymin+0.00001)*255)
                            dyaw_token = int((dyaw-dyawmin)/(dyawmax-dyawmin+0.00001)*255)
                            body_token = int(0) 
                            step_frequency_token = int(0)  
                            gait_0_token = int(0) 
                            gait_1_token = int(0) 
                            gait_2_token = int(0) 
                            footswing_height_token = int(0)  
                            pitch_token = int(0)  
                            stance_width_token = int(0) 

                            dict_json = {}
                            dict_json['id'] = str(image_id).rjust(12,'0')
                            dict_json['image'] = img_path

                            dict_json['conversations'] = []
                            human = {
                                'from': 'human',
                                'value': 'What action should the legged robot take to {}?\n'.format(instruction),
                                'type': 'real'
                            }
                            gpt = {
                                'from': 'gpt',
                                'value': '<0x04> {} {} {} {} {} {} {} {} {} {} {} {}'.format(terminate, dx_token,dy_token,dyaw_token,body_token,step_frequency_token,gait_0_token,gait_1_token,gait_2_token,footswing_height_token,pitch_token,stance_width_token)
                            }
                            dict_json['conversations'].append(human)
                            dict_json['conversations'].append(gpt)

                            json_list.append(dict_json) 
                            image_id += 1  
    return image_id, json_list

def get_real_json(real_instruction_dict, ranges_info_path, commands_info_path, json_path, real_path, sample_rate):

    # tasks in real datasets
    task_list = real_instruction_dict.keys()

    # laod ranges files
    range_dict_path = os.path.join(ranges_info_path, "ranges.npy")
    range_dict = np.load(range_dict_path, allow_pickle=True).item()
    
    image_id = 0
    # make json files
    for task in tqdm(task_list):
        json_list = []
        json_saved_path =  os.path.join(json_path, 'real')
        os.makedirs(json_saved_path, exist_ok=True)
        image_id, json_list = make_real_token_and_json(real_path, commands_info_path, task, range_dict, json_list, image_id, sample_rate)
        with open(json_saved_path + '/{}.json'.format(task), 'w') as f:
            json.dump(json_list, f)

def make_sim_token_and_json(root_path,root_path_unload,info_path,task,range_dict,json_list,image_id,sample_rate = 10):

    all_dict_path = os.path.join(info_path, "commands.npy")
    all_dict = np.load(all_dict_path, allow_pickle=True).item()

    dxmax = range_dict['dx_max']
    dxmin = range_dict['dx_min']
    dymax = range_dict['dy_max']
    dymin = range_dict['dy_min']
    dyawmax = range_dict['dyaw_max']
    dyawmin = range_dict['dyaw_min']
    body_height_max = range_dict['body_height_max'] 
    body_height_min = range_dict['body_height_min'] 
    step_frequency_max = range_dict['step_frequency_max'] 
    step_frequency_min = range_dict['step_frequency_min'] 
    footswing_height_max = range_dict['footswing_height_max'] 
    footswing_height_min = range_dict['footswing_height_min'] 
    pitch_max = range_dict['pitch_max'] 
    pitch_min = range_dict['pitch_min'] 
    stance_width_max = range_dict['stance_width_max'] 
    stance_width_min = range_dict['stance_width_min']

    task_path = os.path.join(root_path, task)
    if not os.path.exists(task_path):
        task_path = os.path.join(root_path_unload, task)

    if os.path.exists(task_path):
        episode_list = os.listdir(task_path)
        for episode in episode_list:
            episode_path = os.path.join(task_path, episode)
            episode_command_path = os.path.join(episode_path, "command")
            if os.path.exists(episode_path):
                # print(episode_path)
                if episode_path in all_dict.keys():
                    commands_dict = all_dict[episode_path]
                    episode_length = len(commands_dict['dx'])
                    for i in range(episode_length):
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
                        dx_token = int((dx-dxmin)/(dxmax-dxmin)*255) 
                        dy_token = int((dy-dymin)/(dymax-dymin+0.00001)*255)
                        dyaw_token = int((dyaw-dyawmin)/(dyawmax-dyawmin+0.00001)*255)
                        body_token = int((body_height-body_height_min)/(body_height_max - body_height_min+0.00001)*255) 
                        step_frequency_token = int((step_frequency-step_frequency_min)/(step_frequency_max-step_frequency_min)*255) 
                        gait_0_token = int(gait_0 * 255) 
                        gait_1_token = int(gait_1 * 255) 
                        gait_2_token = int(gait_2 * 255) 
                        footswing_height_token = int((footswing_height-footswing_height_min)/(footswing_height_max-footswing_height_min+0.00001)*255) 
                        pitch_token = int((pitch-pitch_min)/(pitch_max-pitch_min+0.00001)*255) 
                        stance_width_token = int((stance_width-stance_width_min)/(stance_width_max-stance_width_min+0.00001)*255) 

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
                            'value': '<0x04> {} {} {} {} {} {} {} {} {} {} {} {}'.format(terminate, dx_token,dy_token,dyaw_token,body_token,step_frequency_token,gait_0_token,gait_1_token,gait_2_token,footswing_height_token,pitch_token,stance_width_token)
                        }
                        dict_json['conversations'].append(human)
                        dict_json['conversations'].append(gpt)

                        json_list.append(dict_json) 
                        image_id += 1  
    return image_id, json_list

def get_sim_json(sim_instruction_dict, ranges_info_path, commands_info_path, json_path, sim_path, sim_path_unload, sample_rate):
    # tasks in sim datasets
    task_list = sim_instruction_dict.keys()

    # laod ranges files
    range_dict_path = os.path.join(ranges_info_path, "ranges.npy")
    range_dict = np.load(range_dict_path, allow_pickle=True).item()
    
    image_id = 0
    # make json files
    for task in tqdm(task_list):
        json_list = []
        json_saved_path =  os.path.join(json_path, 'sim')
        os.makedirs(json_saved_path, exist_ok=True)
        image_id, json_list = make_sim_token_and_json(sim_path, sim_path_unload, commands_info_path, task, range_dict, json_list, image_id, sample_rate)
        with open(json_saved_path + '/{}.json'.format(task), 'w') as f:
            json.dump(json_list, f)

def merged_single_json(sim_json_path, real_json_path, sim_instruction_dict, real_instruction_dict):

    sim_task_list = sim_instruction_dict.keys()
    data_type = 'sim'

    all_list = []
    all_json_path = os.path.join(sim_json_path,'sim.json')
    for task in tqdm(sim_task_list):
        json_name = '{}/{}.json'.format(data_type,task)
        json_load_path =  os.path.join(sim_json_path, json_name)
        with open(json_load_path, 'r') as f:
            cur_list = json.load(f)
            all_list = all_list + cur_list

    with open(all_json_path, 'w') as f:
        json.dump(all_list, f)

    real_task_list = real_instruction_dict.keys()
    data_type = 'real'

    all_list = []
    all_json_path = os.path.join(real_json_path,'real.json')
    for task in tqdm(real_task_list):
        json_name = '{}/{}.json'.format(data_type,task)
        json_load_path =  os.path.join(real_json_path, json_name)
        with open(json_load_path, 'r') as f:
            cur_list = json.load(f)
            all_list = all_list + cur_list

    with open(all_json_path, 'w') as f:
        json.dump(all_list, f)

def merged_multiple_json(merged_json_path, sim_instruction_dict, real_instruction_dict):

    # merged_single_json(merged_json_path, merged_json_path, sim_instruction_dict, real_instruction_dict)

    sim_path = os.path.join(merged_json_path,'sim.json')
    # 1007107
    with open(sim_path, 'r') as f:
        sim_list = json.load(f)
    # 92057
    real_path = os.path.join(merged_json_path,'real.json')
    with open(real_path, 'r') as f:
        real_list = json.load(f)

    
    # # 处理不同比例的仿真数据对结果带来的提升
    # ratios = [0.1,0.01,1]
    # for ratio in ratios:
    #     # merged_ratio = math.ceil(len(sim_list)/len(real_list))
    #     sim_list_copy = sim_list.copy()
    #     random.shuffle(sim_list_copy)
    #     new_sim_list = sim_list_copy[:int(ratio*len(sim_list))]
    #     print(len(new_sim_list))
    #     all_json_path = os.path.join(merged_json_path,'sim_ratio_{}.json'.format(ratio))
    #     all_list = new_sim_list
    #     with open(all_json_path, 'w') as f:
    #         json.dump(all_list, f)

    # # 处理仿真环境下单一任务的性能
    # sim_task_list = sim_instruction_dict.keys()
    # data_type = 'sim'
    # for single_type in ['go_to', 'go_avoid', 'go_through', 'stop', 'unload', 'crawl', 'distinguish']:
    #     all_list = []
    #     all_json_path = os.path.join(merged_json_path,'sim_{}.json'.format(single_type))
    #     for task in tqdm(sim_task_list):
    #         if single_type in task:
    #             json_name = '{}/{}.json'.format(data_type,task)
    #             json_load_path =  os.path.join(merged_json_path, json_name)
    #             with open(json_load_path, 'r') as f:
    #                 cur_list = json.load(f)
    #                 all_list = all_list + cur_list

    #     with open(all_json_path, 'w') as f:
    #         json.dump(all_list, f)

    # # 处理和真实数据集结合的任务
    # sim_list_copy = sim_list.copy()

    # real_vqa_path = "/dingpengxiang/Pengxiang/Quart/datasets/Merged/merged_json_path/blip_laion_cc_sbu_558k.json"
    # # 1007107
    # python_obj = json.loads(real_vqa_path)


    # with open(real_vqa_path, 'r') as f:
    #     real_vqa_list = json.load(real_vqa_path)
    # new_list = sim_list_copy + real_vqa_list
    # print(len(real_vqa_list))
    # all_json_path = os.path.join(merged_json_path,'merged.json')
    # all_list = new_list
    # with open(all_json_path, 'w') as f:
    #     json.dump(all_list, f)

if __name__ == "__main__":

    import ipdb; ipdb.set_trace()
    # 0206_test
    # 仿真和真实的ranges用同一个
    instructions_key = 'Full' #字典里套字典

    ROOT_PATH='/dingpengxiang/Pengxiang/Quart++'
    RAW_DATA_PATH='/wangdonglin'

    # get proprioception information
    proprioception_keys = ['joint_pos', 'joint_vel', 'body_linear_vel', 'body_angular_vel', 'contact_states', 'body_pos', 'body_quat']

    # get three ranges files of commands（real, sim, merged）

    sim_sample_rate = 10  # sim_command_dict里间隔的频率
    sim_path = os.path.join(RAW_DATA_PATH, 'sim_quadruped_data_v1')  #'/wangdonglin/sim_quadruped_data_v1'
    sim_path_unload = os.path.join(RAW_DATA_PATH, 'sim_quadruped_data_unload')  #'/wangdonglin/sim_quadruped_data_unload'
    sim_info_path = os.path.join(ROOT_PATH, 'datasets', instructions_key, 'sim_quadruped_data_info')  #'/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_quadruped_data_info'
    sim_instruction_dict = SIM_INSTRUCTION_DICT[instructions_key]

    real_sample_rate = 2
    real_path = os.path.join(RAW_DATA_PATH, 'quadruped_data_with_comand')  #'/wangdonglin/quadruped_data_with_comand'
    real_info_path = os.path.join(ROOT_PATH, 'datasets', instructions_key, 'real_quadruped_data_info') #'/dingpengxiang/Pengxiang/Quart++/datasets/Full/real_quadruped_data_info'
    real_instruction_dict = REAL_INSTRUCTION_DICT[instructions_key]

    merged_info_path = os.path.join(ROOT_PATH, 'datasets', instructions_key, 'merged_quadruped_data_info')  #'/dingpengxiang/Pengxiang/Quart++/datasets/Full/merged_quadruped_data_info'

    # for real data training
    real_json_path = os.path.join(ROOT_PATH, 'datasets', instructions_key, 'real_json_path')  #'/dingpengxiang/Pengxiang/Quart++/datasets/Full/real_json_path'
    # get_real_json(real_instruction_dict, real_info_path, real_info_path, real_json_path, real_path, real_sample_rate)
    
    # for simulation data training
    sim_json_path = os.path.join(ROOT_PATH, 'datasets', instructions_key, 'sim_json_path') #'/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_json_path'
    get_sim_json(sim_instruction_dict, sim_info_path, sim_info_path, sim_json_path, sim_path, sim_path_unload, sim_sample_rate)
    merged_single_json(sim_json_path, real_json_path, sim_instruction_dict, real_instruction_dict)  #多个json合一

    # for merged data training
    merged_json_path = os.path.join(ROOT_PATH, 'datasets', instructions_key, 'merged_json_path')
    get_real_json(real_instruction_dict, merged_info_path, real_info_path, merged_json_path, real_path, real_sample_rate)
    get_sim_json(sim_instruction_dict, merged_info_path, sim_info_path, merged_json_path, sim_path, sim_path_unload, sim_sample_rate)

    merged_multiple_json(merged_json_path, sim_instruction_dict, real_instruction_dict)

