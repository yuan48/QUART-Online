import argparse
import sys
import os
import isaacgym
import torch
import subprocess
import importlib

current_path = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(project_path)


current_script_directory = os.path.dirname(os.path.abspath(__file__))
print("path:",current_script_directory)
# sys.path.append(current_script_directory)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from test_quart import Quart_online, Quart
# from play_auto_config_clean import seen_task_lists, unseen_task_lists,sup_lists
from play_go_isaac_run import play_go1_plus, formatted_datetime
from count_success import count_success


def load_task_settings(task):
    item = task.split('_')

    if len(item) < 4:  # such as: "unload_purple_traybox"
        task_setting={
            'action' : item[0],
            'color' : item[1],
            'object' : item[2]
        }
    elif len(item) > 4:  # such as: "go_through"
        task_setting={
            'action' : item[0] + '_' + item[1] ,
            'color' : item[2],
            'object' : item[3] + ' ' + item[4]
        }
    else:  # such as: "go_to"
        task_setting={
            'action' : item[0] + '_' + item[1],
            'color' : item[2],
            'object' : item[3]
        }

    # obj num
    if 'crawl' in task:
        task_setting['num'] = 1
    else:
        task_setting['num'] = 2
    task_setting['name'] = task

    return task_setting

if __name__ == "__main__":
    save_statistic=True   
    train_mode=True
    model='Quart_online'
    parser = argparse.ArgumentParser(description='Process some parameters.')

    parser.add_argument('--ckpt_path', type=str, required=True, help='Root path for checkpoints')
    parser.add_argument('--test_type', type=str, required=True, help='Type of test (seen/unseen)')
    parser.add_argument('--save_folder', type=str, required=True, help='Folder to save results')
    parser.add_argument('--detype', type=str, required=True, help='float32 or float16')
    parser.add_argument('--dataset_type', type=str, required=True, help='Type of dataset')
    parser.add_argument('--model_name', type=str, required=True, help='Quart, Quart_online')
    parser.add_argument('--headless', type=str, required=True, help='view')
    parser.add_argument('--env_num', type=int, required=True, help='isaac env num')
    parser.add_argument('--task_lists', type=str, required=True, help='task config')
    parser.add_argument('--vocab_path', type=str, required=True, help='vocab config')
    parser.add_argument('--vq_ckpt_path', type=str, required=True, help='vq path')


    # 解析参数
    args = parser.parse_args()
    args.headless = args.headless.lower() == 'true'
    # task_lists=ast.literal_eval(settings['task_lists'])  
    task_module = importlib.import_module(args.task_lists)
    task_lists = task_module.task_lists 

    print("CONFIG SETTING: ", args)
    print("*****************")
    print("\\\ current checkpoint:",args.ckpt_path)
 
    if args.model_name == 'Quart':
        print("---------Using original QUART---------")
        model = Quart(args)     
  
    elif args.model_name == 'Quart_online':
        model = Quart_online(args)  

    print('task_lists: ',task_lists)
    for idx,task in enumerate(task_lists):
        print(f'\033[34m||| The {idx}th task begin: {task}\033[0m')
        task_cfg = load_task_settings(task)
        print("Task settings:", task_cfg)

        play_go1_plus(model, args, task_cfg)

    if save_statistic:
        import pdb; pdb.set_trace()
        result_path = os.path.join(args.save_folder,formatted_datetime)
        count_success(args, task_cfg, result_path)
