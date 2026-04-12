import sys

from init_path import SIM_INSTRUCTION_DICT, REAL_INSTRUCTION_DICT

import os
import json
from tqdm import tqdm



def merged_single_json(sim_json_path, sim_instruction_dict, data_type, json_save_name):
    """
    Merge all task JSON files into a single training JSON file.
    
    Args:
        sim_json_path: Base path for JSON files
        sim_instruction_dict: Dictionary of task instructions (keys = task names)
        data_type: Subdirectory name (e.g., 'sim_vq_ahead_10_seq')
        json_save_name: Output filename (e.g., 'sim_ahead_10_seq.json')
    """
    sim_task_list = sim_instruction_dict.keys()

    all_list = []
    all_json_path = os.path.join(sim_json_path, json_save_name)  
    for task in tqdm(sim_task_list):
        json_name = '{}/{}.json'.format(data_type, task)
        json_load_path = os.path.join(sim_json_path, json_name)
        
        if not os.path.exists(json_load_path):
            print(f"Warning: {json_load_path} not found, skipping...")
            continue
            
        with open(json_load_path, 'r') as f:
            cur_list = json.load(f)
            all_list = all_list + cur_list

    with open(all_json_path, 'w') as f:
        json.dump(all_list, f)
    print("Saved merged JSON to:", all_json_path)
    print("Total samples:", len(all_list))


if __name__ == "__main__":
    #Change the following directory to your local path
    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ahead_step=10  
    instructions_key = 'Full' 
    json_save_name='sim_ahead_'+str(ahead_step)+'_seq.json'  #Change save path: ./datasets/Full/sim_json_path/sim_ahead_10_seq.json
    data_type= 'sim_vq_ahead_'+str(ahead_step)+'_seq'  #Change loading path: ./datasets/Full/sim_json_path/sim_vq_ahead_10_seq/xxx.json

    sim_instruction_dict = SIM_INSTRUCTION_DICT[instructions_key]
    sim_json_path = os.path.join(ROOT_PATH, 'datasets', instructions_key, 'sim_json_path') 

    merged_single_json(sim_json_path, sim_instruction_dict, data_type, json_save_name) 

