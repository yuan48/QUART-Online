# from init_path import SIM_INSTRUCTION_DICT, REAL_INSTRUCTION_DICT

# instructions_key = 'Full'
# sim_instruction_dict = SIM_INSTRUCTION_DICT[instructions_key]
# task_list = sim_instruction_dict.keys()

# # print(task_list)

# for task in task_list:
#     print("'"+task+"'"+',')


import numpy as np
path='/dingpengxiang/Pengxiang/Quart/datasets/Full/sim_quadruped_data_info/commands.npy'
data_dict = np.load(path, allow_pickle=True).item()
import pdb; pdb.set_trace()