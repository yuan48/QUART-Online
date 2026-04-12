#This file is for preprocessing the vq training data into .npy format
import numpy as np
import os
from tqdm import tqdm

# Use relative paths for cross-platform compatibility
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
split_step = 10

# Input: vq_Dataset_{task}.npy files from step 1
vq_npy_path = os.path.join(ROOT_PATH, 'datasets', 'Full', 'sim_quadruped_data_info', 'vq_data')
# Output: vq_data_step_10_each.npy for VQ model training
vq_save_path = os.path.join(ROOT_PATH, 'datasets', 'Full', 'sim_quadruped_data_info')

all_npys = os.listdir(vq_npy_path) if os.path.exists(vq_npy_path) else []
num_npys = len(all_npys)
print("total npy files:", num_npys)

npy_list=[]
temp_list=[]
final_list=[]

mode='each'  # 'each' for sliding window, 'separate' for interval sampling

for task in tqdm(all_npys):
    npy_path = os.path.join(vq_npy_path, task)
    print("task: {} ".format(task))
    data=np.load(npy_path, allow_pickle=True)
    for round in data:
        for i in range(len(round)):
            temp_list.append(round[i])

            if mode=='separate':  # Interval sampling (not used in final version)
                if i%split_step==split_step-1 and i!=0:  
                    final_list.append(temp_list)
                    temp_list=[]
                elif i+1==len(round):
                    final_list.append(round[-split_step:])
                    temp_list=[]

            elif mode=='each': # Sliding window (final approach)
                if i>=split_step-1: 
                    final_list.append(temp_list[-split_step:])

if mode=='each':
    save_path = os.path.join(vq_save_path, 'vq_data_step_'+str(split_step)+'_each.npy')
else:
    save_path = os.path.join(vq_save_path, 'vq_data_step_'+str(split_step)+'.npy')

np.save(save_path, final_list)
print("Saved to:", save_path)
print("Total sequences:", len(final_list))

# Verification
load_test=np.load(save_path, allow_pickle=True)
print("Data shape:", load_test.shape if hasattr(load_test, 'shape') else f"list of {len(load_test)} sequences")
