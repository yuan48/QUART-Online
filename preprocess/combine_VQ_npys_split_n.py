import numpy as np
import os
from tqdm import tqdm
# a=[[1,2,3],[2,3,4]]
# b=[[5,6,7],[6,7,8]]
# np.save('1.npy', a)
# np.save('2.npy', b)

split_step=10

vq_npy_path = '/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_quadruped_data_info/vq_data'
vq_save_path = '/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_quadruped_data_info/'
info_path= '/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_quadruped_data_info'
all_npys = os.listdir(vq_npy_path)
num_npys = len(all_npys)
print("total:",num_npys)

# print(all_npys[0])
npy_list=[]
type="round"
temp_list=[]
final_list=[]

mode='each'
# import pdb; pdb.set_trace()

for task in tqdm(all_npys):
    npy_path = os.path.join(vq_npy_path, task)
    print("task: {} ".format(task))
    data=np.load(npy_path, allow_pickle=True)
    for round in data:
        for i in range(len(round)):
            temp_list.append(round[i])

            if mode=='separate':  #隔着切
                if i%split_step==split_step-1 and i!=0:  
                    final_list.append(temp_list)
                    temp_list=[]
                elif i+1==len(round):
                    final_list.append(round[-split_step:])
                    temp_list=[]

            elif mode=='each': #顺着切
                if i>=split_step-1: 
                    final_list.append(temp_list[-split_step:])

if mode=='each':
    save_path = os.path.join(vq_save_path, 'vq_data_step_'+str(split_step)+'_each.npy')
else:
    save_path = os.path.join(vq_save_path, 'vq_data_step_'+str(split_step)+'.npy')

np.save(save_path, final_list)
# print(data[0])

load_test=np.load(save_path, allow_pickle=True)  #np.ndarry类型,sim_all shape: (1822405, 12)
# (387844, 5, 12)
# (328053, 6, 12)
# 
import pdb; pdb.set_trace()     

#(626818, 3, 12)