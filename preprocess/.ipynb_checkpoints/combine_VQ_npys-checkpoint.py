import numpy as np
import os
from tqdm import tqdm
a=[[1,2,3],[2,3,4]]
b=[[5,6,7],[6,7,8]]
np.save('1.npy', a)
np.save('2.npy', b)

vq_npy_path = '/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_quadruped_data_info/vq_data'
info_path= '/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_quadruped_data_info'
all_npys = os.listdir(vq_npy_path)
num_npys = len(all_npys)
print("total:",num_npys)

# print(all_npys[0])
npy_list=[]
type="single"

# import pdb; pdb.set_trace()

for task in tqdm(all_npys):
    npy_path = os.path.join(vq_npy_path, task)
    print("task: {} ".format(task))
    data=np.load(npy_path, allow_pickle=True)
    for round in data:
        if type=="round":
            npy_list.append(round)
        elif type=="single":
            for step in round:
                npy_list.append(step)

save_path = os.path.join(info_path, 'vq_data_'+type+'.npy')
np.save(save_path, npy_list)
# print(data[0])

load_test=np.load(save_path, allow_pickle=True)  #np.ndarry类型,sim_all shape: (1822405, 12)

import pdb; pdb.set_trace()     

