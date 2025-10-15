import numpy as np
import torch
from torch.utils.data import Dataset

file_path='/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_quadruped_data_info/vq_data.npy'
data = np.load(file_path, allow_pickle=True)
np.savetxt('/dingpengxiang/Pengxiang/Quart++/datasets/data.txt', data, fmt='%f', delimiter=',')

target = torch.from_numpy(data[80:100]).to(torch.float32)
# print(target)
import pdb; pdb.set_trace()