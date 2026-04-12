import numpy as np
import torch
from torch.utils.data import Dataset
from einops import rearrange

NoDis=False  
gait_dict={"[0,0,0]":1, "[0.5,0,0]":2, "[0,0.5,0]":3, "[0,0,0.5]":4}
plan=1

class action_tokenize_dataset(Dataset):
    def __init__(self, file_path):
        super(action_tokenize_dataset, self).__init__()
        self.data = np.load(file_path, allow_pickle=True)

    def __getitem__(self, index):
        arr = self.data[index]
        
        if NoDis==True:
            arr = np.concatenate((arr[1:5], arr[9:12]))           
            
        target = torch.from_numpy(arr).to(torch.float32)  
        
        if plan==2:  
            target[5]=torch.sigmoid(target[5])
        return target

    def __len__(self):
        return self.data.shape[0]

        

class action_tokenize_dataset_n(Dataset):
    def __init__(self, step_ahead, mode='None'):
        super(action_tokenize_dataset_n, self).__init__()
        self.step_ahead=step_ahead   #try 3,5,8
        if mode == 'separate' or mode == 'None':
            file_path="./datasets/Full/sim_quadruped_data_info/vq_data_step_"+str(self.step_ahead)+'.npy'
        else:
            file_path="./datasets/Full/sim_quadruped_data_info/vq_data_step_"+str(self.step_ahead)+'_each.npy'

        self.data = np.load(file_path, allow_pickle=True)
        
    def __getitem__(self, index):
        arr=np.array(self.data[index])
        target = torch.from_numpy(arr).to(torch.float32)
        return target
        
    def __len__(self):
        return self.data.shape[0]
    
if __name__ == '__main__':
    test_dataset=action_tokenize_dataset('/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_quadruped_data_info/vq_data_round.npy')
    import pdb; pdb.set_trace()