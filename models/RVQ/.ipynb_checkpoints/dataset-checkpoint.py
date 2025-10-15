import numpy as np
import torch
from torch.utils.data import Dataset
from einops import rearrange

NoDis=False  #排除离散变量
gait_dict={"[0,0,0]":1, "[0.5,0,0]":2, "[0,0.5,0]":3, "[0,0,0.5]":4}
plan=1

class action_tokenize_dataset(Dataset):
    def __init__(self, file_path):
        super(action_tokenize_dataset, self).__init__()
        # file_path='/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_quadruped_data_info/vq_data.npy'
        self.data = np.load(file_path, allow_pickle=True)
        # self.data = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])

    def __getitem__(self, index):
        # target = torch.from_numpy(self.data[index]).to(torch.float32)
        arr = self.data[index]
        
        if NoDis==True:
            arr = np.concatenate((arr[1:5], arr[9:12]))               #array([0.62351555, 0. , 0.82436633, 0.09811682, 0.08490149,0. , 0.24994092])
            
        target = torch.from_numpy(arr).to(torch.float32)  #tensor([0.0000, 0.6235, 0.0000, 0.8244, 0.0981, 2.0000, 0.5000, 0.0000, 0.0000,0.0849, 0.0000, 0.2499])
        
        if plan==2:  #原12版本，frequency进sigmoid
            target[5]=torch.sigmoid(target[5])
        return target

    def __len__(self):
        return self.data.shape[0]
        # return len(self.data.shape)
    
    
    # np.array([[1,2,3,4,5,6,7,8,9,10,11,12],[6,7,8,9,10,9,8,7,6,5,4,3],[11,12,13,14,15,1,2,3,4,5,1,2]])
    # >>> data[1]
        # array([0.        , 0.62351555, 0.        , 0.82436633, 0.09811682,
        #     2.        , 0.5       , 0.        , 0.        , 0.08490149,
        #     0.        , 0.24994092])
        
        

class action_tokenize_dataset_n(Dataset):
    def __init__(self, file_path):
        super(action_tokenize_dataset_n, self).__init__()
        self.step_ahead=3   #try 3,5,8
        # file_path="/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_quadruped_data_info/vq_data_"+str(self.step_ahead)
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