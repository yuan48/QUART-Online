# import os

# root_path='/dingpengxiang/Pengxiang/Quart++'

# os.chdir(root_path)
# current_dir = os.getcwd()
# print("Current working directory:", current_dir)

import sys
sys.path.append('/dingpengxiang/Pengxiang/Quart++')
import torch
import torch.nn as nn
from models.RVQ.residual_vq import RVQ
from models.RVQ.dataset import action_tokenize_dataset
# from residual_vq import RVQ
import numpy as np
import torch

np.set_printoptions(suppress=True)


input_dim=12
model = RVQ(layers_hidden=[2048, 2048, 2048, 512], input_dim=input_dim, K=512, num_quantizers=3, output_act=False)
model.load_state_dict(torch.load('/dingpengxiang/Pengxiang/Quart++/state_dict/VQ/asa_vq_NoTanh_12_3.pt'))
    #  [terminate, dx,    dy,      dyaw,    body_height, step_freq,  gait0, gait1,   gait2,   foot_height,  roll, stance_width]
input=np.array([[0,2.077075719833374,0.0,0.0,0.0461319237947464,3.0,0.5,0.0,0.0,0.08394856005907059,0.0,0.10386524349451065]])
# origin:[0,      2.0770 , 0.0,     0.0,     0.0461,       3.0,       0.5,     0.0,     0.0,   0.0839,       0.0,     0.1038]
# 11: [0,         0.9907,  0.0042,  0.009,  -0.1837,      1.0000,   0.5049, -0.0040, -0.0003,  0.1417,    -0.0039,    0.2842]
# 12 :[[ 0.0044,  0.9951,  0.0073,  0.0019, -0.1335,      1.0000,   0.5098, -0.0010, 0.0012,   0.1292,    -0.0081,    0.2652]], grad_fn=<TanhBackward0>)

# input=np.array([0.0000, 0.6235, 0.0000, 0.8244, 0.0981, 2.0000, 0.5000, 0.0000, 0.0000,0.0849, 0.0000, 0.2499]) #12
# output and input difference: [[-0.00825815  0.00144185 -0.0022808   0.00327342  0.00408067  1.  0.00266954  0.00470723 -0.000525   -0.05320197 -0.0009979   0.0689262 ]]

# input=np.array([[0.000000,0.260305,0.000000,-0.210055,0.000000,3.000000,0.500000,0.000000,0.000000,0.083949,0.000000,0.250000]])
#output and input difference: [[-0.00453802 -0.00814204  0.00083273  0.02461901 -0.0190923   2.00000018  -0.00445181 -0.00049812  0.00310421 -0.02372175 -0.00279328  0.00028181]]

# input=np.array([[0.000000,0.473518,0.000000,-1.051083,-0.035468,3.000000,0.500000,0.000000,0.000000,0.085623,0.000000,0.244159]])
#output and input difference: [[-0.00351033 -0.03315754  0.00424287 -0.08330807 -0.04366243  2.   0.00039813 -0.00193765  0.00353548 -0.02239764 -0.00354457  0.0138945 ]]

 #  [terminate, dx,    dy,      dyaw,    body_height, step_freq,  gait0, gait1,   gait2,   foot_height,  roll, stance_width]
# input=np.array([[1.,0.481940,0.0,-0.035478, 0.000000,    2.00,       0.500,      0.0 ,0.0   ,  0.136154,     0.000000,    0.250000]])
# output:[ 0.9951,  0.4978,  0.0019, -0.0466, -0.0542,  1.0000,  0.4725,  0.0160,    0.0049,  0.1070,  0.0160,  0.2440]
# output and input difference: [[ 0.00488824 -0.01581621 -0.00191371  0.01114377  0.05415646  1.   0.02746645 -0.01599047 -0.00486193  0.02917792 -0.01600009  0.00595854]]
# output and input difference, 2:[[ 0.03091067  0.23910513 -0.00592329  0.01517074  0.01112035  0.04550374   -0.00305074  0.01097695  0.00434136  0.02769502  0.01446679  0.02765878]]
# output and input difference, 3:[[ 0.00770009  0.03636616 -0.00788642  0.01814775  0.04403643  0.02410042   0.01243204 -0.00140638  0.00018185  0.02328629  0.00513553  0.02163497]]
# input = np.array([input[0:]])
input_vector = torch.from_numpy(input).to(torch.float32)


output, codes, codebook_loss = model(input_vector)
print("input:",input)
print("ouput:",output)
print("output and input difference:",input-output.detach().numpy())
print("codebook_loss",codebook_loss)

# input_vector = torch.tensor(input_vector).to(device)

# encode=model.encode(input_vector)
# tokenize=model.tokenize(input_vector)
# detokenize=model.detokenize(tokenize)
# print("after encode",encode)
# print("after tokenize:",tokenize)
# print("after detokenize",detokenize)  

# #return self.decode(z), codes, codebook_loss
# output, codes, codebook_loss = model(input_vector)
# print("ouput:",output)
# print("codes:",codes)


