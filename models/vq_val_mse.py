    
import sys
sys.path.append('/dingpengxiang/Pengxiang/Quart++')
import torch
import torch.nn as nn
from models.RVQ.residual_vq import RVQ
from models.RVQ.vq_Sequence import RVQ_Seq, RVQ_Seq_10
from models.RVQ.dataset import action_tokenize_dataset,action_tokenize_dataset_n
# from residual_vq import RVQ
import numpy as np

from torch.utils.data import DataLoader, Subset, random_split
from torch import optim
from tqdm import tqdm
import os


def calculate_mse(original, reconstructed):
    return torch.mean((original - reconstructed) ** 2)

def calculate_psnr(original, reconstructed):
    mse = calculate_mse(original, reconstructed)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # 假设像素值范围是 [0, 1]
    return 10 * torch.log10(max_pixel**2 / mse)

def calculate_mae(original, reconstructed):
    return torch.mean(torch.abs(original - reconstructed))

def calculate_nrmse(original, reconstructed):
    mse = calculate_mse(original, reconstructed)
    return torch.sqrt(mse) / (torch.max(original) - torch.min(original))


# 新增的函数
def calculate_rmse(original, reconstructed):
    return torch.sqrt(calculate_mse(original, reconstructed))

def calculate_mape(original, reconstructed):
    return torch.mean(torch.abs((original - reconstructed) / original)) * 100

def calculate_uqi(original, reconstructed):
    n = original.size(0)
    Q1 = torch.sum((original - reconstructed) ** 2)
    Q2 = torch.sum((original - torch.mean(original)) ** 2)
    Q3 = torch.sum((reconstructed - torch.mean(reconstructed)) ** 2)
    if Q2 == 0 or Q3 == 0:
        return float('inf') if Q1 == 0 else 0
    return 1 - (Q1 / (Q2 * Q3)) ** 0.5

def calculate_aki(original, reconstructed):
    n = original.size(0)
    e = original - reconstructed
    alpha = torch.mean(e) / (torch.std(original) + 1e-10)
    beta = torch.mean((e - alpha) ** 2) / (torch.std(reconstructed) ** 2 + 1e-10)
    return alpha ** 2 + beta

input_dim=12

timestep='one'
step=10
split_mode='Each'

# import pdb; pdb.set_trace()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if timestep=="one":
    model = RVQ(layers_hidden=[2048, 2048, 2048, 512], input_dim=input_dim, K=512, num_quantizers=2, output_act=False)
    trainset = action_tokenize_dataset('/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_quadruped_data_info/vq_data.npy')
    save_pt_name='asa_vq_NoTanh_12_2.pt'
elif timestep=="n":
    model = RVQ_n(layers_hidden=[2048, 2048, 2048, 512], input_dim=input_dim, K=512, num_quantizers=2, output_act=False)
    trainset = action_tokenize_dataset_n(step)
    save_pt_name= 'asa_vq_predict_'+str(step)+'_step.pt'
elif timestep=="n_oneSlice":
    input_dim=36
    model = RVQ_n(layers_hidden=[2048, 2048, 2048, 512], input_dim=input_dim, K=512, num_quantizers=3, output_act=False)
    trainset = action_tokenize_dataset_n('/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_quadruped_data_info/vq_data_step_3_oneSlice.npy')
elif timestep=="n_Seq":
    split_mode='Each'
    if step==3:
        model = RVQ_Seq(layers_hidden=[512, 512], input_dim=input_dim, K=512, num_quantizers=2, output_act=False)
    elif step==5:
        model = RVQ_Seq(layers_hidden=[512, 512, 512, 512], input_dim=input_dim, K=512, num_quantizers=2, output_act=False)
    elif step == 10:
        model = RVQ_Seq_10(layers_hidden=[512, 512, 512, 512], input_dim=input_dim, K=512, num_quantizers=2, output_act=False)

    if split_mode=='Each':
        save_pt_name='Sequence_vq_'+str(step)+'_each_conv.pt'
    else:
        save_pt_name='Sequence_vq_'+str(step)+'_interval.pt'
    
    if step==5:
        save_pt_name='Sequence_vq_'+str(step)+'_each_conv_paper.pt'
    trainset = action_tokenize_dataset_n(step, split_mode)

print("LOADING:", save_pt_name)
model.load_state_dict(torch.load('/dingpengxiang/Pengxiang/Quart++/state_dict/VQ/'+save_pt_name))

model.to(device)
    
# 划分出testLoader
train_proportion=0.85
train_size = int(train_proportion * len(trainset))  # 训练集占90%
test_size = len(trainset) - train_size  # 测试集占10%
trainset, valset = random_split(trainset, [train_size, test_size])

trainloader = DataLoader(trainset, batch_size=1024, shuffle=True)
valloader = DataLoader(valset, batch_size=1024, shuffle=True)
print("set train proportion:{}, test proportion:{}".format(train_proportion,1-train_proportion))

# Define loss
criterion = nn.MSELoss()
np.set_printoptions(suppress=True) #禁止以科学计数法显示小的浮点数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loss = 0
train_rec = 0
train_code = 0
mse_loss = 0
psnr_loss = 0
mae_loss = 0
nrmse_loss = 0
rmse_loss = 0
mape_loss = 0
uqi_loss = 0
aki_loss = 0

print("Training set size: {}, Validation set size: {}".format(len(trainset), len(valset)))
print("Training started...")
with torch.no_grad():
    with tqdm(trainloader) as pbar:
        for i, targets in enumerate(pbar):
            # import pdb; pdb.set_trace()

            # targets = targets.view(-1, input_dim).to(device)
            targets = targets.to(device)
            # optimizer.zero_grad()

            # if NetType=="conv":
            #     targets = targets.permute(1, 0).unsqueeze(1)
                # print(targets.size())  # torch.Size([dim, 1, 512])
            # elif timestep=='n':

            output, _, codebook_loss = model(targets)
            loss = model.loss_function(output, targets, codebook_loss)
            # loss['loss'].backward()
            # optimizer.step()
            pbar.set_postfix(loss=loss['loss'].item(), recon_loss=loss['Reconstruction_Loss'].item(), codebook_loss=loss['codebook_loss'].item())
            train_loss += loss['loss'].item()
            train_rec += loss['Reconstruction_Loss'].item()
            train_code += loss['codebook_loss'].item()
            mse = calculate_mse(targets, output)
            psnr = calculate_psnr(targets, output)
            mae = calculate_mae(targets, output)
            nrmse = calculate_nrmse(targets, output)

            rmse = calculate_rmse(targets, output)
            mape = calculate_mape(targets, output)
            uqi = calculate_uqi(targets, output)
            aki = calculate_aki(targets, output)
            
            mse_loss += mse.item()
            psnr_loss += psnr.item()
            mae_loss += mae.item()
            nrmse_loss += nrmse.item()
            rmse_loss += rmse.item()
            mape_loss += mape.item()
            uqi_loss += uqi.item()
            aki_loss += aki.item()
            

train_loss/=len(trainloader)
train_rec/=len(trainloader)
train_code/=len(trainloader)
mse_loss /= len(trainloader)
psnr_loss /= len(trainloader)
mae_loss /= len(trainloader)
nrmse_loss /= len(trainloader)


rmse_loss /= len(trainloader)
mape_loss /= len(trainloader)
uqi_loss /= len(trainloader)
aki_loss /= len(trainloader)

# 输出训练集上的损失值
print("Training loss:", train_loss)
# 输出训练集上的重构损失值
print("Training reconstruction loss:", train_rec)
# 输出训练集上的码本损失值
print("Training codebook loss:", train_code)
# 输出训练集上的平均均方误差
print("Training MSE:", mse_loss)
# 输出训练集上的平均峰值信噪比
print("Training PSNR:", psnr_loss)
# 输出训练集上的平均绝对误差
print("Training MAE:", mae_loss)
# 输出训练集上的平均归一化均方误差
print("Training NRMSE:", nrmse_loss)


print("TRAINING loss: {:.4f}, RECONST: {:.4f},CODEBOOK_LOSS: {:.4f}, MSE: {:.4f}, PSNR: {:.2f}, MAE: {:.4f}, NRMSE: {:.4f}".format(train_loss, train_rec, train_code,mse_loss, psnr_loss, mae_loss, nrmse_loss))
print(f"RMSE: {rmse_loss}")
print(f"MAPE: {mape_loss}%")
print(f"UQI: {uqi_loss}")
print(f"AKI: {aki_loss}")

print("===========")
# Validation
model.eval()
val_loss = 0
val_rec = 0
val_code = 0
mse_loss = 0
psnr_loss = 0
mae_loss = 0
nrmse_loss = 0

rmse_loss = 0
mape_loss = 0
uqi_loss = 0
aki_loss = 0

print("Validation started...")
with torch.no_grad():
    for targets in valloader:
        # targets = targets.view(-1, input_dim).to(device)
        targets = targets.to(device)

        output, _, codebook_loss = model(targets)
        loss = model.loss_function(output, targets, codebook_loss)
        val_loss += loss['loss'].item()
        val_rec += loss['Reconstruction_Loss'].item()
        val_code += loss['codebook_loss'].item()
        mse = calculate_mse(targets, output)
        psnr = calculate_psnr(targets, output)
        mae = calculate_mae(targets, output)
        nrmse = calculate_nrmse(targets, output)

        rmse = calculate_rmse(targets, output)
        mape = calculate_mape(targets, output)
        uqi = calculate_uqi(targets, output)
        aki = calculate_aki(targets, output)

        mse_loss += mse.item()
        psnr_loss += psnr.item()
        mae_loss += mae.item()
        nrmse_loss += nrmse.item()
        rmse_loss += rmse.item()
        mape_loss += mape.item()
        uqi_loss += uqi.item()
        aki_loss += aki.item()
        
val_loss /= len(valloader)
val_rec /= len(valloader)
val_code /= len(valloader)
mse_loss /= len(valloader)
psnr_loss /= len(valloader)
mae_loss /= len(valloader)
nrmse_loss /= len(valloader)
# 计算验证集上的平均值
rmse_loss /= len(valloader)
mape_loss /= len(valloader)
uqi_loss /= len(valloader)
aki_loss /= len(valloader)

# 输出验证集上的损失值
print("Validation loss:", val_loss)
# 输出验证集上的重构损失值
print("Validation reconstruction loss:", val_rec)
# 输出验证集上的码本损失值
print("Validation codebook loss:", val_code)
# 输出验证集上的平均均方误差
print("Validation MSE:", mse_loss)
# 输出验证集上的平均峰值信噪比
print("Validation PSNR:", psnr_loss)
# 输出验证集上的平均绝对误差
print("Validation MAE:", mae_loss)
# 输出验证集上的平均归一化均方误差
print("Validation NRMSE:", nrmse_loss)

print("Validation loss: {:.4f}, RECONST: {:.4f},CODEBOOK_LOSS: {:.4f}, MSE: {:.4f}, PSNR: {:.2f}, MAE: {:.4f}, NRMSE: {:.4f}".format(val_loss, val_rec, val_code,mse_loss, psnr_loss, mae_loss, nrmse_loss))

print(f"Validation RMSE: {rmse_loss}")
print(f"Validation MAPE: {mape_loss}%")
print(f"Validation UQI: {uqi_loss}")
print(f"Validation AKI: {aki_loss}")