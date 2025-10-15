import torch
import torch.nn as nn
from RVQ.residual_vq import RVQ,RVQ_n, RVQ_Seq
from RVQ.vq_Sequence import RVQ_Seq, RVQ_Seq_10
from RVQ.dataset import action_tokenize_dataset,action_tokenize_dataset_n
from torch.utils.data import DataLoader, Subset, random_split
from torch import optim
from tqdm import tqdm
import os
import wandb



current_dir = os.getcwd()
print("Current working directory:", current_dir)
root_path='/dingpengxiang/Pengxiang/Quart++'

os.chdir(root_path)


input_dim = 12  #The same as dataset $input_dim. If change one, change all.
timestep="n_Seq"  #one,n
step=10
# NetType='mlp'

split_mode='each'

# Define model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if timestep=="one":
    model = RVQ(layers_hidden=[2048, 2048, 2048, 512], input_dim=input_dim, K=512, num_quantizers=2, output_act=False)
    trainset = action_tokenize_dataset('/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_quadruped_data_info/vq_data.npy')
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
    trainset = action_tokenize_dataset_n(step, split_mode)

model.to(device)

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
# Define learning rate scheduler
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler = optim.lr_scheduler.ConstantLR(optimizer)


# 划分出testLoader
train_proportion=0.85
train_size = int(train_proportion * len(trainset))  # 训练集占90%
test_size = len(trainset) - train_size  # 测试集占10%
trainset, valset = random_split(trainset, [train_size, test_size])

trainloader = DataLoader(trainset, batch_size=1024, shuffle=True)
valloader = DataLoader(valset, batch_size=1024, shuffle=True)
print("set train proportion:{}, test proportion:{}".format(train_proportion,1-train_proportion))

# Define loss

# 初始化wandb
wandb.init(project="vq")
# wandb.init(project="vq", entity="tongxinyang")

# Define loss
criterion = nn.MSELoss()
best_val_loss = 10
best_epoch = 0

for epoch in range(50):
    # Train
    model.train()
    train_loss = 0
    codebook_loss_accum = 0
    recon_loss_accum = 0
    with tqdm(trainloader) as pbar:
        for i, targets in enumerate(pbar):
            targets = targets.to(device)
            optimizer.zero_grad()

            output, _, codebook_loss = model(targets)
            loss = model.loss_function(output, targets, codebook_loss)
            loss['loss'].backward()
            optimizer.step()

            # 累积损失
            train_loss += loss['loss'].item()
            codebook_loss_accum += loss['codebook_loss'].item()
            recon_loss_accum += loss['Reconstruction_Loss'].item()

            pbar.set_postfix(loss=loss['loss'].item(), recon_loss=loss['Reconstruction_Loss'].item(), codebook_loss=loss['codebook_loss'].item(), lr=optimizer.param_groups[0]['lr'])

    # 计算平均损失
    train_loss /= len(trainloader)
    codebook_loss_accum /= len(trainloader)
    recon_loss_accum /= len(trainloader)

    # 记录到wandb
    wandb.log({"Train Loss": train_loss, "Codebook Loss": codebook_loss_accum, "Reconstruction Loss": recon_loss_accum}, step=epoch)

    # 保存最佳模型
    if train_loss < best_val_loss:
        best_val_loss = train_loss
        best_epoch = epoch
        torch.save(model.state_dict(), "best_model.pth")

# 结束wandb运行
wandb.finish()