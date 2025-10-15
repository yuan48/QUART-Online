import torch
import torch.nn as nn
from RVQ.residual_vq import RVQ,RVQ_n
from RVQ.dataset import action_tokenize_dataset,action_tokenize_dataset_n
from torch.utils.data import DataLoader, Subset, random_split
from torch import optim
from tqdm import tqdm
import os

current_dir = os.getcwd()
print("Current working directory:", current_dir)
root_path='/dingpengxiang/Pengxiang/Quart++'

os.chdir(root_path)


input_dim = 12  #The same as dataset $input_dim. If change one, change all.
timestep="n"  #one,n
NetType='mlp'

# Define model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if timestep=="one":
    model = RVQ(layers_hidden=[2048, 2048, 2048, 512], input_dim=input_dim, K=512, num_quantizers=2, output_act=False)
elif timestep=="n":
    model = RVQ_n(layers_hidden=[2048, 2048, 2048, 512], input_dim=input_dim, K=512, num_quantizers=2, output_act=False)
if timestep=="n_oneSlice":
    input_dim=36
    model = RVQ_n(layers_hidden=[2048, 2048, 2048, 512], input_dim=input_dim, K=512, num_quantizers=3, output_act=False)
model.to(device)

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
# Define learning rate scheduler
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler = optim.lr_scheduler.ConstantLR(optimizer)

if timestep=='one':
    trainset = action_tokenize_dataset('/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_quadruped_data_info/vq_data.npy')
elif timestep=='n':
    trainset = action_tokenize_dataset_n('/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_quadruped_data_info/vq_data_step_3.npy')
elif timestep=='n_oneSlice':
    trainset = action_tokenize_dataset_n('/dingpengxiang/Pengxiang/Quart++/datasets/Full/sim_quadruped_data_info/vq_data_step_3_oneSlice.npy')

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
best_val_loss = 10
best_epoch=0

for epoch in range(20):
    # Train
    model.train()
    with tqdm(trainloader) as pbar:
        for i, targets in enumerate(pbar):
            # import pdb; pdb.set_trace()

            targets = targets.view(-1, input_dim).to(device)
            optimizer.zero_grad()

            # if NetType=="conv":
            #     targets = targets.permute(1, 0).unsqueeze(1)
                # print(targets.size())  # torch.Size([dim, 1, 512])
            # elif timestep=='n':

            output, _, codebook_loss = model(targets)
            loss = model.loss_function(output, targets, codebook_loss)
            loss['loss'].backward()
            optimizer.step()
            pbar.set_postfix(loss=loss['loss'].item(), recon_loss=loss['Reconstruction_Loss'].item(), codebook_loss=loss['codebook_loss'].item(), lr=optimizer.param_groups[0]['lr'])
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for targets in valloader:
            targets = targets.view(-1, input_dim).to(device)
            if NetType=="conv":
                targets = targets.permute(1, 0).unsqueeze(1)

            output, _, codebook_loss = model(targets)
            loss = model.loss_function(output, targets, codebook_loss)
            val_loss += loss['loss'].item()
            
    val_loss /= len(valloader)

    # Update learning rate
    scheduler.step()

    print(
        f"Epoch {epoch + 1}, Val Loss: {val_loss}"
    )
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch=epoch+1
        print("best round saved!")
        best_model_state_dict = model.state_dict()

# path_name = timestep +
torch.save(best_model_state_dict, "state_dict/VQ/asa_vq_predict_3_step.pt")
print("best loss epoch {}, loss is {}".format(best_epoch, best_val_loss))
