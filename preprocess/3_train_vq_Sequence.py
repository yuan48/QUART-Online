#This file is for training the vq model
import torch
import torch.nn as nn
import sys
import os

# Add project root to path for imports
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_PATH)

from models.RVQ.residual_vq import RVQ
from models.RVQ.vq_Sequence import RVQ_Seq, RVQ_Seq_10
from models.RVQ.dataset import action_tokenize_dataset, action_tokenize_dataset_n
from torch.utils.data import DataLoader, Subset, random_split
from torch import optim
from tqdm import tqdm

input_dim = 12  #The same as dataset $input_dim. If change one, change all.
timestep="n_Seq"  #one, n, n_Seq
step=10

split_mode='each'

# Define model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if timestep=="one":
    model = RVQ(layers_hidden=[2048, 2048, 2048, 512], input_dim=input_dim, K=512, num_quantizers=2, output_act=False)
    trainset = action_tokenize_dataset(os.path.join(ROOT_PATH, 'datasets/Full/sim_quadruped_data_info/vq_data.npy'))
elif timestep=="n":  # MLP-based n-sequence compression (not used in final version)
    model = RVQ_n(layers_hidden=[2048, 2048, 2048, 512], input_dim=input_dim, K=512, num_quantizers=2, output_act=False)
    trainset = action_tokenize_dataset_n(step)
    save_pt_name= 'asa_vq_predict_'+str(step)+'_step.pt'
elif timestep=="n_oneSlice": # Flatten all timesteps to 1D (not used in final version)
    input_dim=36
    model = RVQ_n(layers_hidden=[2048, 2048, 2048, 512], input_dim=input_dim, K=512, num_quantizers=3, output_act=False)
    trainset = action_tokenize_dataset_n(os.path.join(ROOT_PATH, 'datasets/Full/sim_quadruped_data_info/vq_data_step_3_oneSlice.npy'))
elif timestep=="n_Seq":  # Final version: sequence-based temporal modeling
    split_mode='Each'
    if step==3:
        model = RVQ_Seq(layers_hidden=[512, 512], input_dim=input_dim, K=512, num_quantizers=2, output_act=False)
    elif step==5:
        model = RVQ_Seq(layers_hidden=[512, 512, 512, 512], input_dim=input_dim, K=512, num_quantizers=2, output_act=False)
    elif step == 10:
        model = RVQ_Seq_10(layers_hidden=[512, 512, 512, 512], input_dim=input_dim, K=512, num_quantizers=2, output_act=False)

    if split_mode=='Each':  # Each timestep with n future steps (final approach)
        save_pt_name='Sequence_vq_'+str(step)+'_each_conv.pt'
    else:   # Sample every n steps (not used in final version)
        save_pt_name='Sequence_vq_'+str(step)+'_interval.pt'
    trainset = action_tokenize_dataset_n(step, split_mode)

model.to(device)

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
# Define learning rate scheduler
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler = optim.lr_scheduler.ConstantLR(optimizer)


# Split dataset into train and validation sets
train_proportion=0.85
train_size = int(train_proportion * len(trainset))  # 85% for training
test_size = len(trainset) - train_size  # 15% for validation
trainset, valset = random_split(trainset, [train_size, test_size])

trainloader = DataLoader(trainset, batch_size=1024, shuffle=True)
valloader = DataLoader(valset, batch_size=1024, shuffle=True)
print("set train proportion:{}, test proportion:{}".format(train_proportion,1-train_proportion))

# Define loss
criterion = nn.MSELoss()
best_val_loss = 10
best_epoch=0

for epoch in range(50):
    # Train
    model.train()
    with tqdm(trainloader) as pbar:
        for i, targets in enumerate(pbar):
            # import pdb; pdb.set_trace()

            # targets = targets.view(-1, input_dim).to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            # if NetType=="conv":
            #     targets = targets.permute(1, 0).unsqueeze(1)
                # print(targets.size())  # torch.Size([dim, 1, 512])
            # elif timestep=='n':

            output, code, codebook_loss = model(targets)
            loss = model.loss_function(output, targets, codebook_loss)

            # Optional: calculate perplexity metric
            # reshaped_data = code.reshape((-1, 1))
            # encodings = torch.zeros(reshaped_data.shape[0], 512).cuda()
            # encodings.scatter_(1, code, 1)
            # avg_probs = torch.mean(encodings, dim=0)
            # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))


            loss['loss'].backward()
            optimizer.step()
            pbar.set_postfix(loss=loss['loss'].item(), recon_loss=loss['Reconstruction_Loss'].item(), codebook_loss=loss['codebook_loss'].item(), lr=optimizer.param_groups[0]['lr'])
            # pbar.set_postfix(loss=loss['loss'].item(), recon_loss=loss['Reconstruction_Loss'].item(), codebook_loss=loss['codebook_loss'].item(),perplexity=perplexity, lr=optimizer.param_groups[0]['lr'])
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for targets in valloader:
            # targets = targets.view(-1, input_dim).to(device)
            targets = targets.to(device)

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
        print("best round saved! current best_val_loss:",best_val_loss)
        best_model_state_dict = model.state_dict()
        save_dir = os.path.join(ROOT_PATH, 'vq_state_dict', 'VQ')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_model_state_dict, os.path.join(save_dir, save_pt_name))


save_dir = os.path.join(ROOT_PATH, 'vq_state_dict', 'VQ')
os.makedirs(save_dir, exist_ok=True)
torch.save(best_model_state_dict, os.path.join(save_dir, save_pt_name))
print("Model saved to:", os.path.join(save_dir, save_pt_name))
print("best loss epoch {}, loss is {}".format(best_epoch, best_val_loss))
