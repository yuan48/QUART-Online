import torch.nn as nn
import torch.nn.functional as F
import torch

class AutoEncoder(nn.Module):
    def __init__(self, input_dim,output_dim, layers_hidden):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        self.encoder.add_module("encoder_input", nn.Linear(input_dim, layers_hidden[0]))
        for i in range(len(layers_hidden) - 1):
            self.encoder.add_module(f"encoder_act_{i}", nn.ReLU())
            self.encoder.add_module(f"encoder_hidden_{i}", nn.Linear(layers_hidden[i], layers_hidden[i + 1]))
            
            self.decoder.add_module(f"decoder_hidden_{i}", nn.Linear(layers_hidden[len(layers_hidden) - 1 - i], layers_hidden[len(layers_hidden) - 2 - i]))
            self.decoder.add_module(f"decoder_act_{i}", nn.ReLU())
        self.decoder.add_module("decoder_output", nn.Linear(layers_hidden[0], output_dim))
        
    def encoder(self,x):
        encoded = self.encoder(x)
        return encoded
    
    def decoder(self,x):
        decoded=decoded = self.decoder(encoded)
        output=F.softmax(decoded, dim=1)
        max_prob, max_idx = torch.max(output, dim=1)
        return max_idx
        

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        output=F.softmax(decoded, dim=1)
        max_prob, max_idx = torch.max(output, dim=1)
        return max_idx




# 创建多个encoder和decoder
dis_layer_dict = {'ter':[256,256,1],'fre':[256,256,1],'gait':[256,256,1]}
output_dim={'ter':2,'fre':3,'gait':4}
test_encoder=AutoEncoder(input_dim=10, output_dim=output_dim['ter'],layers_hidden=dis_layer_dict['ter'])
encoder_decoder_dict = {}
# for key, layers in dis_layer_dict.items():
#     encoder_decoder_dict[key+'_encoder'] = AutoEncoder(input_dim=10, layers_hidden=layers)
#     encoder_decoder_dict[key+'_decoder'] = AutoEncoder(input_dim=layers[-1], layers_hidden=list(reversed(layers)))

# 使用例子
x = torch.randn(1, 10)  # 输入数据
output1,output2=test_encoder(x)
print(x,output)

# print(test_encoder.encoder(x))

# for key, model in encoder_decoder_dict.items():
#     output = model(x)
#     print(f'{key} output shape: {output.shape}')
