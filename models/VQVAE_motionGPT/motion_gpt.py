# Partially from https://github.com/Mael-zys/T2M-GPT
import torch
import torch.nn as nn
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from typing import List, Optional, Union
from resnet import Resnet1D
from quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset


class VQVae(nn.Module):
    def __init__(self,
                 nfeats: int,
                 quantizer: str = "ema_reset",
                 code_num=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "relu",
                 **kwargs) -> None:

        super().__init__()

        self.code_dim = code_dim

        self.encoder = Encoder(nfeats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        self.decoder = Decoder(nfeats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        if quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(code_num, code_dim, mu=0.99)
        elif quantizer == "orig":
            self.quantizer = Quantizer(code_num, code_dim, beta=1.0)
        elif quantizer == "ema":
            self.quantizer = QuantizeEMA(code_num, code_dim, mu=0.99)
        elif quantizer == "reset":
            self.quantizer = QuantizeReset(code_num, code_dim)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)  (batch_size, time_seq, channel*3)
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, features: Tensor):
        # Preprocess
        import pdb; pdb.set_trace()
        x_in = self.preprocess(features)
        print(f"Input shape after preprocessing: {x_in.shape}")
        
        # Encode
        x_encoder = self.encoder(x_in)
        print(f"Shape after encoding: {x_encoder.shape}")

        # Quantization
        x_quantized, loss, perplexity = self.quantizer(x_encoder)
        print(f"Shape after quantization: {x_quantized.shape}, Loss: {loss}, Perplexity: {perplexity}")

        # Decode
        x_decoder = self.decoder(x_quantized)
        print(f"Shape after decoding: {x_decoder.shape}")

        # Postprocess
        x_out = self.postprocess(x_decoder)
        print(f"Output shape after postprocessing: {x_out.shape}")

        return x_out, loss, perplexity

    def encode(
        self,
        features: Tensor,
    ) -> Union[Tensor, Distribution]:

        N, T, _ = features.shape
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1,
                                                x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)

        # latent, dist
        return code_idx, None

    def decode(self, z: Tensor):

        x_d = self.quantizer.dequantize(z)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()

        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


class Encoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))  #input_emb_width是进的时序sequence长度
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, 4, 1, 1),
                # nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t)
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         activation=activation,
                         norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation,
                         norm=norm), 
                         nn.Upsample(scale_factor=2,
                                                 mode='nearest'),
                nn.Conv1d(width, out_dim, 4, 1, 1))
                # nn.Conv1d(width, out_dim, 3, 1, 1))
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)



if __name__ == '__main__':
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 假设已经定义了 VQVae、Encoder 和 Decoder 类

    # 定义一些超参数
    nfeats = 12  # 输入特征的数量（即通道数）
    quantizer = "orig"
    code_num = 512
    code_dim = 512
    output_emb_width = 512  #中间的卷积核个数
    down_t = 3  #卷积层数
    stride_t = 1 #
    width = 512
    depth = 3
    dilation_growth_rate = 3

    # 创建 VQVae 模型并移动到 GPU
    model = VQVae(nfeats=nfeats,
                quantizer=quantizer,
                code_num=code_num,
                code_dim=code_dim,
                output_emb_width=output_emb_width,
                down_t=down_t,
                stride_t=stride_t,
                width=width,
                depth=depth,
                dilation_growth_rate=dilation_growth_rate).to(device)

    # 创建一条示例输入数据并移动到 GPU
    bs = 1  # batch size
    T = 5   # 时间步长（即序列长度）
    input_data = torch.randn(bs, T, nfeats).to(device)  # 形状为 (1, 5, 12)

    # 在模型上前向传播
    output, loss, perplexity = model(input_data)

    # 打印输出
    # print("Input shape:", input_data.shape)
    # print("Output shape:", output.shape)
    # print("Loss:", loss)
    # print("Perplexity:", perplexity)
