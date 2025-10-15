import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from typing import Tuple, List, Optional

class ResidualVectorQuantizer(nn.Module):
    weight: torch.Tensor
    running_mean: torch.Tensor
    code_count: torch.Tensor

    def __init__(
        self,
        num_quantizers: int = 2,
        num_embeddings: int = 512,
        embedding_dim: int = 512,
        decay: float = 0.99,
        code_replace_threshold: float = 0.0001,
        eps: float = 1e-10,
    ) -> None:
        super().__init__()
        self.num_quantizers = num_quantizers
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.register_buffer("weight", torch.empty(num_quantizers, num_embeddings, embedding_dim))
        self.register_buffer("running_mean", torch.empty(num_quantizers, num_embeddings, embedding_dim))
        self.register_buffer("code_count", torch.empty(num_quantizers, num_embeddings))
        self.decay = decay
        self.eps = eps
        self.code_replace_threshold = code_replace_threshold
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.weight)
        self.running_mean[:] = self.weight
        nn.init.ones_(self.code_count)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        # input: [..., channel]
        n = self.num_quantizers
        codes = []
        r = input.type_as(self.running_mean).detach()  #r[1,512] 将input数据赋值给r，并与running_mean数据类型持平
        # import pdb; pdb.set_trace()

        with torch.no_grad():
            for i in range(n):
                w = self.weight[i]  #w[512,512] 是quantizers层embedding matrix中第i层的值
                # r: [..., num_embeddings]
                dist = torch.cdist(r, w) #dist[1,512] 计算这一层embedding与所有输入的距离
                k = torch.argmin(dist, axis=-1) # k为第i次距离矩阵中，距离最近的向量索引值
                codes.append(k) #将索引加入codes中，总共会加入num_quantizers次
                self._update_averages(i, r, k)  
                r = r - F.embedding(k, w) #k是index，w是embedding
        quantized = input - r   #quantized [1, 512]，也即input减去剩余的残差，作为最终loss计算的output。算的是quantizers这一步到底和input（最开始的）的msr
        commitment_loss = torch.mean(torch.square(input - quantized.detach())) #loss是一个值。两个loss，一个是quantizer这里input_quan和input_quan-残差的codebook_loss，一个是外面input和decode(input_quan-残差)的reconstruction_loss
        self.weight.data[:] = self.running_mean / torch.unsqueeze(self.eps + self.code_count, axis=-1)
        return quantized, torch.stack(codes, input.ndim - 1), commitment_loss
        #这里codes本来是[tensor([0]), tensor([267])]，stack改为tensor([[0, 267]])
    
    @torch.cuda.amp.autocast(enabled=False)
    def quantize(self, input: torch.Tensor):
        # input: [..., chennel]
        n = self.num_quantizers
        codes = []
        r = input.type_as(self.running_mean).detach()  #这里的r可以理解为剩余残差
        with torch.no_grad():
            for i in range(n):
                w = self.weight[i]
                # r: [..., num_embeddings]
                dist = torch.cdist(r, w)
                k = torch.argmin(dist, axis=-1)  #取当前最近的weight向量编号，共取quantizers次
                codes.append(k)
                r = r - F.embedding(k, w)
        return torch.stack(codes, input.ndim - 1)  #返回的是codes

    def dequantize(self, input: torch.Tensor, n: Optional[int] = None) -> torch.Tensor:
        # input: [batch_size, num_quantizers]
        if n is None:
            n = input.shape[-1]
        assert 0 < n <= self.num_quantizers
        res = 0
        with torch.no_grad():
            for i in range(n):
                k = input[:, i]
                w = self.weight[i]
                res += F.embedding(k, w)
        return res

    def _update_averages(self, i: int, r: torch.Tensor, k: torch.Tensor) -> None:
        # https://arxiv.org/pdf/1906.00446.pdf
        # Generating Diverse High-Fidelity Images with VQ-VAE-2
        # 2.1 Vector Quantized Variational AutoEncode

        # k: [...]
        one_hot_k = F.one_hot(torch.flatten(k), self.num_embeddings).type_as(self.code_count)
        code_count_update = torch.mean(one_hot_k, axis=0)
        self.code_count[i].lerp_(code_count_update, 1 - self.decay)

        # r: [..., embedding_dim]
        r = r.reshape(-1, self.embedding_dim)
        running_mean_update = (one_hot_k.T @ r) / r.shape[0]
        self.running_mean[i].lerp_(running_mean_update, 1 - self.decay)

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def replace_vectors(self) -> int:
        # https://arxiv.org/pdf/2107.03312.pdf
        # C. Residual Vector Quantizer:

        # The original paper replaces with an input frame randomly
        # sampled within the current batch.
        # Here we replace with random average of running mean instead.
        num_replaced = torch.sum(self.code_count < self.code_replace_threshold).item()
        if num_replaced > 0:
            for i in range(self.num_quantizers):
                mask = self.code_count[i] < self.code_replace_threshold
                # mask: [num_quantizers, num_embeddings]
                w = torch.rand_like(self.code_count[i])
                w /= torch.sum(w)
                self.running_mean[i, mask] = w.type_as(self.running_mean) @ self.running_mean[i]
                self.code_count[i, mask] = w.type_as(self.code_count) @ self.code_count[i]

        return num_replaced

    @torch.no_grad()
    def calc_entropy(self) -> float:
        p = self.code_count / (self.eps + torch.sum(self.code_count, axis=-1, keepdim=True))
        return -torch.sum(torch.log(p) * p).item() / self.num_quantizers
    

class RVQ(torch.nn.Module):
    def __init__(
        self,
        layers_hidden=[2048, 2048, 2048, 512],
        input_dim=11,
        K=512,
        num_quantizers=2,
        output_act=False,
    ):
        super(RVQ, self).__init__()
        self.layers_hidden = layers_hidden
        self.input_dim = input_dim
        self.K = K
        self.num_quantizers = num_quantizers
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        self.encoder.add_module("encoder_input", nn.Linear(input_dim, layers_hidden[0]))
        for i in range(len(layers_hidden) - 1):
            self.encoder.add_module(f"encoder_act_{i}", nn.ReLU())
            self.encoder.add_module(f"encoder_hidden_{i}", nn.Linear(layers_hidden[i], layers_hidden[i + 1]))
            
            self.decoder.add_module(f"decoder_hidden_{i}", nn.Linear(layers_hidden[len(layers_hidden) - 1 - i], layers_hidden[len(layers_hidden) - 2 - i]))
            self.decoder.add_module(f"decoder_act_{i}", nn.ReLU())
        self.decoder.add_module("decoder_output", nn.Linear(layers_hidden[0], input_dim))
        if output_act:
            self.decoder.add_module("decoder_output_act", nn.Tanh())
        self.RVQ = ResidualVectorQuantizer(num_quantizers=num_quantizers, num_embeddings=K, embedding_dim=layers_hidden[-1])

    def encode(self, input):
        posterior_dist = self.encoder(input)
        return posterior_dist

    def decode(self, z):
        priori_dist = self.decoder(z)
        return priori_dist
# decoder的loss通过self.decode(z)传回去，用self.decode(quantized)与input做mse。这里quantized是`input_quantizer-残差`的值。
    def forward(self, input, **kwargs):
        encoding = self.encode(input)
        z, codes, codebook_loss = self.RVQ(encoding)  #z相当于原始过完encode后，input_quan减去残差的值。code是z再经过embedding的idx
        # import pdb; pdb.set_trace()
        return self.decode(z), codes, codebook_loss #z是原始
    
    def tokenize(self, input, **kwargs):
        encoding = self.encode(input)
        return self.RVQ.quantize(encoding)
    
    def detokenize(self, input: torch.Tensor, n: Optional[int] = None):
        z = self.RVQ.dequantize(input, n)
        return self.decoder(z)

    def loss_function(self,
                      input, 
                      ouput, 
                      codebook_loss):

        recons_loss = F.mse_loss(ouput, input)

        loss = recons_loss + codebook_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'codebook_loss':codebook_loss}

class RVQ_n(torch.nn.Module):
    def __init__(
        self,
        layers_hidden=[2048, 2048, 2048, 512],
        input_dim=11,
        K=512,
        num_quantizers=2,
        output_act=False,
    ):
        super(RVQ_n, self).__init__()
        self.layers_hidden = layers_hidden
        self.input_dim = input_dim
        self.K = K
        self.num_quantizers = num_quantizers
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        self.encoder.add_module("encoder_input", nn.Linear(input_dim, layers_hidden[0]))
        for i in range(len(layers_hidden) - 1):
            self.encoder.add_module(f"encoder_act_{i}", nn.ReLU())
            self.encoder.add_module(f"encoder_hidden_{i}", nn.Linear(layers_hidden[i], layers_hidden[i + 1]))
            
            self.decoder.add_module(f"decoder_hidden_{i}", nn.Linear(layers_hidden[len(layers_hidden) - 1 - i], layers_hidden[len(layers_hidden) - 2 - i]))
            self.decoder.add_module(f"decoder_act_{i}", nn.ReLU())
        self.decoder.add_module("decoder_output", nn.Linear(layers_hidden[0], input_dim))
        if output_act:
            self.decoder.add_module("decoder_output_act", nn.Tanh())
        self.RVQ = ResidualVectorQuantizer(num_quantizers=num_quantizers, num_embeddings=K, embedding_dim=layers_hidden[-1])

    def encode(self, input):
        posterior_dist = self.encoder(input)
        return posterior_dist

    def decode(self, z):
        priori_dist = self.decoder(z)
        return priori_dist

    def forward(self, input, **kwargs):
        encoding = self.encode(input)
        z, codes, codebook_loss = self.RVQ(encoding)
        return self.decode(z), codes, codebook_loss
    
    def tokenize(self, input, **kwargs):
        encoding = self.encode(input)
        return self.RVQ.quantize(encoding)
    
    def detokenize(self, input: torch.Tensor, n: Optional[int] = None):
        z = self.RVQ.dequantize(input, n)
        return self.decoder(z)

    def loss_function(self,
                      input, 
                      ouput, 
                      codebook_loss):

        recons_loss = F.mse_loss(ouput, input)

        loss = recons_loss + codebook_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'codebook_loss':codebook_loss}



class RVQ_Seq(torch.nn.Module):
    def __init__(
        self,
        layers_hidden=[2048, 2048, 2048, 512],
        input_dim=11,
        K=512,
        num_quantizers=2,
        output_act=False,
    ):
        super(RVQ_Seq, self).__init__()
        self.layers_hidden = layers_hidden
        self.input_dim = input_dim
        self.K = K
        self.num_quantizers = num_quantizers
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        self.encoder.add_module("encoder_input", nn.Linear(input_dim, layers_hidden[0]))
        for i in range(len(layers_hidden) - 1):
            self.encoder.add_module(f"encoder_act_{i}", nn.ReLU())
            self.encoder.add_module(f"encoder_hidden_{i}", nn.Linear(layers_hidden[i], layers_hidden[i + 1]))
            
            self.decoder.add_module(f"decoder_hidden_{i}", nn.Linear(layers_hidden[len(layers_hidden) - 1 - i], layers_hidden[len(layers_hidden) - 2 - i]))
            self.decoder.add_module(f"decoder_act_{i}", nn.ReLU())
        self.decoder.add_module("decoder_output", nn.Linear(layers_hidden[0], input_dim))
        if output_act:
            self.decoder.add_module("decoder_output_act", nn.Tanh())
        self.RVQ = ResidualVectorQuantizer(num_quantizers=num_quantizers, num_embeddings=K, embedding_dim=layers_hidden[-1])

    def encode(self, input):
        posterior_dist = self.encoder(input)
        return posterior_dist

    def decode(self, z):
        priori_dist = self.decoder(z)
        return priori_dist

    def forward(self, input, **kwargs):
        encoding = self.encode(input)
        z, codes, codebook_loss = self.RVQ(encoding)
        return self.decode(z), codes, codebook_loss
    
    def tokenize(self, input, **kwargs):
        encoding = self.encode(input)
        return self.RVQ.quantize(encoding)
    
    def detokenize(self, input: torch.Tensor, n: Optional[int] = None):
        z = self.RVQ.dequantize(input, n)
        return self.decoder(z)

    def loss_function(self,
                      input, 
                      ouput, 
                      codebook_loss):

        recons_loss = F.mse_loss(ouput, input)

        loss = recons_loss + codebook_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'codebook_loss':codebook_loss}