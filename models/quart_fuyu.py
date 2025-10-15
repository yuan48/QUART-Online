try:
    from .modeling_persimmon import PersimmonForCausalLM
    print("Using local PersimmonForCausalLM with Flash Attention")
except ImportError:
    from transformers import PersimmonForCausalLM
    print("Using transformers PersimmonForCausalLM without Flash Attention")

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.parameter import Parameter
from torch.nn.modules.activation import MultiheadAttention
from collections import OrderedDict


import wandb

from transformers import FuyuPreTrainedModel, FuyuConfig, FuyuForCausalLM, AutoModelForCausalLM, AutoConfig

from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

from transformers.utils.doc import add_start_docstrings_to_model_forward

# from models.transformer import TwoWayTransformer

FUYU_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# class Quart2Config(FuyuConfig):
#     model_type = "quart2"

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class PersimmonRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )        

class ReLUSquaredActivation(nn.Module):
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """

    def forward(self, input):
        relu_applied = nn.functional.relu(input)
        squared = torch.square(relu_applied)
        return squared

class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)

ACT2CLS = {
    "relu2": ReLUSquaredActivation,
}

ACT2FN = ClassInstantier(ACT2CLS)

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, d_model=4096, max_len=16384):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term2 = torch.pow(torch.tensor(10000.0),torch.arange(0, d_model, 2)/d_model)
        div_term1 = torch.pow(torch.tensor(10000.0),torch.arange(1, d_model, 2)/d_model)
        #高级切片方式，即从0开始，两个步长取一个。即奇数和偶数位置赋值不一样。直观来看就是每一句话的
        pe[:, 0::2] = torch.sin(position * div_term2)
        pe[:, 1::2] = torch.cos(position * div_term1)
        #这里是为了与x的维度保持一致，释放了一个维度
        self.register_buffer('pe', pe)

    def forward(self, num_tokens):
        """Generate positional encoding for a grid of the specified size."""
        pe = self.pe[:num_tokens]
        return pe


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=4096, dropout=0.0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model,eps=1e-05)
        self.norm2 = nn.LayerNorm(d_model,eps=1e-05)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = ACT2FN['relu2']

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class Quart2FuyuForCausalLM(FuyuPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`FuyuDecoderLayer`]

    Args:
        config: FuyuConfig
    """

    def __init__(self, config: FuyuConfig, exp_id):
        super().__init__(config, exp_id)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor
        # "model_type": "persimmon"
        # self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.language_model = PersimmonForCausalLM._from_config(config.text_config)

        self.vision_embed_tokens = nn.Linear(
            config.patch_size * config.patch_size * config.num_channels, config.hidden_size
        )
        if 'v1' in exp_id:
            # v1 的policy head
            self.policy_head = nn.Linear(
                config.hidden_size + 37, 256
            )
        elif 'v2' in exp_id:
            # v2 的policy head
            self.policy_head = nn.Linear(
                config.hidden_size + 37, 256 * 12
            )
        elif 'v3' in exp_id:
            self.projection_v =  nn.Linear(config.hidden_size, config.hidden_size)
            self.projection_p = nn.Linear(37, config.hidden_size)
            self.fusion_vp = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8)
            self.policy_head = nn.Linear(
                config.hidden_size, 256 * 12
            )
        elif 'v4' in exp_id:
            self.projection_v =  nn.Linear(config.hidden_size, config.hidden_size)
            # self.projection_p = nn.Linear(37, config.hidden_size)
            self.fusion_v = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8)
            self.policy_head = nn.Linear(
                config.hidden_size, 256 * 12
            )
        elif 'v5' in exp_id:
            # self.projection_v =  nn.Linear(config.hidden_size, config.hidden_size)
            # self.projection_p = nn.Linear(37, config.hidden_size)
            self.query_tokens = nn.Parameter(torch.zeros(1, 1, config.hidden_size))  # 假设ViT模型的隐藏大小为4096
            self.fusion_v = TransformerEncoderLayer(d_model=config.hidden_size, nhead=8)
            self.policy_head = nn.Linear(
                config.hidden_size, 12
            )
        elif 'v6' in exp_id:
            self.projection_v =  nn.Linear(config.hidden_size, config.hidden_size)
            # self.projection_p = nn.Linear(37, config.hidden_size)
            self.fusion_v = TransformerEncoderLayer(d_model=config.hidden_size, nhead=8)
            self.policy_head = nn.Linear(
                config.hidden_size, 12
            )
        elif 'v7' in exp_id:
            # self.cls_token [12, 4096]
            self.cls_token = nn.Embedding(12, config.hidden_size)
            self.transformer = TwoWayTransformer(depth=2, embedding_dim=4096, mlp_dim=2048,num_heads=8)
            self.policy_head = nn.Linear(
                config.hidden_size, 256
            )  
            self.pe_layer = PositionEmbeddingRandom(d_model=4096, max_len=4096)


        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def gather_continuous_embeddings(
        self,
        word_embeddings: torch.Tensor,
        continuous_embeddings: List[torch.Tensor],
        image_patch_input_indices: torch.Tensor,
    ) -> torch.Tensor:
        """This function places the continuous_embeddings into the word_embeddings at the locations
        indicated by image_patch_input_indices. Different batch elements can have different numbers of continuous
        embeddings.

        Args:
            word_embeddings: Tensor of word embeddings. Shape: [b, s, h]
            continuous_embeddings:
                Tensor of continuous embeddings. The length of the list is the batch size. Each entry is
            shape [num_image_embeddings, hidden], and num_image_embeddings needs to match the number of non-negative
            indices in image_patch_input_indices for that batch element.
            image_patch_input_indices: Tensor of indices of the image patches in the input_ids tensor. Shape: [b, s]
        """
        if not (word_embeddings.shape[0] == len(continuous_embeddings)):
            raise ValueError(
                f"Batch sizes must match! Got {len(continuous_embeddings)=} and {word_embeddings.shape[0]=}"
            )

        output_embeddings = word_embeddings.clone()
        for batch_idx in range(word_embeddings.shape[0]):
            dst_indices = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0]
            src_indices = image_patch_input_indices[batch_idx][dst_indices]
            if src_indices.shape[0] > continuous_embeddings[batch_idx].shape[0]:
                raise ValueError(
                    f"Number of continuous embeddings {continuous_embeddings[batch_idx].shape=} does not match "
                    f"number of continuous token ids {src_indices.shape=} in batch element {batch_idx}."
                )
            output_embeddings[batch_idx, dst_indices] = continuous_embeddings[batch_idx][src_indices].to(output_embeddings.dtype)

        return output_embeddings

    @add_start_docstrings_to_model_forward(FUYU_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        image_patches: torch.Tensor = None,  # [batch_size, num_total_patches, patch_size_ x patch_size x num_channels ]
        image_patches_indices: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        local_rank: Optional[bool] = None,
        proprioceptions: Optional[torch.Tensor] = None,
        extra_labels: Optional[torch.Tensor] = None,
        exp_id: Optional[str] = None,
        c_labels: Optional[torch.Tensor] = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        # monitor_gpu_memory_usage()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        # print('*********training_phrase***************')
        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids) # torch.Size([32, 394, 4096])
            # print(inputs_embeds.size())
            if image_patches is not None and past_key_values is None:
                # 视觉embedding
                patch_embeddings = self.vision_embed_tokens(image_patches.to(self.vision_embed_tokens.weight.dtype)) # torch.Size([32, 352, 4096])
                # 替换视觉token为image embedding
                inputs_embeds = self.gather_continuous_embeddings(
                    word_embeddings=inputs_embeds,
                    continuous_embeddings=patch_embeddings,
                    image_patch_input_indices=image_patches_indices,
                ) # torch.Size([32, 394, 4096])
        
        # outputs = self.language_model(
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     output_attentions=output_attentions,
        #     use_cache=use_cache,
        # )
        # output = outputs['logits']
        # print('*********inputs_embeds***************')
        # print(inputs_embeds.size()) # torch.Size([32, 394, 4096])
        # with torch.inference_mode(mode=True):
        outputs = self.language_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # hidden_states (Batchsize, num_token, 4096)
        hidden_states = outputs[0]
        num_token = hidden_states.size(1)
        num_batch = hidden_states.size(0)
        num_dimen = hidden_states.size(2)

        ###############################
        # # raw_version
        # # (Batchsize, num_token, 262144)
        if 'v0' in exp_id:
            logits = self.language_model.lm_head(hidden_states)
        ###############################
        # v1
        elif 'v1' in exp_id:
            # proprioceptions（Batchsize, 37）--> (Batchsize, num_token , 37）
            proprioceptions = torch.unsqueeze(proprioceptions,dim=1)
            proprioceptions = proprioceptions.repeat((1, num_token, 1))
            # muluyan_fea (Batchsize, num_token, 4096+37)
            mul_fea = torch.cat((hidden_states, proprioceptions),dim=2)
            # logits (Batchsize, num_token, 256)
            logits = self.policy_head(mul_fea) 
            # print(logits.size())  
        # ################################ 

        # ###############################
        # v2
        elif 'v2' in exp_id:
            # hidden_states (Batchsize, num_token, 4096）--> (Batchsize, 4096)
            hidden_states = torch.mean(hidden_states,dim=1)
            # mul_fea (Batchsize, 4096+37)
            mul_fea = torch.cat((hidden_states, proprioceptions),dim=1)
            # logits (Batchsize, 256*12) --> (Batchsize, 12, 256)
            logits = torch.reshape(self.policy_head(mul_fea),(num_batch, 12, 256))
            # print(logits.size())  

        elif 'v3' in exp_id:
            # proprioceptions (Batchsize, 37) --> (Batchsize, 4096)
            proprioceptions = torch.unsqueeze(self.projection_p(proprioceptions),dim=1)
            # hidden_states (Batchsize, num_token, 4096）--> (Batchsize, num_token, 4096)
            hidden_states = self.projection_v(hidden_states)
            # mul_fea (Batchsize, num_token + 1, 4096)
            mul_fea = torch.cat((hidden_states, proprioceptions),dim=1)
            # mul_fea (Batchsize, num_token + 1, 4096)
            mul_fea = self.fusion_vp(mul_fea)
            # p_fea (Batchsize, 4096)
            p_fea = mul_fea[:,-1,:]
            # v_fea (Batchsize, 4096)
            v_fea = torch.mean(mul_fea[:,:-1,:],dim=1)
            # logits (Batchsize, 256*12) --> (Batchsize, 12, 256)
            logits = torch.reshape(self.policy_head(p_fea + v_fea),(num_batch, 12, 256))

        elif 'v4' in exp_id:
            # proprioceptions (Batchsize, 37) --> (Batchsize, 4096)
            # proprioceptions = torch.unsqueeze(self.projection_p(proprioceptions),dim=1)
            # hidden_states (Batchsize, num_token, 4096）--> (Batchsize, num_token, 4096)
            hidden_states = self.projection_v(hidden_states)
            # mul_fea (Batchsize, num_token + 1, 4096)
            # mul_fea = torch.cat((hidden_states, proprioceptions),dim=1)
            # mul_fea (Batchsize, num_token, 4096)
            mul_fea = self.fusion_v(hidden_states)
            # p_fea (Batchsize, 4096)
            # p_fea = mul_fea[:,-1,:]
            # v_fea (Batchsize, 4096)
            v_fea = torch.mean(mul_fea[:,:,:],dim=1)
            # logits (Batchsize, 256*12) --> (Batchsize, 12, 256)
            logits = torch.reshape(self.policy_head(v_fea),(num_batch, 12, 256))

        elif 'v5' in exp_id:
            # query_tokens (1, 1, 4096) --> (Batchsize, 1, 4096)
            query_tokens = self.query_tokens.expand(num_batch, -1, -1)
            # hidden_states (Batchsize, num_token, 4096）--> (Batchsize, num_token, 4096)
            # hidden_states = self.projection_v(hidden_states)
            # mul_fea (Batchsize, num_token + 1, 4096)
            mul_fea = torch.cat((query_tokens, hidden_states),dim=1)
            # mul_fea (Batchsize, num_token + 1, 4096)
            mul_fea = self.fusion_v(mul_fea)
            # q_fea (Batchsize, 4096)
            q_fea = mul_fea[:,0,:]
            # logits (Batchsize, 12)
            logits = self.policy_head(q_fea)

        elif 'v6' in exp_id:
            hidden_states = self.projection_v(hidden_states)
            # mul_fea (Batchsize, num_token + 1, 4096)
            # mul_fea = torch.cat((hidden_states, proprioceptions),dim=1)
            # mul_fea (Batchsize, num_token, 4096)
            mul_fea = self.fusion_v(hidden_states)
            # p_fea (Batchsize, 4096)
            # p_fea = mul_fea[:,-1,:]
            # v_fea (Batchsize, 4096)
            v_fea = torch.mean(mul_fea[:,:,:],dim=1)
            # logits (Batchsize, 12)
            logits = self.policy_head(v_fea)
        elif 'v7' in exp_id:
            # self.cls_token (12, 4096) --> (Batchsize, 12, 4096)
            out_tokens = self.cls_token.weight.unsqueeze(0).expand(num_batch, -1, -1)
            # hidden_states (Batchsize, num_token, 4096）
            # hidden_states_rotary (Batchsize, num_token, 4096）
            img_pe = self.pe_layer(num_token).unsqueeze(0)
            hidden_states_rotary = img_pe.expand(num_batch, -1, -1)
            # mul_fea (Batchsize, num_token, 4096)
            # hs (Batchsize, 12, 4096)
            hs, mul_fea = self.transformer(hidden_states, hidden_states_rotary, out_tokens)
            # logits (Batchsize, 12, 256)
            logits = self.policy_head(hs)
            outputs['logits'] = logits
        
        if labels is not None:
            ##############################################
            # # raw_version
            if 'v0' in exp_id:
                output = logits[:, :-1, :] # [2, num_token, 262144]
                labels = labels[:, 1:] # [2, num_token]
                # labels 来做这个判断没毛病,判断label结束的标识
                indexs = torch.nonzero(labels.flatten()==71013).squeeze() #llava是2 

                c_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction = 'none')(torch.flatten(output, start_dim=0, end_dim=1), labels.flatten())
                outputs['loss'] = torch.sum(c_loss) / torch.sum(labels != -100)
                outputs['logits'] = logits 

                terminate = torch.mean(c_loss[indexs-12]) 
                dx_token = torch.mean(c_loss[indexs-11])
                dy_token = torch.mean(c_loss[indexs-10])
                dyaw_token = torch.mean(c_loss[indexs-9])
                body_token = torch.mean(c_loss[indexs-8])
                step_frequency_token = torch.mean(c_loss[indexs-7])
                gait_0_token = torch.mean(c_loss[indexs-6])
                gait_1_token = torch.mean(c_loss[indexs-5])
                gait_2_token = torch.mean(c_loss[indexs-4])
                footswing_height_token = torch.mean(c_loss[indexs-3])
                pitch_token = torch.mean(c_loss[indexs-2])
                stance_width_token = torch.mean(c_loss[indexs-1])

            elif 'v1' in exp_id:
            # ##############################################
            # # v1
                output = logits[:, :-1, :] # [2, num_token, 256]
                # need to change the label
                labels = labels[:, 1:] # [2, num_token]
                # labels 来做这个判断没毛病,判断label结束的标识
                indexs = torch.nonzero(labels.flatten()==71013).squeeze() #llava是2 

                extra_labels = extra_labels[:, 1:] # [2, num_token]
                c_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction = 'none')(torch.flatten(output, start_dim=0, end_dim=1), extra_labels.flatten())
                outputs['loss'] = torch.sum(c_loss) / torch.sum(extra_labels != -100)
                outputs['logits'] = logits

                terminate = torch.mean(c_loss[indexs-12]) 
                dx_token = torch.mean(c_loss[indexs-11])
                dy_token = torch.mean(c_loss[indexs-10])
                dyaw_token = torch.mean(c_loss[indexs-9])
                body_token = torch.mean(c_loss[indexs-8])
                step_frequency_token = torch.mean(c_loss[indexs-7])
                gait_0_token = torch.mean(c_loss[indexs-6])
                gait_1_token = torch.mean(c_loss[indexs-5])
                gait_2_token = torch.mean(c_loss[indexs-4])
                footswing_height_token = torch.mean(c_loss[indexs-3])
                pitch_token = torch.mean(c_loss[indexs-2])
                stance_width_token = torch.mean(c_loss[indexs-1])

            elif 'v5' in exp_id or 'v6' in exp_id:
                # output (Batchsize, 12)
                # c_labels (Batchsize, 12)
                output = logits
                c_loss = torch.mean((output - c_labels) ** 2,dim=0)
                print(c_loss)
                # print(c_loss)
                outputs['loss'] = torch.mean(c_loss)
                outputs['logits'] = logits

                indexs = torch.tensor([i for i in range(0, 12 * num_batch, 12)])

                terminate = torch.mean(c_loss[indexs]) 
                dx_token = torch.mean(c_loss[indexs+1])
                dy_token = torch.mean(c_loss[indexs+2])
                dyaw_token = torch.mean(c_loss[indexs+3])
                body_token = torch.mean(c_loss[indexs+4])
                step_frequency_token = torch.mean(c_loss[indexs+5])
                gait_0_token = torch.mean(c_loss[indexs+6])
                gait_1_token = torch.mean(c_loss[indexs+7])
                gait_2_token = torch.mean(c_loss[indexs+8])
                footswing_height_token = torch.mean(c_loss[indexs+9])
                pitch_token = torch.mean(c_loss[indexs+10])
                stance_width_token = torch.mean(c_loss[indexs+11])
                
            ##############################################
            else:
                output = logits[:, :, :] # [2, 12, 256]
                labels = labels[:, 1:] # [2, num_token]
                # labels 来做这个判断没毛病,判断label结束的标识
                indexs = torch.nonzero(labels.flatten()==71013).squeeze() #llava是2 
                # print(indexs)

                # need to change the label
                extra_labels = extra_labels# [2, 12]
                c_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction = 'none')(torch.flatten(output, start_dim=0, end_dim=1), extra_labels.flatten())
                outputs['loss'] = torch.sum(c_loss) / torch.sum(extra_labels != -100)
                outputs['logits'] = logits 

                indexs = torch.tensor([i for i in range(0, 12 * num_batch, 12)])

                terminate = torch.mean(c_loss[indexs]) 
                dx_token = torch.mean(c_loss[indexs+1])
                dy_token = torch.mean(c_loss[indexs+2])
                dyaw_token = torch.mean(c_loss[indexs+3])
                body_token = torch.mean(c_loss[indexs+4])
                step_frequency_token = torch.mean(c_loss[indexs+5])
                gait_0_token = torch.mean(c_loss[indexs+6])
                gait_1_token = torch.mean(c_loss[indexs+7])
                gait_2_token = torch.mean(c_loss[indexs+8])
                footswing_height_token = torch.mean(c_loss[indexs+9])
                pitch_token = torch.mean(c_loss[indexs+10])
                stance_width_token = torch.mean(c_loss[indexs+11])

            # if local_rank:
            #     # print(dx_token.item())
            #     wandb.log({"c_x_loss": dx_token.item()})
            #     wandb.log({"c_y_loss": dy_token.item()})
            #     wandb.log({"c_yaw_loss": dyaw_token.item()})
            #     wandb.log({"body_token": body_token.item()})
            #     wandb.log({"step_frequency_token": step_frequency_token.item()})
            #     wandb.log({"gait_0_token": gait_0_token.item()})
            #     wandb.log({"gait_1_token": gait_1_token.item()})
            #     wandb.log({"gait_2_token": gait_2_token.item()})
            #     wandb.log({"footswing_height_token": footswing_height_token.item()})
            #     wandb.log({"pitch_token": pitch_token.item()})
            #     wandb.log({"stance_width_token": stance_width_token.item()})
            #     wandb.log({"c_terminate_loss": terminate.item()})

        # TODO： 这边的输出可能需要考虑一下哈，后面可能会有问题
        # 可以查查看BaseModelOutputWithPast的键值可不可以加入一些自设的一些变量
        if not return_dict:
            return tuple(v for v in outputs if v is not None)
        # provide a loss for trainer
        # this file don't provide
        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        image_patches=None,
        image_patches_indices=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if image_patches_indices is not None:
            model_inputs["image_patches_indices"] = image_patches_indices

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "image_patches_indices": image_patches_indices if past_key_values is None else None,
                "image_patches": image_patches if past_key_values is None else None,
                "exp_id": kwargs.get("exp_id"),
            }
        )

        return model_inputs



class Quart2FuyuForCausalLM_Less(FuyuPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`FuyuDecoderLayer`]

    Args:
        config: FuyuConfig
    """

    def __init__(self, config: FuyuConfig, exp_id):
        super().__init__(config, exp_id)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor
        # "model_type": "persimmon"
        # self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.language_model = PersimmonForCausalLM_Less._from_config(config.text_config)

        self.vision_embed_tokens = nn.Linear(
            config.patch_size * config.patch_size * config.num_channels, config.hidden_size
        )
        if 'v1' in exp_id:
            # v1 的policy head
            self.policy_head = nn.Linear(
                config.hidden_size + 37, 256
            )
        elif 'v2' in exp_id:
            # v2 的policy head
            self.policy_head = nn.Linear(
                config.hidden_size + 37, 256 * 12
            )
        elif 'v3' in exp_id:
            self.projection_v =  nn.Linear(config.hidden_size, config.hidden_size)
            self.projection_p = nn.Linear(37, config.hidden_size)
            self.fusion_vp = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8)
            self.policy_head = nn.Linear(
                config.hidden_size, 256 * 12
            )
        elif 'v4' in exp_id:
            self.projection_v =  nn.Linear(config.hidden_size, config.hidden_size)
            # self.projection_p = nn.Linear(37, config.hidden_size)
            self.fusion_v = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8)
            self.policy_head = nn.Linear(
                config.hidden_size, 256 * 12
            )
        elif 'v5' in exp_id:
            # self.projection_v =  nn.Linear(config.hidden_size, config.hidden_size)
            # self.projection_p = nn.Linear(37, config.hidden_size)
            self.query_tokens = nn.Parameter(torch.zeros(1, 1, config.hidden_size))  # 假设ViT模型的隐藏大小为4096
            self.fusion_v = TransformerEncoderLayer(d_model=config.hidden_size, nhead=8)
            self.policy_head = nn.Linear(
                config.hidden_size, 12
            )
        elif 'v6' in exp_id:
            self.projection_v =  nn.Linear(config.hidden_size, config.hidden_size)
            # self.projection_p = nn.Linear(37, config.hidden_size)
            self.fusion_v = TransformerEncoderLayer(d_model=config.hidden_size, nhead=8)
            self.policy_head = nn.Linear(
                config.hidden_size, 12
            )
        elif 'v7' in exp_id:
            # self.cls_token [12, 4096]
            self.cls_token = nn.Embedding(12, config.hidden_size)
            self.transformer = TwoWayTransformer(depth=2, embedding_dim=4096, mlp_dim=2048,num_heads=8)
            self.policy_head = nn.Linear(
                config.hidden_size, 256
            )  
            self.pe_layer = PositionEmbeddingRandom(d_model=4096, max_len=4096)


        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def gather_continuous_embeddings(
        self,
        word_embeddings: torch.Tensor,
        continuous_embeddings: List[torch.Tensor],
        image_patch_input_indices: torch.Tensor,
    ) -> torch.Tensor:
        """This function places the continuous_embeddings into the word_embeddings at the locations
        indicated by image_patch_input_indices. Different batch elements can have different numbers of continuous
        embeddings.

        Args:
            word_embeddings: Tensor of word embeddings. Shape: [b, s, h]
            continuous_embeddings:
                Tensor of continuous embeddings. The length of the list is the batch size. Each entry is
            shape [num_image_embeddings, hidden], and num_image_embeddings needs to match the number of non-negative
            indices in image_patch_input_indices for that batch element.
            image_patch_input_indices: Tensor of indices of the image patches in the input_ids tensor. Shape: [b, s]
        """
        if not (word_embeddings.shape[0] == len(continuous_embeddings)):
            raise ValueError(
                f"Batch sizes must match! Got {len(continuous_embeddings)=} and {word_embeddings.shape[0]=}"
            )

        output_embeddings = word_embeddings.clone()
        for batch_idx in range(word_embeddings.shape[0]):
            dst_indices = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0]
            src_indices = image_patch_input_indices[batch_idx][dst_indices]
            if src_indices.shape[0] > continuous_embeddings[batch_idx].shape[0]:
                raise ValueError(
                    f"Number of continuous embeddings {continuous_embeddings[batch_idx].shape=} does not match "
                    f"number of continuous token ids {src_indices.shape=} in batch element {batch_idx}."
                )
            output_embeddings[batch_idx, dst_indices] = continuous_embeddings[batch_idx][src_indices].to(output_embeddings.dtype)

        return output_embeddings

    @add_start_docstrings_to_model_forward(FUYU_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        image_patches: torch.Tensor = None,  # [batch_size, num_total_patches, patch_size_ x patch_size x num_channels ]
        image_patches_indices: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        local_rank: Optional[bool] = None,
        proprioceptions: Optional[torch.Tensor] = None,
        extra_labels: Optional[torch.Tensor] = None,
        exp_id: Optional[str] = None,
        c_labels: Optional[torch.Tensor] = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        # monitor_gpu_memory_usage()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        # print('*********training_phrase***************')
        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids) # torch.Size([32, 394, 4096])
            # print(inputs_embeds.size())
            if image_patches is not None and past_key_values is None:
                # 视觉embedding
                patch_embeddings = self.vision_embed_tokens(image_patches.to(self.vision_embed_tokens.weight.dtype)) # torch.Size([32, 352, 4096])
                # 替换视觉token为image embedding
                inputs_embeds = self.gather_continuous_embeddings(
                    word_embeddings=inputs_embeds,
                    continuous_embeddings=patch_embeddings,
                    image_patch_input_indices=image_patches_indices,
                ) # torch.Size([32, 394, 4096])
        
        # outputs = self.language_model(
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     output_attentions=output_attentions,
        #     use_cache=use_cache,
        # )
        # output = outputs['logits']
        # print('*********inputs_embeds***************')
        # print(inputs_embeds.size()) # torch.Size([32, 394, 4096])
        # with torch.inference_mode(mode=True):
        outputs = self.language_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # hidden_states (Batchsize, num_token, 4096)
        hidden_states = outputs[0]
        num_token = hidden_states.size(1)
        num_batch = hidden_states.size(0)
        num_dimen = hidden_states.size(2)

        ###############################
        # # raw_version
        # # (Batchsize, num_token, 262144)
        if 'v0' in exp_id:
            logits = self.language_model.lm_head(hidden_states)
        ###############################
        # v1
        elif 'v1' in exp_id:
            # proprioceptions（Batchsize, 37）--> (Batchsize, num_token , 37）
            proprioceptions = torch.unsqueeze(proprioceptions,dim=1)
            proprioceptions = proprioceptions.repeat((1, num_token, 1))
            # muluyan_fea (Batchsize, num_token, 4096+37)
            mul_fea = torch.cat((hidden_states, proprioceptions),dim=2)
            # logits (Batchsize, num_token, 256)
            logits = self.policy_head(mul_fea) 
            # print(logits.size())  
        # ################################ 

        # ###############################
        # v2
        elif 'v2' in exp_id:
            # hidden_states (Batchsize, num_token, 4096）--> (Batchsize, 4096)
            hidden_states = torch.mean(hidden_states,dim=1)
            # mul_fea (Batchsize, 4096+37)
            mul_fea = torch.cat((hidden_states, proprioceptions),dim=1)
            # logits (Batchsize, 256*12) --> (Batchsize, 12, 256)
            logits = torch.reshape(self.policy_head(mul_fea),(num_batch, 12, 256))
            # print(logits.size())  

        elif 'v3' in exp_id:
            # proprioceptions (Batchsize, 37) --> (Batchsize, 4096)
            proprioceptions = torch.unsqueeze(self.projection_p(proprioceptions),dim=1)
            # hidden_states (Batchsize, num_token, 4096）--> (Batchsize, num_token, 4096)
            hidden_states = self.projection_v(hidden_states)
            # mul_fea (Batchsize, num_token + 1, 4096)
            mul_fea = torch.cat((hidden_states, proprioceptions),dim=1)
            # mul_fea (Batchsize, num_token + 1, 4096)
            mul_fea = self.fusion_vp(mul_fea)
            # p_fea (Batchsize, 4096)
            p_fea = mul_fea[:,-1,:]
            # v_fea (Batchsize, 4096)
            v_fea = torch.mean(mul_fea[:,:-1,:],dim=1)
            # logits (Batchsize, 256*12) --> (Batchsize, 12, 256)
            logits = torch.reshape(self.policy_head(p_fea + v_fea),(num_batch, 12, 256))

        elif 'v4' in exp_id:
            # proprioceptions (Batchsize, 37) --> (Batchsize, 4096)
            # proprioceptions = torch.unsqueeze(self.projection_p(proprioceptions),dim=1)
            # hidden_states (Batchsize, num_token, 4096）--> (Batchsize, num_token, 4096)
            hidden_states = self.projection_v(hidden_states)
            # mul_fea (Batchsize, num_token + 1, 4096)
            # mul_fea = torch.cat((hidden_states, proprioceptions),dim=1)
            # mul_fea (Batchsize, num_token, 4096)
            mul_fea = self.fusion_v(hidden_states)
            # p_fea (Batchsize, 4096)
            # p_fea = mul_fea[:,-1,:]
            # v_fea (Batchsize, 4096)
            v_fea = torch.mean(mul_fea[:,:,:],dim=1)
            # logits (Batchsize, 256*12) --> (Batchsize, 12, 256)
            logits = torch.reshape(self.policy_head(v_fea),(num_batch, 12, 256))

        elif 'v5' in exp_id:
            # query_tokens (1, 1, 4096) --> (Batchsize, 1, 4096)
            query_tokens = self.query_tokens.expand(num_batch, -1, -1)
            # hidden_states (Batchsize, num_token, 4096）--> (Batchsize, num_token, 4096)
            # hidden_states = self.projection_v(hidden_states)
            # mul_fea (Batchsize, num_token + 1, 4096)
            mul_fea = torch.cat((query_tokens, hidden_states),dim=1)
            # mul_fea (Batchsize, num_token + 1, 4096)
            mul_fea = self.fusion_v(mul_fea)
            # q_fea (Batchsize, 4096)
            q_fea = mul_fea[:,0,:]
            # logits (Batchsize, 12)
            logits = self.policy_head(q_fea)

        elif 'v6' in exp_id:
            hidden_states = self.projection_v(hidden_states)
            # mul_fea (Batchsize, num_token + 1, 4096)
            # mul_fea = torch.cat((hidden_states, proprioceptions),dim=1)
            # mul_fea (Batchsize, num_token, 4096)
            mul_fea = self.fusion_v(hidden_states)
            # p_fea (Batchsize, 4096)
            # p_fea = mul_fea[:,-1,:]
            # v_fea (Batchsize, 4096)
            v_fea = torch.mean(mul_fea[:,:,:],dim=1)
            # logits (Batchsize, 12)
            logits = self.policy_head(v_fea)
        elif 'v7' in exp_id:
            # self.cls_token (12, 4096) --> (Batchsize, 12, 4096)
            out_tokens = self.cls_token.weight.unsqueeze(0).expand(num_batch, -1, -1)
            # hidden_states (Batchsize, num_token, 4096）
            # hidden_states_rotary (Batchsize, num_token, 4096）
            img_pe = self.pe_layer(num_token).unsqueeze(0)
            hidden_states_rotary = img_pe.expand(num_batch, -1, -1)
            # mul_fea (Batchsize, num_token, 4096)
            # hs (Batchsize, 12, 4096)
            hs, mul_fea = self.transformer(hidden_states, hidden_states_rotary, out_tokens)
            # logits (Batchsize, 12, 256)
            logits = self.policy_head(hs)
            outputs['logits'] = logits
        
        if labels is not None:
            ##############################################
            # # raw_version
            if 'v0' in exp_id:
                output = logits[:, :-1, :] # [2, num_token, 262144]
                labels = labels[:, 1:] # [2, num_token]
                # labels 来做这个判断没毛病,判断label结束的标识
                indexs = torch.nonzero(labels.flatten()==71013).squeeze() #llava是2 

                c_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction = 'none')(torch.flatten(output, start_dim=0, end_dim=1), labels.flatten())
                outputs['loss'] = torch.sum(c_loss) / torch.sum(labels != -100)
                outputs['logits'] = logits 

                terminate = torch.mean(c_loss[indexs-12]) 
                dx_token = torch.mean(c_loss[indexs-11])
                dy_token = torch.mean(c_loss[indexs-10])
                dyaw_token = torch.mean(c_loss[indexs-9])
                body_token = torch.mean(c_loss[indexs-8])
                step_frequency_token = torch.mean(c_loss[indexs-7])
                gait_0_token = torch.mean(c_loss[indexs-6])
                gait_1_token = torch.mean(c_loss[indexs-5])
                gait_2_token = torch.mean(c_loss[indexs-4])
                footswing_height_token = torch.mean(c_loss[indexs-3])
                pitch_token = torch.mean(c_loss[indexs-2])
                stance_width_token = torch.mean(c_loss[indexs-1])

            elif 'v1' in exp_id:
            # ##############################################
            # # v1
                output = logits[:, :-1, :] # [2, num_token, 256]
                # need to change the label
                labels = labels[:, 1:] # [2, num_token]
                # labels 来做这个判断没毛病,判断label结束的标识
                indexs = torch.nonzero(labels.flatten()==71013).squeeze() #llava是2 

                extra_labels = extra_labels[:, 1:] # [2, num_token]
                c_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction = 'none')(torch.flatten(output, start_dim=0, end_dim=1), extra_labels.flatten())
                outputs['loss'] = torch.sum(c_loss) / torch.sum(extra_labels != -100)
                outputs['logits'] = logits

                terminate = torch.mean(c_loss[indexs-12]) 
                dx_token = torch.mean(c_loss[indexs-11])
                dy_token = torch.mean(c_loss[indexs-10])
                dyaw_token = torch.mean(c_loss[indexs-9])
                body_token = torch.mean(c_loss[indexs-8])
                step_frequency_token = torch.mean(c_loss[indexs-7])
                gait_0_token = torch.mean(c_loss[indexs-6])
                gait_1_token = torch.mean(c_loss[indexs-5])
                gait_2_token = torch.mean(c_loss[indexs-4])
                footswing_height_token = torch.mean(c_loss[indexs-3])
                pitch_token = torch.mean(c_loss[indexs-2])
                stance_width_token = torch.mean(c_loss[indexs-1])

            elif 'v5' in exp_id or 'v6' in exp_id:
                # output (Batchsize, 12)
                # c_labels (Batchsize, 12)
                output = logits
                c_loss = torch.mean((output - c_labels) ** 2,dim=0)
                print(c_loss)
                # print(c_loss)
                outputs['loss'] = torch.mean(c_loss)
                outputs['logits'] = logits

                indexs = torch.tensor([i for i in range(0, 12 * num_batch, 12)])

                terminate = torch.mean(c_loss[indexs]) 
                dx_token = torch.mean(c_loss[indexs+1])
                dy_token = torch.mean(c_loss[indexs+2])
                dyaw_token = torch.mean(c_loss[indexs+3])
                body_token = torch.mean(c_loss[indexs+4])
                step_frequency_token = torch.mean(c_loss[indexs+5])
                gait_0_token = torch.mean(c_loss[indexs+6])
                gait_1_token = torch.mean(c_loss[indexs+7])
                gait_2_token = torch.mean(c_loss[indexs+8])
                footswing_height_token = torch.mean(c_loss[indexs+9])
                pitch_token = torch.mean(c_loss[indexs+10])
                stance_width_token = torch.mean(c_loss[indexs+11])
                
            ##############################################
            else:
                output = logits[:, :, :] # [2, 12, 256]
                labels = labels[:, 1:] # [2, num_token]
                # labels 来做这个判断没毛病,判断label结束的标识
                indexs = torch.nonzero(labels.flatten()==71013).squeeze() #llava是2 
                # print(indexs)

                # need to change the label
                extra_labels = extra_labels# [2, 12]
                c_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction = 'none')(torch.flatten(output, start_dim=0, end_dim=1), extra_labels.flatten())
                outputs['loss'] = torch.sum(c_loss) / torch.sum(extra_labels != -100)
                outputs['logits'] = logits 

                indexs = torch.tensor([i for i in range(0, 12 * num_batch, 12)])

                terminate = torch.mean(c_loss[indexs]) 
                dx_token = torch.mean(c_loss[indexs+1])
                dy_token = torch.mean(c_loss[indexs+2])
                dyaw_token = torch.mean(c_loss[indexs+3])
                body_token = torch.mean(c_loss[indexs+4])
                step_frequency_token = torch.mean(c_loss[indexs+5])
                gait_0_token = torch.mean(c_loss[indexs+6])
                gait_1_token = torch.mean(c_loss[indexs+7])
                gait_2_token = torch.mean(c_loss[indexs+8])
                footswing_height_token = torch.mean(c_loss[indexs+9])
                pitch_token = torch.mean(c_loss[indexs+10])
                stance_width_token = torch.mean(c_loss[indexs+11])

            # if local_rank:
            #     # print(dx_token.item())
            #     wandb.log({"c_x_loss": dx_token.item()})
            #     wandb.log({"c_y_loss": dy_token.item()})
            #     wandb.log({"c_yaw_loss": dyaw_token.item()})
            #     wandb.log({"body_token": body_token.item()})
            #     wandb.log({"step_frequency_token": step_frequency_token.item()})
            #     wandb.log({"gait_0_token": gait_0_token.item()})
            #     wandb.log({"gait_1_token": gait_1_token.item()})
            #     wandb.log({"gait_2_token": gait_2_token.item()})
            #     wandb.log({"footswing_height_token": footswing_height_token.item()})
            #     wandb.log({"pitch_token": pitch_token.item()})
            #     wandb.log({"stance_width_token": stance_width_token.item()})
            #     wandb.log({"c_terminate_loss": terminate.item()})

        # TODO： 这边的输出可能需要考虑一下哈，后面可能会有问题
        # 可以查查看BaseModelOutputWithPast的键值可不可以加入一些自设的一些变量
        if not return_dict:
            return tuple(v for v in outputs if v is not None)
        # provide a loss for trainer
        # this file don't provide
        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        image_patches=None,
        image_patches_indices=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if image_patches_indices is not None:
            model_inputs["image_patches_indices"] = image_patches_indices

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "image_patches_indices": image_patches_indices if past_key_values is None else None,
                "image_patches": image_patches if past_key_values is None else None,
                "exp_id": kwargs.get("exp_id"),
            }
        )

        return model_inputs