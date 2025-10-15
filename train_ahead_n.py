import sys
R3M_path = '/dingpengxiang/Wenxuan/walk-these-ways/Quart/models/r3m'
sys.path.append(R3M_path)
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
import warnings
from typing import Dict, Optional, Sequence, List
from sklearn.metrics import precision_recall_fscore_support
from IPython import embed

import torch
import wandb
import numpy as np
import transformers
from transformers import LlamaTokenizer
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from models.quart_fuyu import Quart2FuyuForCausalLM

from torch.utils.data import Dataset
from utils import FuyuTrainer,get_inputs_train,get_inputs_vq_train,get_inputs_vq_n_train

from transformers import FuyuProcessor, FuyuImageProcessor, ViTImageProcessor
from transformers import BitsAndBytesConfig, Trainer, EvalPrediction


local_rank = None
IGNORE_INDEX = -100
exp_id = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="./Pretrained/huggingface/hub/models--adept--fuyu-8b")
    tune_mm_mlp_adapter: bool = field(default=False)
    training_mode: Optional[str] = field(default=None)
    exp_id: Optional[str] = field(default=None)
    vocab_name: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_patch_merge_type: Optional[str] = field(default='flat')


@dataclass
class DataArguments:
    trainingdata_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    evaluatingdata_path: str = field(default=None,
                        metadata={"help": "Path to the testing data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    group_by_modality_length: bool = field(default=False)

#deepspeed
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

#
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

#llava
def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

#llava
def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len

#llava
def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

#llava
def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

#llava
def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def data_collator(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

#preprocess here belongs to LLAVA, LLaVA-Llama-3
def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def quat_rotate_inverse(q,v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

class ActionSupervisedDataset(Dataset): 
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 model_args: ModelArguments):
        super(ActionSupervisedDataset, self).__init__()
        
        rank0_print("Formatting inputs...Skip in lazy mode")
        list_data_dict = json.load(open(data_path, "r"))   #data_path数据集
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.model_args = model_args
        self.tokenizer = tokenizer

        
        self.gravity_vec = torch.unsqueeze(torch.FloatTensor([0,0,-1]),0)
        self.parent_data_path = os.path.dirname(os.path.dirname(data_path))  
        
        #data_path such as'./datasets/Full/merged_json_path/sim_distinguish.json'
        #parent_data_path such as './datasets/Full/'

        # './datasets/Full/sim_quadruped_data_info'
        self.sim_info_path = os.path.join(self.parent_data_path, 'sim_quadruped_data_info')
        # './datasets/Full/sim_quadruped_data_info/proprioceptions.npy'
        self.sim_proprioceptions_path = os.path.join(self.sim_info_path, "proprioceptions.npy")
        self.sim_proprioceptions_dict = np.load(self.sim_proprioceptions_path, allow_pickle=True).item()  #load a dict，which is proprioceptions.npy

        self.real_info_path = os.path.join(self.parent_data_path, 'real_quadruped_data_info')

        self.sim_commands_path = os.path.join(self.sim_info_path, "commands.npy")
        self.sim_commands_dict = np.load(self.sim_commands_path, allow_pickle=True).item()
        # self.vq_model=RVQ(layers_hidden=[2048, 2048, 2048, 512], input_dim=input_dim, K=512, num_quantizers=2, output_act=False)
        # self.vq_model.load_state_dict(torch.load(self.model_args.vq_ckpt))
        # self.real_commands_path = os.path.join(self.real_info_path, "commands.npy")
        # self.real_commands_dict = np.load(self.real_commands_path, allow_pickle=True).item()


        if 'Fuyu' in self.model_args.exp_id:
            self.image_processor = FuyuImageProcessor()  #调的transformers里的FuyuImageProcessor
            self.processor = FuyuProcessor(image_processor=self.image_processor, tokenizer=self.tokenizer)  #建了一个processor，类别为<class 'transformers.models.fuyu.processing_fuyu.FuyuProcessor'>
            vocab_path = os.path.join(self.model_args.vocab_name)  #读了一下字典'./vocabs/vocab_fuyu.json'
            with open(vocab_path, 'r') as f:
                self.vocab_dict = json.load(f)   #self.vocab_dict 把字典读入
        else:
            warnings.warn("Unknown model type.")

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        if 'image' in sources.keys():

            # import time
            # t0 = time.time()

            image = np.asarray(Image.open(sources['image']).resize((240,240)))
            question = sources['conversations'][0]['value']
            # 问题里面有texture这个字符，引发歧义
            question = question.replace(' texture', '')
            # answer = sources['conversations'][1]['value']
            answer = sources['vq']  # ahead_vq_n这里是改变了数据集，调用dict['vq']
            type = sources['conversations'][0]['type']

            if type == 'sim':
                episode_path = os.path.dirname(os.path.dirname(sources['image']))
                episode_idx = int(os.path.basename(sources['image']).split('.')[0])

                # load proprioceptions
                proprioceptions_dict = self.sim_proprioceptions_dict[episode_path]
                episode_len = len(proprioceptions_dict['joint_pos'])
                joint_pos = proprioceptions_dict['joint_pos'][episode_idx] * 1.0
                joint_vel = proprioceptions_dict['joint_vel'][episode_idx] * 0.05
                body_linear_vel = proprioceptions_dict['body_linear_vel'][episode_idx] * 2.0
                body_angular_vel = proprioceptions_dict['body_angular_vel'][episode_idx] * 0.25
                contact_states = proprioceptions_dict['contact_states'][episode_idx]
                # body_pos = proprioceptions_dict['body_pos'][episode_idx]
                body_quat = proprioceptions_dict['body_quat'][episode_idx]
                # convert via a martix
                body_quat = torch.unsqueeze(torch.from_numpy(body_quat),0)
                body_quat = torch.squeeze(quat_rotate_inverse(body_quat, self.gravity_vec),dim=0).numpy()

                proprioceptions = np.concatenate((joint_pos,joint_vel,body_linear_vel,body_angular_vel,contact_states,body_quat))
                proprioceptions = torch.from_numpy(proprioceptions)

            else:
                episode_path = os.path.dirname(os.path.dirname(sources['image']))
                episode_idx = int(os.path.basename(sources['image']).split('.')[0])
                # # real have no proprioceptions
                proprioceptions = torch.ones((37,))
                terminate_str = answer.split(" ")[1]

            # load continous action
            if type == 'sim':
                commands_dict = self.sim_commands_dict[episode_path]
                terminate = int(i == episode_len - 1)
            else:
                commands_dict = self.real_commands_dict[episode_path]
                terminate = int(terminate_str)
            
            dx_token = commands_dict['dx'][episode_idx]
            dy_token = commands_dict['dy'][episode_idx]
            dyaw_token = commands_dict['dyaw'][episode_idx]
            body_token = commands_dict['body_height'][episode_idx]
            step_frequency_token = commands_dict['step_frequency'][episode_idx]
            gait_0_token = commands_dict['gait_0'][episode_idx]
            gait_1_token = commands_dict['gait_1'][episode_idx]
            gait_2_token = commands_dict['gait_2'][episode_idx]
            footswing_height_token = commands_dict['footswing_height'][episode_idx]
            pitch_token = commands_dict['pitch'][episode_idx]
            stance_width_token = commands_dict['stance_width'][episode_idx]
            c_labels = [terminate,dx_token,dy_token,dyaw_token,body_token,step_frequency_token,gait_0_token,gait_1_token,gait_2_token,footswing_height_token,pitch_token,stance_width_token]
            c_labels = torch.tensor(c_labels)

            if 'Fuyu' in self.model_args.exp_id:

                # prompt = question + answer
                # print(prompt)
                # 这个prompt生成的有点问题，因为最后一个token的意思是
                # 71122表示换行符，response将要开始
                # print("vq_train:",self.tokenizer, image, question, , answer)
                
                model_inputs = get_inputs_vq_n_train(self.tokenizer, image, question, self.vocab_dict, answer)    
                # model_inputs = self.processor.process_fortraining(text=prompt, images=[image], return_tensors="pt")
                # if type == 'real':
                #     model_inputs["input_ids"][0,-9:-1] = self.tokenizer.unk_token_id

                input_ids = model_inputs["input_ids"][0]
                indexs = torch.nonzero(input_ids==71122).squeeze()

                labels = copy.deepcopy(input_ids)
                labels[:indexs+1] = -100
                if type == 'real':
                    labels[-9:-1] = -100
                    # 0228 为了让模型不预测开始或者结束的token
                    labels[-13] = -100
                
                if 'v1' in exp_id:
                    extra_labels = copy.deepcopy(labels)
                    extra_labels[-1] = -100
                    extra_labels[-13:-1] -= 70003
                else:
                    extra_labels = copy.deepcopy(labels)
                    extra_labels[-1] = -100
                    extra_labels -= 70003
                    extra_labels = extra_labels[-13:-1]

                data_dict = dict(input_ids=input_ids,
                                image_patches=model_inputs["image_patches"][0],
                                image_patches_indices=model_inputs["image_patches_indices"][0],
                                labels=labels,
                                extra_labels=extra_labels,
                                proprioceptions=proprioceptions,
                                c_labels=c_labels)
                # print(time.time()-t0)
                # import pdb; pdb.set_trace()
        
        return data_dict

@dataclass
class ActionDataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        if 'Fuyu' in exp_id:
            input_ids, image_patches, image_patches_indices, labels, extra_labels, proprioceptions, c_labels = tuple([instance[key] for instance in instances]
                    for key in ("input_ids", "image_patches", "image_patches_indices", "labels", "extra_labels", "proprioceptions", "c_labels"))

            image_patches = torch.stack(image_patches,0)
            proprioceptions = torch.stack(proprioceptions,0)
            c_labels = torch.stack(c_labels,0)
        
            image_patches_indices = torch.nn.utils.rnn.pad_sequence(
                image_patches_indices,
                batch_first=True,
                padding_value=-1)
            
            # use padding_value to pad
            # [32,392]
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.unk_token_id)


            labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                batch_first=True,
                                                padding_value=IGNORE_INDEX)

            extra_labels = torch.nn.utils.rnn.pad_sequence(extra_labels,
                            batch_first=True,
                            padding_value=IGNORE_INDEX)

            if local_rank == 0 :
                batch = dict(
                    input_ids=input_ids,
                    labels=labels,
                    image_patches=image_patches,
                    image_patches_indices=image_patches_indices,
                    attention_mask=input_ids.ne(self.tokenizer.unk_token_id),
                    local_rank=True,
                    proprioceptions=proprioceptions,
                    extra_labels=extra_labels,
                    exp_id=exp_id,
                    c_labels=c_labels
            )
                
            else:
                batch = dict(
                    input_ids=input_ids,
                    labels=labels,
                    image_patches=image_patches,
                    image_patches_indices=image_patches_indices,
                    attention_mask=input_ids.ne(self.tokenizer.unk_token_id),
                    local_rank=False,
                    proprioceptions=proprioceptions,
                    extra_labels=extra_labels,
                    exp_id=exp_id,
                    c_labels=c_labels
            )

        elif 'LLaVA' in exp_id:
            input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX)
            input_ids = input_ids[:, :self.tokenizer.model_max_length]
            labels = labels[:, :self.tokenizer.model_max_length]

            if local_rank == 0 :
                batch = dict(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=input_ids.ne(self.tokenizer.unk_token_id),
                    local_rank=True
            )
                
            else:
                batch = dict(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=input_ids.ne(self.tokenizer.unk_token_id),
                    local_rank=False
            )

            if 'image' in instances[0]:
                images = [instance['image'] for instance in instances]
                if all(x is not None and x.shape == images[0].shape for x in images):
                    batch['images'] = torch.stack(images)
                else:
                    batch['images'] = images

        return batch

def make_action_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,model_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = ActionSupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.trainingdata_path,
                                data_args=data_args,
                                model_args=model_args)
    

    data_collator = ActionDataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def train():

    global local_rank
    global exp_id
    
    # 加载huggingface的参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # parser.parse_args().tf32=False
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    # embed()

    if 'pretrain' in model_args.training_mode:
        print("pretrain mode！")
        training_args.lora_enable = False
        training_args.bits = 16
    else:
        pass

    # init wandb
    # bug: 如果给定有用的地址的话，我新建的wandb.log就加载不进去
    log_dir = './log/pre-training'
    training_args.output_dir = training_args.output_dir + \
                                 "lr{}" .format(training_args.learning_rate) +  \
                                '_' + "batsiz{}".format(training_args.per_device_train_batch_size) + \
                                '_' + "graacc{}".format(training_args.gradient_accumulation_steps)  +\
                                '_' + "epoch{}".format(training_args.num_train_epochs) + \
                                '_' + "bits{}".format(training_args.bits) + \
                                '_' + "datasets{}".format(data_args.trainingdata_path.split('/')[2]) + \
                                '_' + "subset{}".format(data_args.trainingdata_path.split('/')[4][:-5]) 


    wandb_name = model_args.exp_id +  \
     "lr{}" .format(training_args.learning_rate) +  \
    '_' + "batsiz{}".format(training_args.per_device_train_batch_size) +  \
    '_' + "graacc{}".format(training_args.gradient_accumulation_steps)  +\
    '_' + "epoch{}".format(training_args.num_train_epochs) + \
    '_' + "bits{}".format(training_args.bits)+ \
    '_' + "datasets{}".format(data_args.trainingdata_path.split('/')[2]) + \
    '_' + "subset{}".format(data_args.trainingdata_path.split('/')[4][:-5])


    exp_id = model_args.exp_id

    if 'Fuyu' in exp_id:
        model = Quart2FuyuForCausalLM.from_pretrained(
            #在最顶ModelArguments类别有设model_name_or_path='/dingpengxiang/Pretrained/huggingface/hub/models--adept--fuyu-8b'
            model_args.model_name_or_path,   
            local_files_only=True,
            exp_id=model_args.exp_id
        )
        if 'v0' in exp_id:
            pass
        else:
            # 冻结住LLM的参数
            for p in model.language_model.parameters():
                    p.requires_grad = False
    else:
        pass
    
    # fuyu模型的设置
    data_args.model_max_length = training_args.model_max_length
    # embed()

    # 是否修改视觉encoder
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    print('If train the visual adapter:{}'.format(training_args.tune_mm_mlp_adapter))
    if not training_args.tune_mm_mlp_adapter:
        if 'Fuyu' in exp_id:
            if 'v0' in exp_id:
                if not training_args.tune_mm_mlp_adapter:
                    for p in model.vision_embed_tokens.parameters():
                        p.requires_grad = False
            else:
                for p in model.vision_embed_tokens.parameters():
                        p.requires_grad = False
        elif 'LLaVA' in exp_id:
            if not model_args.tune_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = False
    else:
        pass
    
    

    rank0_print("Load datasets: begin!")

    data_module = make_action_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              model_args=model_args)

    rank0_print("Current allocated memory:", torch.cuda.memory_allocated()/1024/1024/1024)

    trainer = FuyuTrainer(model=model,
            tokenizer=tokenizer,
            args=training_args,
            **data_module)

    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer,
                                    output_dir=training_args.output_dir)

    

if __name__ == "__main__":
    train()
