import os
import torch
import numpy as np

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    has_length,
)
from torchvision import transforms
from typing import List, Optional, Iterable,  Tuple, Union

# from models.RVQ.residual_vq import RVQ
# from models.RVQ.dataset import action_tokenize_dataset

# input_dim=12
# quantizers=2
# model = RVQ(layers_hidden=[2048, 2048, 2048, 512], input_dim=input_dim, K=512, num_quantizers=2, output_act=False)
# if quantizers==2:
#     model.load_state_dict(torch.load('/dingpengxiang/Pengxiang/Quart++/state_dict/VQ/asa_vq_NoTanh_12_2.pt'))
# elif quantizers==3:
#     model.load_state_dict(torch.load('/dingpengxiang/Pengxiang/Quart++/state_dict/VQ/asa_vq_NoTanh_12_3.pt'))


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks

def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    assert len(mm_indices) > 0, "Should have at least one multimodal sample."
    assert len(lang_indices) > 0, "Should have at least one language sample."

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) >= megabatch_size:
        megabatches = [additional_batch[:megabatch_size]] + megabatches
        additional_batch = additional_batch[megabatch_size:]

    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    return [i for megabatch in megabatches for i in megabatch]

def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]

class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)

class FuyuTrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            # world_size = int(os.environ['WORLD_SIZE'])
            # print(world_size)
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def _save_checkpoint(self, model, trial, metrics=None):
        super(FuyuTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        
        super(FuyuTrainer, self)._save(output_dir, state_dict)

def normalize(image,mean,std):
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array")

    image = (image - mean) / std

    return image

def patchify_image(image, patch_dim_h, patch_dim_w): #(3,240,240),image,30,30
    channels, height, width = image.shape
    unfolded_along_height = image.unfold(1, patch_dim_h, patch_dim_h) #(3,8,240,30) 
    patches = unfolded_along_height.unfold(2, patch_dim_w, patch_dim_w) #(3,8,8,30,30)
    # (channels, height/patch_dim_h, width/patch_dim_w, patch_dim_h,  patch_dim_w)
    patches_reshaped = patches.contiguous().view(channels, -1, patch_dim_h, patch_dim_w) #重排形状，-1为自动计算(3,64,30,30)

    patches_final = patches_reshaped.permute(1, 2, 3, 0).reshape(
     -1, channels * patch_dim_h * patch_dim_w
    )
    # ( -- , channels * patch_dim_h * patch_dim_w)
    return patches_final

global inputs_ids_prefix,inputs_patches_indices_prefix

inputs_ids_prefix = [ 71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71019,
        71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71019,
        71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71019,
        71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71019,
        71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71019,
        71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71019,
        71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71019,
        71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71019]
inputs_patches_indices_prefix = [ 0,  1,  2,  3,  4,  5,  6,  7, -1,  8,  9, 10, 11, 12, 13, 14, 15, -1,
        16, 17, 18, 19, 20, 21, 22, 23, -1, 24, 25, 26, 27, 28, 29, 30, 31, -1,
        32, 33, 34, 35, 36, 37, 38, 39, -1, 40, 41, 42, 43, 44, 45, 46, 47, -1,
        48, 49, 50, 51, 52, 53, 54, 55, -1, 56, 57, 58, 59, 60, 61, 62, 63]

def get_inputs_test(tokenizer, image, text_prompt, vocab_dict):
    # tokenizer:<class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>
    # image: <class 'numpy.ndarray'>
    # text_prompt: <class 'str'>,
    # vocab: <class 'dict'>
    image_numpy = normalize(np.asarray(image) / 255.0, 0.5, 0.5).astype(np.float32)  #归一化，均值0,5，方差0.5，范围[0,255]→[-1~1]
    text_list = ["<s>"] + tokenizer.tokenize(text_prompt) + ["<0x04>"] #['<s>', '▁What', '▁action', '▁should', '▁the▁leg', 'ged', '▁robot', '▁take', '▁to▁go▁to▁the', '▁yellow', '▁ball', '▁slowly', '▁with▁a', '▁', 'tr', 'otting', '▁', 'ga', 'it', '?', '<0x0A>', '<0x04>']
    prompts = [vocab_dict[i] for i in text_list]
    img_tensor = torch.from_numpy(image_numpy).permute(2, 0, 1)   #image_numpy (240,240,3) → img_tensor torch.Size([3, 240, 240])
    image_patches = patchify_image(img_tensor, patch_dim_h=30, patch_dim_w=30) #torch.Size([64, 2700])
    # image_patches = patchify(image_numpy, (30,30,3), 30 ).reshape(64,2700)
    # image_patches = image_patches[np.newaxis, :]
    inputs_patches_indices = np.array(inputs_patches_indices_prefix, dtype=np.int64)
    inputs_patches_indices = inputs_patches_indices[np.newaxis, :]
    inputs_ids = np.array(inputs_ids_prefix + prompts, dtype=np.int64)
    inputs_ids = inputs_ids[np.newaxis, :]
    # print(f"\n end time: {time.time()-t0}s")
    return {
                "input_ids": torch.from_numpy(inputs_ids),  #pic token+prompts
                "image_patches": image_patches.unsqueeze(0),  #[1,64,2700]
                "image_patches_indices": torch.from_numpy(inputs_patches_indices), #separate patch id, see fuyu structure fro more detail
            }

def get_inputs_train(tokenizer, image, text_prompt, vocab_dict, answer):
    image_numpy = normalize(np.asarray(image) / 255.0, 0.5, 0.5).astype(np.float32)
    text_list = ["<s>"] + tokenizer.tokenize(text_prompt) + ["<0x04>"]
    prompts = [vocab_dict[i] for i in text_list]

    answer_list =  answer.split('<0x04>')[1].split(' ')[1:]
    answer_token = [vocab_dict[i] for i in answer_list]

    eof_token = [vocab_dict["|ENDOFTEXT|"]]

    img_tensor = torch.from_numpy(image_numpy).permute(2, 0, 1)
    image_patches = patchify_image(img_tensor, patch_dim_h=30, patch_dim_w=30)
    # image_patches = patchify(image_numpy, (30,30,3), 30 ).reshape(64,2700)
    # image_patches = image_patches[np.newaxis, :]
    inputs_patches_indices = np.array(inputs_patches_indices_prefix, dtype=np.int64)
    inputs_patches_indices = inputs_patches_indices[np.newaxis, :]
    inputs_ids = np.array(inputs_ids_prefix + prompts + answer_token + eof_token, dtype=np.int64)
    inputs_ids = inputs_ids[np.newaxis, :]
    # print(f"\n end time: {time.time()-t0}s")
    return {
                "input_ids": torch.from_numpy(inputs_ids),
                "image_patches": image_patches.unsqueeze(0),
                "image_patches_indices": torch.from_numpy(inputs_patches_indices),
            }

def get_inputs_vq_train(tokenizer, image, text_prompt, vocab_dict, answer,VQ_model):
    image_numpy = normalize(np.asarray(image) / 255.0, 0.5, 0.5).astype(np.float32)
    text_list = ["<s>"] + tokenizer.tokenize(text_prompt) + ["<0x04>"]
    prompts = [vocab_dict[i] for i in text_list]

    answer_list =  answer.split('<0x04>')[1].split(' ')[1:]  #str
    
    action_ori_list = [float(item) for item in answer_list] # convert to float
    input = np.array([action_ori_list])
    input_vector=torch.from_numpy(input).to(torch.float32)
    tokenized_list = VQ_model.tokenize(input_vector)

    tokenized_answer = [str(i) for i in tokenized_list[0].numpy()]
    tokenized_full_answer= [str(int(action_ori_list[0]))] + tokenized_answer #add terminate sign

    answer_token = [vocab_dict[i] for i in tokenized_full_answer]

    eof_token = [vocab_dict["|ENDOFTEXT|"]]

    img_tensor = torch.from_numpy(image_numpy).permute(2, 0, 1)
    image_patches = patchify_image(img_tensor, patch_dim_h=30, patch_dim_w=30)
    # image_patches = patchify(image_numpy, (30,30,3), 30 ).reshape(64,2700)
    # image_patches = image_patches[np.newaxis, :]
    inputs_patches_indices = np.array(inputs_patches_indices_prefix, dtype=np.int64)
    inputs_patches_indices = inputs_patches_indices[np.newaxis, :]
    # print("input_ids:",inputs_ids_prefix,prompts,"\nanswer - tokens",answer_token,"eof:",eof_token)
    inputs_ids = np.array(inputs_ids_prefix + prompts + answer_token + eof_token, dtype=np.int64)
    inputs_ids = inputs_ids[np.newaxis, :]
    # import pdb; pdb.set_trace()
    # print(f"\n end time: {time.time()-t0}s")
    return {
                "input_ids": torch.from_numpy(inputs_ids),
                "image_patches": image_patches.unsqueeze(0),
                "image_patches_indices": torch.from_numpy(inputs_patches_indices),
            }


def get_inputs_vq_n_train(tokenizer, image, text_prompt, vocab_dict, answer):
    image_numpy = normalize(np.asarray(image) / 255.0, 0.5, 0.5).astype(np.float32)
    text_list = ["<s>"] + tokenizer.tokenize(text_prompt) + ["<0x04>"]
    prompts = [vocab_dict[i] for i in text_list]

    answer_list =  answer.split('<0x04>')[1].split(' ')[1:]
    answer_token = [vocab_dict[i] for i in answer_list]

    eof_token = [vocab_dict["|ENDOFTEXT|"]]

    img_tensor = torch.from_numpy(image_numpy).permute(2, 0, 1)
    image_patches = patchify_image(img_tensor, patch_dim_h=30, patch_dim_w=30)
    # image_patches = patchify(image_numpy, (30,30,3), 30 ).reshape(64,2700)
    # image_patches = image_patches[np.newaxis, :]
    inputs_patches_indices = np.array(inputs_patches_indices_prefix, dtype=np.int64)
    inputs_patches_indices = inputs_patches_indices[np.newaxis, :]
    inputs_ids = np.array(inputs_ids_prefix + prompts + answer_token + eof_token, dtype=np.int64)
    inputs_ids = inputs_ids[np.newaxis, :]
    # print(f"\n end time: {time.time()-t0}s")
    return {
                "input_ids": torch.from_numpy(inputs_ids),
                "image_patches": image_patches.unsqueeze(0),
                "image_patches_indices": torch.from_numpy(inputs_patches_indices),
            }