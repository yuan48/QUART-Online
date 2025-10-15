import sys
from preprocess.init_path import REAL_INSTRUCTION_DICT,SIM_INSTRUCTION_DICT
import time
import numpy as np
import torch.nn as nn
import torch
import json
import os
import argparse
from PIL import Image
from utils import get_inputs_test

from models.RVQ.residual_vq import RVQ
from models.RVQ.vq_Sequence import RVQ_Seq, RVQ_Seq_10

from transformers import AutoProcessor,AutoTokenizer, FuyuProcessor, FuyuImageProcessor, FuyuForCausalLM, BitsAndBytesConfig
from models.quart_fuyu import Quart2FuyuForCausalLM

import accelerate 


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params



class Quart_online():
    """Pretrained fuyu model of Adept via huggingface"""
    def __init__(self, args, vocab_path=None):
        print("You are running the model on GPU: ", torch.cuda.is_available())
        self.device = torch.device("cuda")
        self.dtype = torch.float16 if args.detype=='float16' else torch.float32
        self.ckpt_path=args.ckpt_path
        self.RVQmodel = RVQ_Seq_10(layers_hidden=[512, 512, 512, 512], input_dim=12, K=512, num_quantizers=2, output_act=False)
        self.RVQmodel.load_state_dict(torch.load(args.vq_ckpt_path))

        vocab_path=args.vocab_path

        print("current_pth:",self.ckpt_path)
        self.model = FuyuForCausalLM.from_pretrained(self.ckpt_path, local_files_only=True, torch_dtype=self.dtype, low_cpu_mem_usage=True,device_map="cuda").cuda()
        print('model loaded')
        torch.cuda.empty_cache()
        # initialize tokenizer and fuyu processor, pretrained and via huggingface
        self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt_path)
        self.processor = FuyuProcessor(image_processor=FuyuImageProcessor(), tokenizer=self.tokenizer)
        with open(vocab_path, 'r') as f:
            self.vocab_dict = json.load(f)
        self.inverse_vocab_dict = {v : k for k, v in self.vocab_dict.items()}
        # import pdb; pdb.set_trace()


    
    def inference_one_time_step(self, images, instructions, proprioceptions=None, network_state=None):
        ter=0       
        t0 = time.time()
        inference_tokens=4

        inputs = get_inputs_test(self.tokenizer, images, instructions, self.vocab_dict)  #utils.py {pic token+prompts, [1,64,2700] dim of image_patches}
        inputs = {k: v.to(dtype=self.dtype if torch.is_floating_point(v) else v.dtype, device=self.device) for k,v in inputs.items()}

        with torch.inference_mode():
            generation_output = self.model.generate(**inputs,num_beams=1, do_sample=True, max_new_tokens=inference_tokens, pad_token_id=self.tokenizer.eos_token_id)
        generate_time= time.time()-t0
        # print(f"VLA Generation time: {time.time()-t0}s")
        inference_tokens=4
        generation_text = [self.inverse_vocab_dict[float(i)] for i in generation_output[0, -inference_tokens:].cpu().numpy()]
        discrete_action_tokens = torch.tensor(list(map(int, generation_text)))
        input_vector=torch.unsqueeze(discrete_action_tokens,dim=0)
        actions=self.RVQmodel.detokenize(input_vector.view(2, 2))
        # print(f"VLA+VQ Generation time: {time.time()-t0}s")
        actions[0,:,0]=actions[0,:,0].round()
        actions = actions.cpu().detach().numpy().tolist()[0]
        actions = torch.tensor(actions).to(torch.float32)

        output_actions = {"commands":actions,"terminate_episode": ter}
        return output_actions,generate_time
    

class Quart():
    """Pretrained fuyu model of Adept via huggingface"""
 
    def __init__(self, args,exp_id, ckpt_path, range_path, vocab_path=None):
        print("You are running the model on GPU: ", torch.cuda.is_available())
        self.device = torch.device("cuda")
        self.dtype = torch.float16

        self.model = FuyuForCausalLM.from_pretrained(ckpt_path, local_files_only=True).cuda()

        print('model loaded')

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        self.processor = FuyuProcessor(image_processor=FuyuImageProcessor(), tokenizer=self.tokenizer)
        self.range_path=args.range_path
        with open(vocab_path, 'r') as f:
            self.vocab_dict = json.load(f)
        self.inverse_vocab_dict = {v : k for k, v in self.vocab_dict.items()}

        
        ranges = np.load(os.path.join(range_path, 'ranges.npy'), allow_pickle=True).item()
        ranges = list(ranges.values())
        self.ranges = ranges[:9]
        self.min = ranges[-2]
        self.max = ranges[-1]
        self.ranges = self.max - self.min
    
    def inference_one_time_step(self, images, instructions, proprioceptions=None, network_state=None):
        # pre processing image and text
        # input: 0.5s
        t0 = time.time()

        inputs = get_inputs_test(self.tokenizer, images, instructions, self.vocab_dict)
        inputs = {k: v.to(dtype=self.dtype if torch.is_floating_point(v) else v.dtype, device=self.device) for k,v in inputs.items()}
        with torch.inference_mode():
            generation_output = self.model.generate(**inputs,num_beams=1, do_sample=True, max_new_tokens=12, pad_token_id=self.tokenizer.eos_token_id)
        generation_text = [self.inverse_vocab_dict[float(i)] for i in generation_output[0, -12:].cpu().numpy()]
        action_tokens = list(map(float, generation_text))

        # detokenizer
        actions = []
        for i in range(1, 12):
            a = action_tokens[i]
            # de-normalize
            a = a / 256
            a = a * self.ranges[i-1] + self.min[i-1]
            actions.append(a)

        generate_time= time.time()-t0
        print(f"Generation time: {time.time()-t0}s")

        actions = torch.tensor(actions).to(torch.float32)
        ter = torch.tensor(action_tokens[0]).to(torch.float32)
        

        output_actions = {"commands":actions,"terminate_episode": ter}
        return output_actions,generate_time



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference mode for quart')
    parser.add_argument('--exp_id', help='pretrained baseline',default='Fuyu_v0')
    parser.add_argument('--ckpt_path', help='ckpt save path',default='./ckpts/quart_online')
    parser.add_argument('--dataset_path', help='datset save path',default='./sample_data/sim_quadruped_data_unload')
    parser.add_argument('--dataset_type', help='task included',default='Full')
    parser.add_argument('--vocab_path', help='fuyu vacab save path',default='./vocabs/vocab_fuyu.json')
    parser.add_argument('--vq_ckpt_path', help='vq save path',default='./vq_state_dict/Sequence_vq_10_each_conv.pt')
    # parser.add_argument('--range_dir', help='QUART range path',default='./datasets/Full/sim_quadruped_data_info')
    parser.add_argument('--detype', help='float16 or float32',default='float16')
    args = parser.parse_args()
    
    
    dataset_type = args.dataset_type   

    quart = Quart_online(args)

    scene_name = 'go_avoid_green_cube'
    episode_name = '000000'
    action_id = 0
    sample_rate = 10
    image_size = 240
    instructions = SIM_INSTRUCTION_DICT[args.dataset_type][scene_name]
        
    # get episode path

    # import pdb;pdb.set_trace()
    img_dir = os.path.join(args.dataset_path, scene_name,episode_name,'image') 
    text_prompt = "What action should the legged robot take to {} slowly with a trotting gait?".format(instructions)
    
    # for true_idx in range(0, episode_length, sample_rate):
    img_idx = "{:03d}".format(action_id)
    img_path = os.path.join(img_dir, f"{img_idx}.png")
    image_pil = Image.open(img_path).resize((image_size, image_size))
    image_np = np.asarray(image_pil)
    import pdb; pdb.set_trace()

    output_actions,generate_time = quart.inference_one_time_step(image_np,text_prompt + '\n')  

    print("commands:",output_actions["commands"],"\nter:",output_actions["terminate_episode"])
    #output_actions["commands"]: torch.Size([5, 12]), output_actions["terminate_episode"]: 0

