#!/bin/bash

source /opt/conda/bin/activate llava
PROJECT_PATH='your/quart/path' 

CKPT_PATH="${PROJECT_PATH}/ckpts"
VQ_CKPT_PATH="${PROJECT_PATH}/ckpts/vq_state_dict/Sequence_vq_10_each_conv.pt"
TEST_TYPE="seen"
SAVE_FOLDER="${PROJECT_PATH}/output"
MODEL_NAME="Quart_online"
DATASET_TYPE="Full"
HEADLESS=True
ENV_NUM=10
DETYPE=float16

TASK_CONFIGS='task_configs'
VOCAB_PATH='./vocabs/vocab_fuyu.json'

python ./gym_eval_scripts/gym_task_loop.py \
    --test_type $TEST_TYPE \
    --save_folder $SAVE_FOLDER \
    --model_name $MODEL_NAME \
    --headless $HEADLESS \
    --env_num $ENV_NUM \
    --task_lists $TASK_CONFIGS \
    --vocab_path $VOCAB_PATH \
    --vq_ckpt_path $VQ_CKPT_PATH \
    --ckpt_path $CKPT_PATH \
    --dataset_type $DATASET_TYPE \
    --detype $DETYPE \
