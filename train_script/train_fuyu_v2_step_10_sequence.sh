#!/bin/bash
source /opt/conda/bin/activate llava

# add root path
ROOT_PATH=/tongxinyang/projects/QUART_publish_shorted
cd $ROOT_PATH
export PYTHONPATH=$PYTHONPATH:$ROOT_PATH
unset LD_LIBRARY_PATH

RAW_DATA_PATH=/wangdonglin
# fuyu-8b path
PRETRAINED_CKPT_PATH=./Pretrained/huggingface/hub/models--adept--fuyu-8b

current_time=$(date +%Y%m%d)
# current_time=20240830_sequence_5
DATASET_TYPE=Full
TRAINING_DATA_PATH=./datasets/$DATASET_TYPE/sim_json_path/sim_ahead_10_seq.json 
LEARNING_RATE=2e-5
SAVE_STEPS=10000
TRAINING_MODE=pretrain
# TRAINING_MODE=None
GPU_NUM=4
BATCHSIZE_PERDEVICE=32
GRADACC_PERDEVICE=1
EPOCHS=10
TUNE_MM_MLP_ADAPTER=True
EXP_ID=Fuyu_v0
OUTPUT_CKPT_PATH=./ckpts/$EXP_ID/$current_time/

VOCAB_NAME=vocab_fuyu.json

deepspeed ./train_ahead_n.py \
    --deepspeed ../Quart/scripts/zero3.json \
    --model_name_or_path  $PRETRAINED_CKPT_PATH\
    --trainingdata_path  $TRAINING_DATA_PATH\
    --evaluatingdata_path ./datasets/test_real_v1_vel_1.json \
    --tune_mm_mlp_adapter $TUNE_MM_MLP_ADAPTER \
    --bf16 True \
    --training_mode $TRAINING_MODE \
    --output_dir $OUTPUT_CKPT_PATH \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCHSIZE_PERDEVICE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRADACC_PERDEVICE \
    --evaluation_strategy "no" \
    --group_by_modality_length False \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 12 \
    --learning_rate $LEARNING_RATE \
    --prediction_loss_only False\
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --exp_id $EXP_ID \
    --vocab_name ./vocabs/$VOCAB_NAME \

