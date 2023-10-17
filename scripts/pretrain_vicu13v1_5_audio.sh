#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

# MODEL_VERSION=vicuna-v1-3-7b
# MODEL_VERSION=llama-2-7b-chat

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=plain
########### DO NOT CHANGE ###########

python /hy-tmp/LLaVA-plus/llava/train/train_mem.py \
    --model_name_or_path /hy-tmp/vicuna/ \
    --version v1 \
    --data_path /hy-tmp/audiocap/out.json \
    --audio_folder /hy-tmp/audiocap/audio_all/ \
    --audio_tower /hy-tmp/data2vec-audio-base-960h \
    --tune_mm_audio_mlp_adapter True \
    --mm_audio_select_layer -1 \
    --mm_audio_use_im_start_end False \
    --mm_audio_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/robotgpt_pretrain-audio-13b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
