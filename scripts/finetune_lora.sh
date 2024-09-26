#!/bin/bash

MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
# MODEL_NAME="meta-llama/Llama-3.2-90B-Vision-Instruct"

# LLaMA3.2-Vision Does not support flash-attnetion2 yet.

export PYTHONPATH=src:$PYTHONPATH

deepspeed src/training/train.py \
    --lora_enable True \
    --vision_lora False \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --num_lora_modules 1 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path /home/workspace/description/traffic_sample.json \
    --image_folder /home/workspace/dataset \
    --disable_flash_attn2 True \
    --tune_img_projector True \
    --freeze_vision_tower True \
    --freeze_llm False \
    --bf16 True \
    --fp16 False \
    --output_dir output/test_qlora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bits 4 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --save_steps 500 \
    --save_total_limit 10 \