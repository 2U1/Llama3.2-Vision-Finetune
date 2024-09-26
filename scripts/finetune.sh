#!/bin/bash

MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
# MODEL_NAME="meta-llama/Llama-3.2-90B-Vision-Instruct"

# LLaMA3.2-Vision Does not support flash-attnetion2 yet.
# It only supports batch_size=1 for now.

export PYTHONPATH=src:$PYTHONPATH

deepspeed src/training/train.py \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path /path/to/your/training/data.json \
    --image_folder /path/to/your/image/folder \
    --disable_flash_attn2 True \
    --lora_enable False \
    --tune_img_projector True \
    --freeze_vision_tower False \
    --freeze_llm False \
    --bf16 True \
    --output_dir output/test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4