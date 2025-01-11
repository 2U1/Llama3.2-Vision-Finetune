# #!/bin/bash

MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
# MODEL_NAME="meta-llama/Llama-3.2-90B-Vision-Instruct"

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path /home/workspace/Llama3-vision-ft/output/test_lora \
    --model-base $MODEL_NAME  \
    --save-model-path /home/workspace/Llama3-vision-ft/output/merge_test \
    --safe-serialization