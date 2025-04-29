# Fine-tuning Llama3.2-Vision

This repository contains a script for training [Llama3.2-Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) with only using HuggingFace and [Liger-Kernel](https://github.com/linkedin/Liger-Kernel).

## Other projects

**[[Phi3-Vision Finetuning]](https://github.com/2U1/Phi3-Vision-Finetune)**<br>
**[[Qwen2-VL Finetuning]](https://github.com/2U1/Qwen2-VL-Finetune)**<br>
**[[Molmo Finetuning]](https://github.com/2U1/Molmo-Finetune)**<br>
**[[Pixtral Finetune]](https://github.com/2U1/Pixtral-Finetune)**<br>
**[[SmolVLM Finetune]](https://github.com/2U1/SmolVLM-Finetune)**<br>
**[[Gemma3 Finetune]](https://github.com/2U1/Gemma3-Finetune)**

## Update

- [2025/04/29] 🔥Add DPO Training.
- [2025/01/24] Add option for using DoRA.
- [2025/01/24] Fix error in LoRA training.
- [2025/01/18] 🔥Supports mixed-modality data.
- [2025/01/11] Updated 8-bit training using ms_amp fp8 with opt_level O3.
- [2024/11/05] Add memory efficient 8-bit training.
- [2024/11/05] 🔥Supports training with liger-kernel.
- [2024/10/04] 🔥Supports text-only data.

## Table of Contents

- [Fine-tuning Llama3.2-Vision](#fine-tuning-llama32-vision)
  - [Other projects](#other-projects)
  - [Update](#update)
  - [Table of Contents](#table-of-contents)
  - [Supported Features](#supported-features)
  - [Docker](#docker)
  - [Installation](#installation)
    - [Environments](#environments)
    - [Using `environment.yaml`](#using-environmentyaml)
    - [Using `requirements.txt`](#using-requirementstxt)
  - [Dataset Preparation](#dataset-preparation)
  - [Supervised FineTuning](#supervised-finetuning)
    - [Full Finetuning](#full-finetuning)
    - [Finetune with LoRA](#finetune-with-lora)
    - [Train with video dataset](#train-with-video-dataset)
      - [Merge LoRA Weights](#merge-lora-weights)
  - [DPO Finetuning](#dpo-finetuning)
  - [Issue for libcudnn error](#issue-for-libcudnn-error)
  - [TODO](#todo)
  - [Known Issues](#known-issues)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)

## Supported Features

- Deepspeed
- LoRA, QLoRA
- Full-finetuning
- Multi-image and video training

## Docker

To simplfy the setting process for training, you could use the provided pre-build environments.<br>
The settings are done in the conda env named `train`.<br><br>
You could find more information about the image [here](https://hub.docker.com/repository/docker/john119/vlm/general).

```
docker pull john119/vlm
docker run --gpus all -it -v /host/path:/docker/path --name vlm --ipc=host john119/vlm /bin/bash
```

## Installation

### Environments

- Ubuntu 22.04
- Nvidia-Driver 550.120
- Cuda version 12.4

Install the required packages using `environment.yml`.

### Using `environment.yaml`

```bash
conda env create -f environment.yaml
conda activate train
```

**Note:** Llama3.2-Vision does not support flash-attention2.

### Using `requirements.txt`

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu126
```

## Dataset Preparation

The script requires a dataset formatted according to the LLaVA specification. The dataset should be a JSON file where each entry contains information about conversations and images. Ensure that the image paths in the dataset match the provided `--image_folder`.<br>

**When using a multi-image dataset, the image tokens should all be `<image>`, and the image file names should have been in a list.**
**Please see the example below and follow format your data.**

<details>
<summary>Example for single image dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": "000000033471.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      },
      {
        "from": "human",
        "value": "What feature can be seen on the back of the bus?"
      },
      {
        "from": "gpt",
        "value": "The back of the bus features an advertisement."
      },
      {
        "from": "human",
        "value": "Is the bus driving down the street or pulled off to the side?"
      },
      {
        "from": "gpt",
        "value": "The bus is driving down the street, which is crowded with people and other vehicles."
      }
    ]
  }
  ...
]
```

</details>

<details>
<summary>Example for multi image dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": ["000000033471.jpg", "000000033472.jpg"],
    "conversations": [
      {
        "from": "human",
        "value": "<image>\n<image>\nIs the perspective of the camera differnt?"
      },
      {
        "from": "gpt",
        "value": "Yes, It the perspective of the camera is different."
      }
    ]
  }
  ...
]
```

</details>

<details>
<summary>Example for video dataset</summary>

```json
[
  {
    "id": "sample1",
    "video": "sample1.mp4",
    "conversations": [
      {
        "from": "human",
        "value": "<video>\nWhat is going on in this video?"
      },
      {
        "from": "gpt",
        "value": "A man is walking down the road."
      }
    ]
  }
  ...
]
```

**Note:** Llama3.2-Vision uses a video as a sequential of images.

</details>

<details>
<summary>Example for DPO dataet</summary>

```json
[
  {
    "id": "e4cd017f-3854-4019-9cc1-9dacacb5d424",
    "image": "example_0_0.png",
    "prompt": "<image>\nDescribe the key features of the image in great detail.",
    "chosen": "The image features a modern, multi-colored train parked on a train track, occupying a significant portion of the scene. The train has a unique and striking design, with red, white, and blue colors, as well as yellow accents. It appears to be a high-speed train, indicating its advanced engineering and potential for fast travel.\n\nThe train is situated next to a platform, suggesting that it may be at a train station. The overall scene conveys a sense of transportation and modern technology, with the vibrant and sleek train as the main subject.",
    "rejected": "The image features a modern, colorful train parked on a set of train tracks. The train has a striking design, with red, white, and blue colors as well as yellow accents. It appears to be a high-speed train, ready for departure.\n\nThe train spans a significant portion of the image, from the left to the right side. The train tracks can be seen clearly beneath the train, emphasizing its position on the railway line. The overall scene gives off an impression of a contemporary and efficient mode of transportation."
  },
  {
    "id": "5e19e647-e5d3-4bcf-82e9-d262570743ae",
    "image": "example_1_0.png",
    "prompt": "<image>\nIs this bus in the USA?",
    "chosen": "Yes, based on the image, it can be assumed that this bus is in the USA. The location of the bus cannot be accurately determined.",
    "rejected": "No, it's not in the USA. The image does not provide specific information on where the bus is located. However, we can say that it's not in the United States."
  }
  ...
]
```

</details>
<br><br>

Adding the new domain-specific data on top of the general data from open-source data will enhance downstream capabilities while retaining the foundational skills. Of course, you can also choose to fine-tune solely on the new data based on your requirements.

## Supervised FineTuning

**Note:** Deepspeed zero2 is faster than zero3, however it consumes more memory. Also, most of the time zero2 is more stable than zero3.<br><br>
**Tip:** You could use `adamw_bnb_8bit` for optimizer to save memory.

To run the training script, use the following command:

### Full Finetuning

```bash
bash scripts/finetune.sh
```

### Finetune with LoRA

If you want to train only the language model with LoRA and perform full training for the vision model:

```bash
bash scripts/finetune_lora.sh
```

If you want to train both the language model and the vision model with LoRA:

```bash
bash scripts/finetune_lora_vision.sh
```

**IMPORTANT:** If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together.

<details>
<summary>Training arguments</summary>

- `--deepspeed` (str): Path to DeepSpeed config file (default: "scripts/zero2.json").
- `--data_path` (str): Path to the LLaVA formatted training data (a JSON file). **(Required)**
- `--image_folder` (str): Path to the images folder as referenced in the LLaVA formatted training data. **(Required)**
- `--model_id` (str): Path to the Llama3.2-Vision model. **(Required)**
- `--optim` (str): Optimizer when training (default: `adamw_torch`).
- `--output_dir` (str): Output directory for model checkpoints
- `--num_train_epochs` (int): Number of training epochs (default: 1).
- `--per_device_train_batch_size` (int): Training batch size per GPU per forwarding step.
- `--gradient_accumulation_steps` (int): Gradient accumulation steps (default: 4).
- `--freeze_vision_tower` (bool): Option to freeze vision_model (default: False).
- `--tune_merger` (bool): Option to tune projector (default: True).
- `--num_lora_modules` (int): Number of target modules to add LoRA (-1 means all layers).
- `--vision_lr` (float): Learning rate for vision_model.
- `--projector_lr` (float): Learning rate for projector.
- `--learning_rate` (float): Learning rate for language module.
- `--bf16` (bool): Option for using bfloat16.
- `--fp16` (bool): Option for using fp16.
- `--lora_enable` (bool): Option for enabling LoRA (default: False)
- `--vision_lora` (bool): Option for including vision_tower to the LoRA module. The `lora_enable` should be `True` to use this option. (default: False)
- `--use_dora` (bool): Option for using DoRA instead of LoRA. The `lora_enable` should be `True` to use this option. (default: False)
- `--lora_namespan_exclude` (str): Exclude modules with namespans to add LoRA.
- `--max_seq_length` (int): Maximum sequence length (default: 128K).
- `--bits` (int): Quantization bits (default: 16).
- `--disable_flash_attn2` (bool): Disable Flash Attention 2.
- `--report_to` (str): Reporting tool (choices: 'tensorboard', 'wandb', 'none') (default: 'tensorboard').
- `--logging_dir` (str): Logging directory (default: "./tf-logs").
- `--lora_rank` (int): LoRA rank (default: 128).
- `--lora_alpha` (int): LoRA alpha (default: 256).
- `--lora_dropout` (float): LoRA dropout (default: 0.05).
- `--logging_steps` (int): Logging steps (default: 1).
- `--dataloader_num_workers` (int): Number of data loader workers (default: 4).

**Note:** The learning rate of `vision_model` should be 10x ~ 5x smaller than the `language_model`.

</details>

### Train with video dataset

You can train the model using a video dataset. However, Llama3.2-Vision processes videos as a sequence of images, so you’ll need to select specific frames and treat them as multiple images for training. You can set LoRA configs and use for LoRA too.

```bash
bash scripts/finetune_video.sh
```

If you run out of vram, you can use [zero3_offload](./scripts/zero3_offload.json) instead of [zero3](./scripts/zero3_offload.json). However, using zero3 is preferred.

#### Merge LoRA Weights

```
bash scripts/merge_lora.sh
```

**Note:** Remember to replace the paths in `finetune.sh` or `finetune_lora.sh` with your specific paths. (Also in `merge_lora.sh` when using LoRA.)

## DPO Finetuning

You can train the model using Direct Preference Optimization (DPO).<br>
The process is quite similar to Supervised Fine-Tuning (SFT), and you can also apply LoRA during DPO training just like in SFT.

```bash
bash scripts/finetune_dpo.sh
```

## Issue for libcudnn error

```
Could not load library libcudnn_cnn_train.so.8. Error: /usr/local/cuda-12.1/lib/libcudnn_cnn_train.so.8: undefined symbol: _ZN5cudnn3cnn34layerNormFwd_execute_internal_implERKNS_7backend11VariantPackEP11CUstream_stRNS0_18LayerNormFwdParamsERKNS1_20NormForwardOperationEmb, version libcudnn_cnn_infer.so.8
```

You could run `unset LD_LIBRARY_PATH` for this error.
You could see this [issue](https://github.com/andimarafioti/florence2-finetuning/issues/2)

## TODO

- [x] Support for multi-image & video data
- [x] Support for batch_size > 1
- [x] Handle mixed-modality data
- [x] Add DPO Support

## Known Issues

- [libcudnn issue](#issue-for-libcudnn-error)

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

## Citation

If you find this repository useful in your project, please consider giving a :star: and citing:

```bibtex
@misc{Llama3.2-Vision-Finetuning,
  author = {Yuwon Lee},
  title = {Llama3.2-Vision-Finetune},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/2U1/Llama3.2-Vision-Ft}
}
```

## Acknowledgement

This project is based on

- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT): An amazing open-source project of LMM.
- [Llama3.2-Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct): Awesome pretrained MLLM by Meta.
