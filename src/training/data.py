import copy
import os
from dataclasses import dataclass, field
from typing import Dict

import torch
import transformers
from transformers.models.mllama.processing_mllama import get_cross_attention_token_mask, convert_sparse_cross_attention_mask_to_dense
import ujson as json
from PIL import Image
from torch.utils.data import Dataset
from decord import VideoReader, cpu

from .params import DataArguments
from .constants import *

def encode_video(video_path, max_num_frames=10):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    return frames

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        padding=True,
    ):
        super(LazySupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.max_num_frames = data_args.max_num_frames

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        sources = self.list_data_dict[i]

        is_video = False
        num_frames = None

        processor = self.processor
        if "image" in sources:
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
        
            for image_file in image_files:
                if not os.path.exists(image_file):
                    image_file = os.path.join(image_folder, image_file)
                images.append(Image.open(image_file))

        elif "video" in sources:
            video_file = sources["video"]
            video_folder = self.data_args.image_folder

            if not os.path.exists(video_file):
                video_file = os.path.join(video_folder, video_file)

            images = encode_video(video_file, self.max_num_frames)
            
            is_video = True
            num_frames = len(images)

        else:
            images = None

        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video, num_frames=num_frames))

        all_input_ids = [] 
        all_labels = []
        pixel_values = None
        aspect_ratio_ids = None
        aspect_ratio_mask = None
        cross_attention_mask = None

        if images is not None:
            input_text = processor.apply_chat_template(sources, add_generation_prompt=False)
            inputs = processor(images=images, text=input_text, return_tensors="pt")
            pixel_values = inputs['pixel_values']
            aspect_ratio_ids = inputs['aspect_ratio_ids']
            aspect_ratio_mask = inputs['aspect_ratio_mask']
            cross_attention_mask = inputs['cross_attention_mask']

        del inputs

        for idx, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]
            gpt_prompt = f"{gpt_response['content'][0]['text']}{EOT_TOKEN}\n"
            if idx == 0:
                user_prompt = processor.apply_chat_template([user_input], add_generation_prompt=True)
                prompt_input_ids = processor.tokenizer(user_prompt, add_special_tokens=False, return_tensors='pt')['input_ids']

            else:
                user_prompt = f"{START_HEADER_TOKEN}{user_input['role']}{END_HEADER_TOKEN}\n\n{user_input['content'][0]['text']}{EOT_TOKEN}\n{START_HEADER_TOKEN}{gpt_response['role']}{END_HEADER_TOKEN}\n\n"
                prompt_input_ids = processor.tokenizer(user_prompt, add_special_tokens=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_prompt, add_special_tokens=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            pixel_values=pixel_values,
            aspect_ratio_mask=aspect_ratio_mask,
            aspect_ratio_ids=aspect_ratio_ids,
            cross_attention_mask=cross_attention_mask,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return data_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int, processor: transformers.ProcessorMixin):
        self.pad_token_id = pad_token_id
        self.processor = processor

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_aspect_ratio_ids = []
        batch_aspect_ratio_mask = []
        batch_cross_attention_mask = []

        for example in examples:
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            batch_pixel_values.append(example.get("pixel_values"))
            batch_aspect_ratio_ids.append(example.get("aspect_ratio_ids"))
            batch_aspect_ratio_mask.append(example.get("aspect_ratio_mask"))
            batch_cross_attention_mask.append(
                example.get("cross_attention_mask", [None])[0] if example.get("cross_attention_mask") else None
            )
                
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        cross_attention_mask = pad_sequence(
            [cam for cam in batch_cross_attention_mask if cam is not None], padding_side='right', padding_value=0
        )
        
        labels = pad_sequence(
            batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX
        )
        
        attention_mask = input_ids != self.pad_token_id
        pixel_values = torch.cat([pv for pv in batch_pixel_values if pv is not None], dim=0) if any(batch_pixel_values) else None
        aspect_ratio_ids = torch.cat([ar for ar in batch_aspect_ratio_ids if ar is not None], dim=0) if any(batch_aspect_ratio_ids) else None
        aspect_ratio_mask = torch.cat([am for am in batch_aspect_ratio_mask if am is not None], dim=0) if any(batch_aspect_ratio_mask) else None


        batch_dict = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if pixel_values is not None:
            batch_dict['pixel_values'] = pixel_values
            batch_dict['aspect_ratio_ids'] = aspect_ratio_ids
            batch_dict['aspect_ratio_mask'] = aspect_ratio_mask
            batch_dict['cross_attention_mask'] = cross_attention_mask

        return batch_dict

def replace_image_tokens(input_string, start_count=0):

    count = start_count
    has_image = False

    if LLAVA_IMAGE_TOKEN not in input_string:
        return input_string, count, has_image

    while LLAVA_IMAGE_TOKEN in input_string:
        has_image = True
        input_string = input_string.replace(LLAVA_IMAGE_TOKEN+'\n', '', 1)
        count += 1

    return input_string, count, has_image

def video_to_image_tokens(input_string, num_frames):

    frame_tokens = "\n".join([LLAVA_IMAGE_TOKEN] * num_frames)
    input_string = input_string.replace(LLAVA_VIDEO_TOKEN, frame_tokens)

    return input_string

def llava_to_openai(conversations, is_video=False, num_frames=None):

    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    image_count = 0
    for conversation in conversations:
        if is_video:
            conversation['value'] = video_to_image_tokens(conversation["value"], num_frames)
        
        transformed_content, image_count, has_image = replace_image_tokens(conversation["value"], image_count)
        content = []
        if has_image:
            for _ in range(image_count):
                content.append({"type":"image"})
        content.append({"type":"text", "text":transformed_content})
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data

def make_supervised_data_module(processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = LazySupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id, processor=processor)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)