import os
import torch
import transformers
from peft import LoraConfig, get_peft_model
import ast
from transformers import AutoProcessor, BitsAndBytesConfig, MllamaForConditionalGeneration
from train.dpo_trainer import LLamaVDPOTrainer
from train.data import make_dpo_data_module
from train.params import DataArguments, ModelArguments, DPOArguments
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib
from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_mllama

local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.vision_model
    vision_tower.to(dtype=compute_dtype, device=device)

    img_projection_params = model.multi_modal_projector.parameters()
    set_requires_grad(img_projection_params, not training_args.freeze_img_projector)

    vision_model_params = vision_tower.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)

    if training_args.bits in [4, 8]:
        model.multi_modal_projector.to(dtype=compute_dtype, device=device)

def configure_llm(model, training_args):
    llm_params = model.language_model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, DPOArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if training_args.use_liger:
        apply_liger_kernel_to_mllama()
    
    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    if training_args.lora_namespan_exclude is not None:
        training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
    else:
        training_args.lora_namespan_exclude = ["multi_modal_projector"]

    if not training_args.vision_lora:
        training_args.lora_namespan_exclude += ["vision_model", "multi_modal_projector"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["vision_model", "multi_modal_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))
    
    model = MllamaForConditionalGeneration.from_pretrained(
        model_args.model_id,
        torch_dtype=compute_dtype,
        cache_dir=training_args.cache_dir,
        attn_implementation="sdpa",
        **bnb_model_from_pretrained_args
    )
    
    ref_model = None

    if not training_args.lora_enable:
        ref_model = MllamaForConditionalGeneration.from_pretrained(
            model_args.model_id,
            torch_dtype=compute_dtype,
            cache_dir=training_args.cache_dir,
            attn_implementation="sdpa"
            **bnb_model_from_pretrained_args
        )
    
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    model.config.hidden_size = model.config.text_config.hidden_size
    model.config.text_config.use_cache = False
    model.config.use_cache = False

    if ref_model is not None:
        ref_model.eval()
        ref_model.config.use_cache = False

    if training_args.bits in [4,8]:
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        # This is a workaround for a bug in the current implementation of gradient checkpointing
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": True})
    
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        # This is a workaround for a bug in the current implementation of gradient checkpointing
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
        )

        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

        # Peft maodel makes vision tower and projector freezed again.
        # Configuring fuction could be called here, but sometimes it does not work properly.
        # So I just made it this way.
        # Need to be fixed in the future.

        if not training_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "vision_model" in name:
                    param.requires_grad = True

        if not training_args.freeze_img_projector:
            for name, param in model.named_parameters():
                if "multi_modal_projector" in name:
                    param.requires_grad = True

    processor = AutoProcessor.from_pretrained(model_args.model_id)
        
    model.config.vision_lr = training_args.vision_lr
    model.config.projector_lr = training_args.projector_lr

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_dpo_data_module(processor=processor,
                                              data_args=data_args)

    trainer = LLamaVDPOTrainer(
        model=model,
        ref_model=ref_model,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"],
        processing_class=processor,
        args=training_args,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()