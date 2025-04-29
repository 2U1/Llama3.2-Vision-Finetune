import os
import torch
from torch import nn
from typing import Union
import torch.nn.functional as F

from transformers.trainer import (
    is_peft_available,
    WEIGHTS_NAME,
    TRAINING_ARGS_NAME,
    SAFE_WEIGHTS_NAME,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy
)
import safetensors
from peft import PeftModel
from typing import Optional
from transformers.modeling_utils import PreTrainedModel
from peft import PeftModel
from trl import DPOTrainer
from trl.trainer.utils import pad_to_length, flush_left, selective_log_softmax
from train.train_utils import get_peft_state_non_lora_maybe_zero_3

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

class LLamaVDPOTrainer(DPOTrainer):

    def __init__(self, *args, **kwargs):
        super(LLamaVDPOTrainer, self).__init__(*args, **kwargs)

    def _prepare_dataset(
        self,
        dataset,
        processing_class,
        args,
        dataset_name
    ):
        return dataset

    @staticmethod
    def concatenated_inputs(
        batch: dict[str, Union[list, torch.LongTensor]], padding_value: int
    ) -> dict[str, torch.LongTensor]:

        concatenated_batch = {}
        concatenated_batch['prompt_input_ids'] = torch.cat([batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0)
        concatenated_batch['prompt_attention_mask'] = torch.cat([batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0)
        concatenated_batch['pixel_values'] = torch.cat([batch["pixel_values"], batch["pixel_values"]], dim=0)
        concatenated_batch["cross_attention_mask"] = torch.cat([batch["cross_attention_mask"], batch["cross_attention_mask"]], dim=0)
        concatenated_batch["aspect_ratio_ids"] = torch.cat(batch["aspect_ratio_ids"], batch["aspect_ratio_ids"], dim=0)
        concatenated_batch["aspect_ratio_mask"] = torch.cat(batch["aspect_ratio_mask"], batch["aspect_ratio_mask"], dim=0)

        max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        concatenated_batch['completion_input_ids'] = torch.cat(
            (
                pad_to_length(batch["chosen_input_ids"], max_completion_length, pad_value=padding_value),
                pad_to_length(batch["rejected_input_ids"], max_completion_length, pad_value=padding_value),
            ),
        )

        concatenated_batch['completion_attention_mask'] = torch.cat(
            (
                pad_to_length(batch["chosen_attention_mask"], max_completion_length, pad_value=0),
                pad_to_length(batch["rejected_attention_mask"], max_completion_length, pad_value=0),
            ),
        )

        return concatenated_batch
    

    def concatenated_forward(self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]]):

        num_examples = batch['prompt_input_ids'].shape[0]
        
        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {}

        if self.aux_loss_enabled:
            model_kwargs['output_router_logits'] = True

        # Add image/video values to model kwargs
        model_kwargs['pixel_values'] = concatenated_batch['pixel_values']
        model_kwargs["cross_attention_mask"] = concatenated_batch['cross_attention_mask']
        model_kwargs["aspect_ratio_ids"] = concatenated_batch['aspect_ratio_ids']
        model_kwargs["aspect_ratio_mask"] = concatenated_batch['aspect_ratio_mask']

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        
        input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
        loss_mask = torch.cat(
            (torch.zeros_like(prompt_attention_mask), completion_attention_mask), dim=1
        )

        # Flush left to reduce the memory usage
        # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
        #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
        attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

        model_kwargs["attention_mask"] = attention_mask

        outputs = model(input_ids, **model_kwargs)
        logits = outputs.logits

        labels = torch.roll(input_ids, shifts=-1, dims=1)
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

        if logits.shape[:2] != labels.shape[:2]:
            # for llava, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        all_logps = per_token_logps.sum(-1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)  # same as sum(probs**2) in log space
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

        if self.args.rpo_alpha is not None:
            # Only use the chosen logits for the RPO loss
            chosen_logits = logits[:num_examples]
            chosen_labels = labels[:num_examples]

            # Compute the log probabilities of the labels
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
            )

        if self.loss_type == "ipo":
            all_logps = all_logps / loss_mask.sum(-1)

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]
        output["mean_chosen_logits"] = logits[:num_examples][loss_mask[:num_examples]].mean()
        output["mean_rejected_logits"] = logits[num_examples:][loss_mask[num_examples:]].mean()

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output


    def _save_checkpoint(self, model, trial):
            # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
            # want to save except FullyShardedDDP.
            # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

            # Save model checkpoint
            if self.args.lora_enable:
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

                if self.hp_search_backend is None and trial is None:
                    self.store_flos()

                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)
                self.save_model(output_dir, _internal_call=True)
                non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=False)
                torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

                if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
                    best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
                    best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

                    if os.path.exists(best_checkpoint_dir):
                        self.state.best_model_checkpoint = best_checkpoint_dir

                if not self.args.save_only_model:
                    # Save optimizer and scheduler
                    self._save_optimizer_and_scheduler(output_dir)
                    self._save_scaler(output_dir)
                    # Save RNG state
                    self._save_rng_state(output_dir)

                # Save the Trainer state
                if self.args.should_save:
                    # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
                    for cb in [
                        cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
                    ]:
                        cb_name = cb.__class__.__name__
                        cb_state = cb.state()
                        if isinstance(self.state.stateful_callbacks[cb_name], list):
                            self.state.stateful_callbacks[cb_name].append(cb_state)
                        else:
                            self.state.stateful_callbacks[cb_name] = cb_state
                    self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

                if self.args.push_to_hub:
                    self._push_from_checkpoint(output_dir)
            else:
                super(LLamaVDPOTrainer, self)._save_checkpoint(model, trial)