# The script below is adapted from trl <https://github.com/huggingface/trl>
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
import warnings
from collections import defaultdict
from dataclasses import FrozenInstanceError, replace
from typing import Any, Callable, Optional, Union, List, Dict

import pandas as pd
import torch
import torch.nn as nn
from accelerate import PartialState
from accelerate.utils import gather_object
from datasets import Dataset
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    is_wandb_available,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available
from transformers.utils.deprecation import deprecate_kwarg

from optimas.train.reward_data_utils import maybe_apply_chat_template
from optimas.train.reward_trainer_utils import (
    compute_accuracy,
    decode_and_strip_padding,
    disable_dropout_in_model,
    print_rich_table,
    RunningMoments
)

import torch
import numpy as np

from dataclasses import dataclass, field
from optimas.train.reward_config import RewardConfig
from optimas.utils.logger import setup_logger


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb

logger = setup_logger(__name__)

@dataclass
class RewardDataCollatorWithPadding:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str]`, optional, defaults to `True`):
            Padding strategy to pass to the tokenizer.
        pad_to_multiple_of (`int` or `None`, optional, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, optional, defaults to `"pt"`):
            The tensor type to use.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        features_chosen = []
        features_rejected = []
        margin = []
        component_names = []
        # Check if margin is provided
        has_margin = "margin" in features[0]
        for feature in features:
            # Ensure the expected keys exist
            if (
                "input_ids_chosen" not in feature
                or "input_ids_rejected" not in feature
                or "attention_mask_chosen" not in feature
                or "attention_mask_rejected" not in feature
            ):
                raise ValueError(
                    "The features should include `input_ids_chosen`, `attention_mask_chosen`, "
                    "`input_ids_rejected` and `attention_mask_rejected`"
                )
            features_chosen.append({
                "input_ids": feature["input_ids_chosen"],
                "attention_mask": feature["attention_mask_chosen"],
            })
            features_rejected.append({
                "input_ids": feature["input_ids_rejected"],
                "attention_mask": feature["attention_mask_rejected"],
            })
            if has_margin:
                margin.append(feature["margin"])
            # Collect the component name for each example
            component_names.append(feature["component_name"])

        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "return_loss": True,
            "component_name": component_names,  # include the component names in the batch
        }
        if has_margin:
            margin = torch.tensor(margin, dtype=torch.float)
            batch["margin"] = margin
        return batch


def _tokenize(batch: dict[str, list[Any]], tokenizer: "PreTrainedTokenizerBase") -> dict[str, list[Any]]:
    """Tokenize a batch from a reward modelling dataset."""
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(batch["chosen"], batch["rejected"]):
        tokenized_chosen = tokenizer(chosen)
        tokenized_rejected = tokenizer(rejected)
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    if "component_name" in batch:
        new_examples["component_name"] = batch["component_name"]

    return new_examples


class RewardTrainer(Trainer):
    _tag_names = ["trl", "reward-trainer"]

    @deprecate_kwarg(
        "tokenizer", "0.15.0", "processing_class", warn_if_greater_or_equal_version=True, raise_if_both_names=True
    )
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: Optional[RewardConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[dict] = None,
        component_to_idx = None,
    ):
        """
        Initialize RewardTrainer.

        Args:
            model (`transformers.PreTrainedModel`):
                The model to train, preferably an `AutoModelForSequenceClassification`.
            args (`RewardConfig`):
                The arguments to use for training.
            data_collator (`transformers.DataCollator`):
                The data collator to use for training. If None is specified, the default data collator (`RewardDataCollatorWithPadding`) will be used
                which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
            train_dataset (`datasets.Dataset`):
                The dataset to use for training.
            eval_dataset (`datasets.Dataset`):
                The dataset to use for evaluation.
            processing_class (`PreTrainedTokenizerBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`, *optional*):
                Processing class used to process the data. If provided, will be used to automatically process the inputs
                for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
                reuse the fine-tuned model.
            model_init (`Callable[[], transformers.PreTrainedModel]`):
                The model initializer to use for training. If None is specified, the default model initializer will be used.
            compute_metrics (`Callable[[transformers.EvalPrediction], dict]`, *optional* defaults to `compute_accuracy`):
                The metrics to use for evaluation. If no metrics are specified, the default metric (`compute_accuracy`) will be used.
            callbacks (`list[transformers.TrainerCallback]`):
                The callbacks to use for training.
            optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
                The optimizer and scheduler to use for training.
            preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
                The function to use to preprocess the logits before computing the metrics.
            peft_config (`dict`, defaults to `None`):
                The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        """
        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if not isinstance(model, PeftModel):
                if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_quantized", False):
                    _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
                        inspect.signature(prepare_model_for_kbit_training).parameters
                    )

                    prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                    if not _supports_gc_kwargs and args.gradient_checkpointing_kwargs is not None:
                        warnings.warn(
                            "You passed `gradient_checkpointing_kwargs` in the trainer's kwargs, but your peft version does not support it. "
                            "please update to the latest version of peft to use `gradient_checkpointing_kwargs`.",
                            UserWarning,
                        )
                    elif _supports_gc_kwargs and args.gradient_checkpointing_kwargs is not None:
                        prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                    model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

                model = get_peft_model(model, peft_config)

        self.component_to_idx = component_to_idx
        # Disable dropout in the model
        if args.disable_dropout:
            disable_dropout_in_model(model)

        if compute_metrics is None:
            compute_metrics = compute_accuracy

        if data_collator is None:
            if processing_class is None:
                raise ValueError(
                    "A processing_class must be specified when using the default RewardDataCollatorWithPadding"
                )

            max_length = args.max_length
            data_collator = RewardDataCollatorWithPadding(processing_class)

            if args.remove_unused_columns:
                try:  # for bc before https://github.com/huggingface/transformers/pull/25435
                    args.remove_unused_columns = False
                except FrozenInstanceError:
                    args = replace(args, remove_unused_columns=False)
                # warn users
                warnings.warn(
                    "When using RewardDataCollatorWithPadding, you should set `remove_unused_columns=False` in your RewardConfig"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_reward_data_collator = True
        else:
            self.use_reward_data_collator = False

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in Reward, the sampled data does not include the
        # "input_ids" key. Instead, the available keys are "input_ids_chosen" and "input_ids_rejected". As a result,
        # the trainer issues the warning: "Could not estimate the number of tokens of the input, floating-point
        # operations will not be computed." To suppress this warning, we set the "estimate_tokens" key in the model's
        # "warnings_issued" dictionary to True. This acts as a flag to indicate that the warning has already been
        # issued.
        model.warnings_issued["estimate_tokens"] = True

        if "input_ids_chosen" not in train_dataset.column_names:
            with PartialState().local_main_process_first():
                fn_kwargs = {"tokenizer": processing_class}
                train_dataset = train_dataset.map(maybe_apply_chat_template, fn_kwargs={"tokenizer": processing_class})
                train_dataset = train_dataset.map(
                    _tokenize,
                    batched=True,
                    fn_kwargs=fn_kwargs,
                    num_proc=args.dataset_num_proc,
                )
                # This filter is important because otherwise you get samples that exceed the model's context length and
                # get truncated => noisy signal the chosen/rejected label gets lost. The downside is that the
                # user might get surprised if N samples are missing from training.
                logger.info(f'# Train data before filtering: {len(train_dataset)}')
                train_dataset = train_dataset.filter(
                    lambda x: len(x["input_ids_chosen"]) <= max_length and len(x["input_ids_rejected"]) <= max_length,
                    num_proc=args.dataset_num_proc,
                )
                logger.info(f'# Train data after filtering: {len(train_dataset)}')

                if eval_dataset is not None:

                    if isinstance(eval_dataset, dict):
                        for key in eval_dataset.keys():
                            eval_dataset[key] = eval_dataset[key].map(
                                maybe_apply_chat_template, fn_kwargs={"tokenizer": processing_class}
                            )
                            eval_dataset[key] = eval_dataset[key].map(
                                _tokenize,
                                fn_kwargs=fn_kwargs,
                                batched=True,
                                num_proc=args.dataset_num_proc,
                            )
                            logger.info(f'# Eval ({key}) data before filtering: {len(eval_dataset[key])}')
                            eval_dataset[key] = eval_dataset[key].filter(
                                lambda x: len(x["input_ids_chosen"]) <= max_length
                                and len(x["input_ids_rejected"]) <= max_length,
                                num_proc=args.dataset_num_proc,
                            )
                            logger.info(f'# Eval ({key}) data after filtering: {len(eval_dataset[key])}')
                    else:
                        eval_dataset = eval_dataset.map(
                            maybe_apply_chat_template, fn_kwargs={"tokenizer": processing_class}
                        )
                        eval_dataset = eval_dataset.map(
                            _tokenize,
                            fn_kwargs=fn_kwargs,
                            batched=True,
                            num_proc=args.dataset_num_proc,
                        )
                        # This filter is important because otherwise you get samples that exceed the model's context length and
                        # get truncated => noisy signal the chosen/rejected label gets lost. The downside is that the
                        # user might get surprised if N samples are missing from training.
                        eval_dataset = eval_dataset.filter(
                            lambda x: len(x["input_ids_chosen"]) <= max_length
                            and len(x["input_ids_rejected"]) <= max_length,
                            num_proc=args.dataset_num_proc,
                        )


        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        self.running = RunningMoments(self.accelerator)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:

        logits_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
            return_dict=True,
        )["logits"]
        logits_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
            return_dict=True,
        )["logits"]

        if self.component_to_idx is not None and "component_name" in inputs:
            component_names = inputs["component_name"]  
            indices = [self.component_to_idx[m] for m in component_names]
            indices_tensor = torch.tensor(indices, device=logits_chosen.device)
            batch_size = logits_chosen.size(0)
            rewards_chosen = logits_chosen[torch.arange(batch_size), indices_tensor]
            rewards_rejected = logits_rejected[torch.arange(batch_size), indices_tensor]
        else:
            # Fallback: use the first head if no component information is provided.
            rewards_chosen = logits_chosen[:, 0]
            rewards_rejected = logits_rejected[:, 0]

        if self.args.use_score_scaling:
            rewards = torch.cat([rewards_chosen, rewards_rejected], dim=0)
            rewards_mean, rewards_std = self.running.update(rewards)
            tensor_to_kwargs = dict(dtype=rewards.dtype, device=rewards.device)
            score_scaling_factor = self.running.std + torch.finfo(rewards.dtype).eps
            if self.args.use_score_norm:
                rewards_chosen = (rewards_chosen - self.running.mean) / score_scaling_factor
                rewards_rejected = (rewards_rejected - self.running.mean) / score_scaling_factor
            else:
                rewards_chosen /= score_scaling_factor
                rewards_rejected /= score_scaling_factor

        if self.args.score_clip is not None:
            # Score clipping
            rewards_dtype = rewards.dtype
            rewards = torch.clip(rewards.float(), -self.args.score_clip, self.args.score_clip).to(dtype=rewards_dtype)

        if "margin" in inputs:
            loss = - (inputs["margin"] * nn.functional.logsigmoid(rewards_chosen - rewards_rejected)).mean()
        else:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        if self.args.center_rewards_coefficient is not None:
            loss += self.args.center_rewards_coefficient * torch.mean((rewards_chosen + rewards_rejected) ** 2)

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()
        # Build a two-column tensor by stacking the chosen and rejected rewards.
        # Each row corresponds to a sample; the first column is the chosen head’s score,
        # the second is the rejected head’s score.
        rewards_chosen = outputs["rewards_chosen"]
        rewards_rejected = outputs["rewards_rejected"]
        logits = torch.stack([rewards_chosen, rewards_rejected], dim=1)  # shape: (batch_size, 2)
        logits = logits.softmax(dim=1)

        # For a binary comparison, we assume that the "correct" (i.e. chosen) side is index 0.
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return loss, logits, labels

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        NOTE: Override this function for evaluating on multiple datasets.
        Evaluates the model and computes aggregate metrics (average and weighted average).

        Args:
            *args: Positional arguments to pass to the superclass's evaluate method.
            **kwargs: Keyword arguments, including "metric_key_prefix" (default: "eval").

        Returns:
            dict: A dictionary containing individual evaluation metrics, their averages,
                and weighted averages.
        """
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset

        if isinstance(eval_dataset, dict):
            keys = list(eval_dataset.keys())

            # Extract unique metric names by matching against prefix
            metric_names = {
                metric[len(f"{metric_key_prefix}_{key}_"):]
                for metric in metrics
                for key in keys
                if metric.startswith(f"{metric_key_prefix}_{key}_")
            }

            # Compute average and weighted average of metrics
            balanced_avg_metrics = {
                name: sum(metrics[f"{metric_key_prefix}_{key}_{name}"] for key in keys) / len(keys)
                for name in metric_names
            }
            avg_metrics = {
                name: sum(metrics[f"{metric_key_prefix}_{key}_{name}"] * len(eval_dataset[key])
                        for key in keys) / sum(len(eval_dataset[key]) for key in keys)
                for name in metric_names
            }

            # Update metrics with computed averages
            overall_metrics = {f"{metric_key_prefix}_{name}_balanced": balanced_avg_metrics[name] for name in metric_names}
            overall_metrics.update({f"{metric_key_prefix}_{name}": avg_metrics[name] for name in metric_names})

            self.log(overall_metrics)

            overall_metrics.update({"completion": True})
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, overall_metrics)

            self._memory_tracker.stop_and_update_metrics(overall_metrics)

            metrics.update(overall_metrics)

            return metrics

        return metrics

    def visualize_samples(self, num_print_samples: int):
        """
        Visualize the reward model logits prediction

        Args:
            num_print_samples (`int`, defaults to `4`):
                The number of samples to print. Set to `-1` to print all samples.
        """
        eval_dataloader = self.get_eval_dataloader()
        table = defaultdict(list)
        for _, inputs in enumerate(eval_dataloader):
            _, logits, _ = self.prediction_step(self.model, inputs, prediction_loss_only=False)
            chosen_text = decode_and_strip_padding(inputs["input_ids_chosen"], self.processing_class)
            rejected_text = decode_and_strip_padding(inputs["input_ids_rejected"], self.processing_class)
            table["chosen_text"].extend(gather_object(chosen_text))
            table["rejected_text"].extend(gather_object(rejected_text))
            table["logits"].extend(
                gather_object([[round(inner_item, 4) for inner_item in item] for item in logits.tolist()])
            )
            if num_print_samples >= 0 and len(table["chosen_text"]) >= num_print_samples:
                break
        df = pd.DataFrame(table)
        if self.accelerator.process_index == 0:
            print_rich_table(df[:num_print_samples])
            if "wandb" in self.args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            