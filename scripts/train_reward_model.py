import os
import wandb
import warnings
import yaml
import torch
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, BitsAndBytesConfig, EarlyStoppingCallback
)
import copy
from dataclasses import dataclass, field
from datasets import load_dataset
from peft import LoraConfig
from typing import List, Optional
import os.path as osp
import json
from dotenv import load_dotenv
import sys

from optimas.train.callback import PerComponentSaveCallback
from optimas.utils.logger import setup_logger
from optimas.train.reward_config import RewardConfig
from optimas.train.reward_trainer import RewardTrainer 
from optimas.utils.load import load_model_and_tokenizer
from optimas.reward_dataset import RewardDataset
from optimas.train.finetune import run_finetune

from examples.systems import registered_systems
from examples.datasets import registered_datasets


@dataclass
class ScriptArgs:
    base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    hf_repo_or_local_dir: str = ""
    system_name: str = ""
    dataset: str = ""

    train_multi_head: bool = True
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    early_stopping_patience: int = 512

    state_dict_path: Optional[str] = None
    save_model_per_component: bool = True

    wandb_run_name: str = ""

    component_name_lst: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for key, value in vars(self).items():
            if isinstance(value, str) and value.strip().lower() == "none":
                setattr(self, key, None)


def main():
    load_dotenv('.env')
    
    # Parse both dataclasses from the same config
    parser = HfArgumentParser([ScriptArgs, RewardConfig])
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        with open(sys.argv[1], "r") as f:
            yaml_config = yaml.safe_load(f)
        args, training_args = parser.parse_dict(yaml_config)
    else:
        args, training_args = parser.parse_args_into_dataclasses()
    
    # Set output_dir based on script args
    output_dir = osp.join(training_args.output_dir, args.system_name)
    training_args.output_dir = output_dir
    training_args.logging_dir = output_dir
    
    logger = setup_logger(__name__, log_file=osp.join(output_dir, "output.log"))

    if os.environ.get('LOCAL_RANK', '0') == '0':
        wandb.init(
            entity=os.environ.get('WANDB_ENTITY'),
            project=os.environ.get('WANDB_PROJECT'),
            name=args.wandb_run_name,
            config={**vars(args), **vars(training_args)},
            save_code=True
        )

    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]
        )
    else:
        peft_config = None

    system = registered_systems[args.system_name]()
    trainset, valset, testset = registered_datasets[args.dataset]()

    reward_dataset = RewardDataset(
        args.hf_repo_or_local_dir, system,
        original_dataset=trainset + valset
        )
    ds = reward_dataset.to_preference_dataset(
        eval_ratio=training_args.eval_ratio,
        add_margin=training_args.add_margin
    )

    num_labels = len(system.optimizable_components) if args.train_multi_head else 1
    logger.info(f"[reward_model_train] Setting the number of output dims to {num_labels}")

    model, tokenizer = load_model_and_tokenizer(
        args.base_model_name,
        peft_config=peft_config,
        state_dict_path=args.state_dict_path,
        num_labels=num_labels
    )

    ############ Callback ############
    save_callback = PerComponentSaveCallback(
        system=system,
        tokenizer=tokenizer,
        repo_name=args.hf_repo_or_local_dir,
        metric_for_best_model=training_args.metric_for_best_model,
        save_model_per_component=args.save_model_per_component
    )
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
    
    ############ Train model ############
    logger.info(f"[reward_model_train] {system.optimizable_components=}")

    trainer = run_finetune(
        ds, model, tokenizer, training_args,  # Pass training_args directly
        train_last_layer=True, 
        callbacks=[save_callback, early_stopping_callback], 
        component_to_idx=system.optimizable_component_to_idx
    )

    ############ SAVING ############
    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    wandb.finish()

if __name__ == "__main__":
    main()