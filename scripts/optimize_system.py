import sys
import os
import json
import yaml
import datetime
from peft import LoraConfig
from typing import List, Optional

import torch
import numpy as np
import wandb
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, HfArgumentParser
from dataclasses import dataclass, field

from optimas.eval import eval_system
from optimas.utils.logger import setup_logger
from optimas.optim.optimizer import OptimasOptimizer
from optimas.optim.args import OptimasArguments
from optimas.train.reward_config import RewardConfig
from optimas.utils.load import load_model_and_tokenizer
from examples.systems import registered_systems
from examples.datasets import registered_datasets


@dataclass
class ScriptArgs:
    # Dataset and pipeline
    dataset: str
    system: str
    wandb_run_name: str
    val_size: int = 10
    num_repeat: int = 1
    max_sample_workers: int = 4
    dotenv_path: str = ".env"

    # Model and data paths
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    state_dict_path: Optional[str] = None
    system_state_dict_path: Optional[str] = None
    preference_dataset: str = ""

    # Reward model
    train_multi_head: bool = True

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.0


if __name__ == "__main__":

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    # print rank and world size
    parser = HfArgumentParser([ScriptArgs, RewardConfig, OptimasArguments])
    args, training_args, optimas_args = parser.parse_dict(config)
    load_dotenv('.env')

    # Customize output dir with timestamp
    training_args.output_dir = os.path.join(training_args.output_dir, args.dataset, args.system, args.wandb_run_name)
    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging
    logger = setup_logger(
        __name__, log_file=os.path.join(output_dir, "output.log")
    )

    if os.environ.get('LOCAL_RANK', '0') == '0':
        wandb.init(
            entity=os.environ.get('WANDB_ENTITY'),
            project=os.environ.get('WANDB_PROJECT'),
            name=f"{args.wandb_run_name}_{args.dataset}",
            config=args,
            save_code=True
        )
        logger.info(f"Arguments: {args}")
        logger.info(f"Optimas Arguments: {optimas_args}")
        
    # Load environment variables
    load_dotenv(args.dotenv_path)

    # Initialize system with appropriate parameters based on system type
    system = registered_systems[args.system](
        log_dir=output_dir, max_sample_workers=args.max_sample_workers
    )

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    trainset, valset, testset = registered_datasets[args.dataset]()
    valset = valset[:args.val_size]

    logger.info(
        f"Dataset sizes - Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset) if testset else 'N/A'}"
    )

    # Load preference dataset
    logger.info(f"Loading preference dataset from {args.preference_dataset}")
    preference_dataset = load_dataset(args.preference_dataset)

    # Load initial system state if provided
    if args.system_state_dict_path:
        logger.info(f"Loading system state from {args.system_state_dict_path}")
        system_state_dict = torch.load(args.system_state_dict_path)
        system.load_state_dict(system_state_dict)

    # Configure LoRA for the reward model
    logger.info(f"Initializing LoRA-based reward model with {args.base_model}")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Load initial model for reward training
    model, tokenizer = load_model_and_tokenizer(
        args.base_model,
        peft_config=peft_config,
        num_labels=(
            len(system.optimizable_components) if args.train_multi_head else 1
        ),
        state_dict_path=args.state_dict_path,
    )

    # Create optimizer
    logger.info("Initializing On-Policy Optimizer")
    optimizer = OptimasOptimizer(
        args=optimas_args,  
        training_args=training_args,
        system=system,
        train_dataset=trainset,
        val_dataset=valset,
        preference_dataset=preference_dataset,
        reward_model=model,
        tokenizer=tokenizer
    )

    # Run optimization
    logger.info("Starting optimization process")
    optimized_system, history = optimizer.optimize()

    # Save optimization history
    history_path = os.path.join(output_dir, "optimization_history.json")
    logger.info(f"Saving optimization history to {history_path}")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Evaluate final system on test set if available
    if testset and len(testset) > 0:
        logger.info("Evaluating optimized system on test set")
        try:
            metrics, preds = eval_system(
                system=optimized_system, testset=testset, num_repeat=args.num_repeat
            )
            metrics_path = os.path.join(output_dir, "eval_metrics.json")
            logger.info(f"Saving evaluation metrics to {metrics_path}")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, sort_keys=True, indent=2)

            wandb.log(
                {   "iteration": optimas_args.num_iterations,
                    "test/score": metrics["mean"],
                    "test/std": metrics["std"],
                }
            )
        except Exception as e:
            logger.error(f"Error evaluating optimized system: {e}")

    # Save final optimized system
    final_system_path = os.path.join(output_dir, "final_optimized_system.pth")
    logger.info(f"Saving final optimized system to {final_system_path}")
    torch.save(optimized_system.state_dict(), final_system_path)

    # Save final adapter state
    final_adapter_state = optimizer._get_current_adapter_state()
    with open(os.path.join(output_dir, "final_adapter_state.json"), "w") as f:
        json.dump(final_adapter_state, f, indent=2)

    # Log improvement summary
    if "overall_improvement" in history:
        improvement = history["overall_improvement"]
        logger.info(f"Final improvement: {improvement:.4f}")

        # Create a summary file with key metrics
        summary = {
            "timestamp": datetime.datetime.now().isoformat(),
            "system": args.system,
            "dataset": args.dataset,
            "iterations": optimas_args.num_iterations,
            "initial_score": history["overall_performance"][0],
            "final_score": history["overall_performance"][-1],
            "improvement": improvement,
            "components_optimized": (
                [it["component_optimized"] for it in history["iterations"]]
                if "iterations" in history
                else []
            ),
            "final_adapter_state": final_adapter_state,
        }

        with open(os.path.join(output_dir, "optimization_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    logger.info(f"Optimization complete. Final system saved to {final_system_path}")
    logger.info(f"Overall improvement: {history.get('overall_improvement', 'N/A')}")
    logger.info(f"Final adapter state: {final_adapter_state}")
