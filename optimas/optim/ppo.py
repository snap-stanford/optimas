import os
import math
import wandb
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from optimas.arch.system import CompoundAISystem
from optimas.reward_model import RewardModel
from optimas.utils.logger import setup_logger

logger = setup_logger(__name__)

def train_ppo(
    args: argparse.Namespace,
    dataset: Dataset,
    system: CompoundAISystem,
    component_name: str,
    reward_model: RewardModel,
    lora_cfg: LoraConfig,
    run_name: str,
    output_dir: str,
    *,
    temperature: float = 0.7,
    max_new_tokens: int = 512
) -> str:
    """
    Finetune a *single* component with PPO and return the directory
    containing the up-to-date LoRA adapter.
    """
    assert len(system.components[component_name].output_fields) == 1
    config = system.components[component_name].config

    args.max_new_tokens = getattr(config, "max_new_tokens", max_new_tokens)
    args.temperature = getattr(config, "temperature", temperature)
    os.makedirs(output_dir, exist_ok=True)

    # ───────────────────────────── Load Tokenizer and Model
    tok = AutoTokenizer.from_pretrained(args.ppo_base_model_name)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base_policy = AutoModelForCausalLM.from_pretrained(args.ppo_base_model_name)
    peft_model = get_peft_model(base_policy, lora_cfg)
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model)
    policy.config.use_cache = False

    if not hasattr(policy, "generation_config"):
        policy.generation_config = GenerationConfig.from_pretrained(args.ppo_base_model_name)

    device = f"cuda:{os.getenv('LOCAL_RANK', '0')}" if torch.cuda.is_available() else "cpu"
    policy.to(device)
    logger.info(f"Using device: {device}")

    # ───────────────────────────── Resume From Adapter If Applicable
    if args.ppo_resume_adapter:
        policy.pretrained_model.load_adapter(
            args.ppo_resume_adapter, adapter_name="default", is_trainable=True
        )

    # ───────────────────────────── Prepare Dataset
    logger.info(f"Preparing dataset for component: {component_name}")
    def add_prompt(example):
        example["prompt"] = reward_model.process_prompt(system.components[component_name], **example)
        return example

    dataset = dataset.map(add_prompt)
    dl = DataLoader(
        dataset,
        batch_size=args.ppo_batch_size,
        shuffle=True,
        collate_fn=lambda x: x,
        drop_last=False
    )

    # ───────────────────────────── PPO Trainer Setup
    ppo_cfg = PPOConfig(
        batch_size=args.ppo_batch_size,
        mini_batch_size=args.ppo_mini_batch_size,
        gradient_accumulation_steps=args.ppo_gradient_accumulation_steps,
        ppo_epochs=args.ppo_epochs,
        learning_rate=args.ppo_learning_rate,
        log_with='wandb',
        whiten_rewards=True,
    )

    trainer = PPOTrainer(
        config=ppo_cfg,
        model=policy,
        tokenizer=tok,
        ref_model=None,
    )

    # ───────────────────────────── Training Loop
    step = 0
    for epoch in range(args.ppo_epochs):
        logger.info(f"\n=== Starting Epoch {epoch + 1}/{args.ppo_epochs} ===")
        epoch_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
        os.makedirs(epoch_dir, exist_ok=True)

        epoch_pbar = tqdm(dl, desc=f"Epoch {epoch + 1}", dynamic_ncols=True)
        epoch_rewards, epoch_losses = [], []

        for batch_idx, batch in enumerate(epoch_pbar):
            if not batch:
                continue

            prompts = [ex["prompt"] for ex in batch]
            query_tensors = tok(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

            gen_tensors = policy.generate(
                query_tensors,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                pad_token_id=tok.eos_token_id,
                do_sample=True,
            )
            resp_tensors = [gen[len(q):] for gen, q in zip(gen_tensors, query_tensors)]
            completions = tok.batch_decode(resp_tensors, skip_special_tokens=True)
            logger.debug(f"Generated completions: {completions}")

            inputs_n_outputs = [
                {
                    system.components[component_name].output_fields[0]: c.strip(),
                    **{input_field: ex[input_field] for input_field in system.components[component_name].input_fields}
                }
                for c, ex in zip(completions, batch)
            ]

            rewards = reward_model.batch_evaluate(component_name, inputs_n_outputs)
            rewards = [torch.tensor(r, dtype=torch.float, device=device) for r in rewards]
            mean_reward = sum(rewards) / len(rewards)
            epoch_rewards.extend(rewards)

            # Handle dynamic batch sizing
            original_cfg = (
                trainer.config.batch_size,
                trainer.config.mini_batch_size,
                trainer.config.gradient_accumulation_steps,
                trainer.config.backward_batch_size,
            )

            current_bs = len(prompts)
            if current_bs != trainer.config.batch_size:
                trainer.config.batch_size = current_bs
                trainer.config.mini_batch_size = min(current_bs, trainer.config.mini_batch_size)
                trainer.config.gradient_accumulation_steps = max(
                    1, math.ceil(current_bs / trainer.config.mini_batch_size)
                )
                trainer.config.backward_batch_size = (
                    trainer.config.mini_batch_size * trainer.config.gradient_accumulation_steps
                )
                trainer.backward_batch_size = trainer.config.backward_batch_size

            stats = trainer.step(
                queries=[t.detach() for t in query_tensors],
                responses=[t.detach() for t in resp_tensors],
                scores=rewards,
            )

            # Reset trainer config
            (
                trainer.config.batch_size,
                trainer.config.mini_batch_size,
                trainer.config.gradient_accumulation_steps,
                trainer.config.backward_batch_size,
            ) = original_cfg

            if stats and "ppo/loss/total" in stats:
                loss_value = float(stats["ppo/loss/total"])
                epoch_losses.append(loss_value)
            else:
                loss_value = 0.0

            epoch_pbar.set_postfix({
                'reward': f"{mean_reward:.4f}",
                'loss': f"{loss_value:.4f}",
            })

            wandb.log({
                f"{run_name}_epoch": epoch + 1,
                f"{run_name}_batch": batch_idx,
                f"{run_name}_step": step,
                f"{run_name}_mean_reward": mean_reward,
                f"{run_name}_loss": loss_value,
            })

            step += 1

            if step % args.ppo_save_steps == 0:
                checkpoint_dir = os.path.join(output_dir, f"step_{step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                policy.save_pretrained(checkpoint_dir, safe_serialization=True)
                tok.save_pretrained(checkpoint_dir)
                logger.info(f"Saved checkpoint at step {step} to {checkpoint_dir}")

        # ───────────────────────────── End-of-Epoch Logging & Saving
        epoch_mean_reward = sum(epoch_rewards) / len(epoch_rewards)
        logger.info(f"Epoch {epoch + 1} mean reward: {epoch_mean_reward:.4f}")

        if epoch_losses:
            epoch_mean_loss = sum(epoch_losses) / len(epoch_losses)
            logger.info(f"Epoch {epoch + 1} mean loss: {epoch_mean_loss:.4f}")
        else:
            epoch_mean_loss = 0.0

        wandb.log({
            f"{run_name}_epoch": epoch + 1,
            f"{run_name}_epoch_mean_reward": epoch_mean_reward,
            f"{run_name}_epoch_mean_loss": epoch_mean_loss,
        })

        policy.save_pretrained(epoch_dir, safe_serialization=True)
        tok.save_pretrained(epoch_dir)
        logger.info(f"Saved model for epoch {epoch + 1} to {epoch_dir}")

    # ───────────────────────────── Final Save
    policy.save_pretrained(output_dir, safe_serialization=True)
    tok.save_pretrained(output_dir)
    logger.info(f"Saved final model to {output_dir}")
