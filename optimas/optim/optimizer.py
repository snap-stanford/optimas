import copy
import torch
from typing import Any, Dict, List
from datasets import DatasetDict
import json
import itertools
import os
import random   
from pathlib import Path
import numpy as np
import random
import wandb
from collections import defaultdict, deque
from datasets import Dataset
import torch.distributed as dist
from contextlib import contextmanager
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from optimas.arch.system import CompoundAISystem
from optimas.collect.rollouts import generate_reward_dataset
from optimas.eval import eval_system
from optimas.reward_model import RewardModel
from optimas.optim.args import OptimasArguments
from optimas.optim.cp_optimizer import ComponentOptimizer
from optimas.reward_dataset import RewardDataset
from optimas.train.finetune import run_finetune
from optimas.train.reward_config import RewardConfig
from optimas.utils.logger import setup_logger
from optimas.utils.lora import *
from optimas.utils.parallel import run_parallel_tasks
from optimas.wrappers.example import Example
from optimas.wrappers.prediction import Prediction


logger = setup_logger(__name__)


class OptimasOptimizer:
    def __init__(
        self,
        args: OptimasArguments, 
        training_args: RewardConfig,
        system: CompoundAISystem,
        train_dataset: DatasetDict,
        val_dataset: DatasetDict,
        reward_model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        preference_dataset: DatasetDict = None
    ):
        self.args = args
        self.training_args = training_args
        self.system = system
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.preference_dataset = preference_dataset
        
        self.optimizable_components = list(preference_dataset.keys()) if preference_dataset else \
                                      list(system.optimizable_component_to_idx.keys())
        self.output_dir = self.training_args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.replay_buffer = {c: deque(maxlen=self.args.replay_buffer_size) if self.args.use_replay_buffer else None for c in self.optimizable_components}
        self._fill_in_replay_buffer(preference_dataset)
        
        # Track recently optimized components
        self.recently_optimized = {}  # Module name -> iteration when last optimized

        # Track current adapter paths for local_lm components
        self.current_adapters = {}  # component_name -> adapter_path

        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.is_main_process = not self.is_distributed or dist.get_rank() == 0
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
    
    def _fill_in_replay_buffer(self, dataset_dict):

        if self.args.use_replay_buffer and dataset_dict:
            for component_name, dataset in dataset_dict.items():
                for i in range(len(dataset)):
                    sample = {
                        "context": dataset["context"][i],
                        "response_chosen": dataset["response_chosen"][i],
                        "response_rejected": dataset["response_rejected"][i],
                        "score_chosen": dataset["score_chosen"][i],
                        "score_rejected": dataset["score_rejected"][i],
                    }
                    if "margin" in dataset.column_names:
                        sample["margin"] = dataset["margin"][i]
                    self.replay_buffer[component_name].append(sample)

                logger.info(
                    f"{component_name} buffer size = {len(self.replay_buffer[component_name])}/{self.args.replay_buffer_size}"
                )

    def select_component_to_optimize(self, preference_dataset, current_iteration):
        """
        Select a component to optimize based on the score gaps in preference data.
        Uses softmax to probabilistically select components with higher average gaps.
        Args:
            preference_dataset: DatasetDict containing preference pairs for each component
            current_iteration: The current optimization iteration
        Returns:
            String: The selected component name, or None if no eligible components
        """
        if not preference_dataset:
            return random.choice(self.optimizable_components)

        # Calculate average score gap for each component
        component_gaps = {}
        for component_name, dataset in preference_dataset.items():
            # Skip components that were recently optimized (still in cooldown)
            if (
                component_name in self.recently_optimized
                and current_iteration - self.recently_optimized[component_name]
                < self.args.cooldown_period
            ):
                logger.info(
                    f"Module {component_name} is in cooldown period, skipping"
                )
                continue

            # Calculate average gap between chosen and rejected scores
            score_chosen = dataset["score_chosen"]
            score_rejected = dataset["score_rejected"]
            if len(score_chosen) == 0:
                continue

            avg_gap = sum(c - r for c, r in zip(score_chosen, score_rejected)) / len(
                score_chosen
            )
            component_gaps[component_name] = avg_gap

        if not component_gaps:
            logger.warning(
                "No eligible components found (all in cooldown or no data)"
            )
            return None

        # Apply softmax to create probability distribution
        component_names = list(component_gaps.keys())
        gap_values = [component_gaps[name] for name in component_names]

        # Convert to torch tensor and apply softmax
        gaps_tensor = torch.tensor(gap_values)
        probs = torch.nn.functional.softmax(gaps_tensor, dim=0).numpy()

        # Sample a component based on probabilities
        selected_idx = np.random.choice(len(component_names), p=probs)

        # log each component's probability
        wandb.log(
            {"iteration": current_iteration,
             **{f"prob_{component_name}": prob for component_name, prob in zip(component_names, probs)}}
        )

        selected_component = component_names[selected_idx]
        logger.info(
            f"Selected component {selected_component} for optimization with probability {probs[selected_idx]:.4f}"
        )

        # Log all component probabilities for transparency
        for i, name in enumerate(component_names):
            logger.info(
                f"Module {name}: gap={gap_values[i]:.4f}, probability={probs[i]:.4f}"
            )
        return selected_component

    def _is_local_lm_component(self, component_name):
        component = self.system.components[component_name]
        return isinstance(component.variable, Path)

    def _update_system_with_adapter(self, component_name, adapter_path):
        """Update system to use the new adapter for a local LLM component"""
        # Store the adapter path
        self.current_adapters[component_name] = adapter_path

        # Update the component's adapter_path to use the new adapter
        component = self.system.components[component_name]
        component.update(adapter_path)
        
    def _get_current_adapter_state(self):
        """Get the current state of all adapters for saving/loading"""
        return copy.deepcopy(self.current_adapters)

    def _restore_adapter_state(self, adapter_state):
        """Restore adapter state and update system accordingly"""
        for component_name, adapter_path in adapter_state.items():
            if adapter_path and self._is_local_lm_component(component_name):
                self._update_system_with_adapter(component_name, adapter_path)
        self.current_adapters = copy.deepcopy(adapter_state)

    def optimize_component(self, component_name, hf_repo_or_local_dir):
        """
        Optimize a specific component using techniques from the existing framework.
        For local LLM components, this includes PPO training.
        """
        logger.info(f"Optimizing component: {component_name}")

        self.reward_model.eval()
        reward_model = RewardModel(self.reward_model, self.tokenizer, self.system)

        # Prepare dataset specific to this component
        reward_dataset = RewardDataset(
            system=self.system,
            hf_repo_or_local_dir=hf_repo_or_local_dir,
            original_dataset=self.train_dataset + self.val_dataset
        ).to_inputs_only_dataset()

        logger.info(f'Training dataset size: {len(reward_dataset[component_name]["input"])}')

        # Clone and modify args for this specific component
        component_args = copy.deepcopy(self.args)

        # Override with component-specific settings
        component_args.components_to_apply = [
            component_name
        ]  # Focus only on this component
        component_args.per_component_train_size = min(
            len(reward_dataset[component_name]["input"]), component_args.per_component_train_size
        )
        component_args.output_dir = self.output_dir

        logger.info(
            f"Configured OptimasArguments for {component_name}: {vars(component_args)}"
        )

        # Initialize and run optimizer with the args from command line
        optimizer = ComponentOptimizer(
            args=component_args,
            system=self.system,
            reward_model=reward_model,
            reward_dataset=reward_dataset,
            original_trainset=self.train_dataset
        )

        # Optimize just this component
        optimized_system = optimizer.optimize()

        # For local LLM components, check if PPO training produced new adapters
        if self._is_local_lm_component(component_name):
            ppo_output_dir = self.system.components[component_name].variable
            if os.path.exists(ppo_output_dir):
                self._update_system_with_adapter(component_name, ppo_output_dir)
            else:
                logger.info(f"No PPO output directory found: {ppo_output_dir}")

        return optimized_system

    def train_reward_model(self, component_name, hf_repo_or_local_dir, per_iteration_rm_train_size=-1):
        """
        Train a reward model on the collected preference data for the specified component.
        Args:
            component_name: Name of the component to train the reward model for
            hf_repo_or_local_dir: Path to the preference dataset
        Returns:
            Trained reward model
        """
        if self.is_main_process:
            logger.info(f"Training reward model for component: {component_name}")

        # Create component-specific output directory
        component_output_dir = os.path.join(self.output_dir, f"reward_model_{component_name}")
        os.makedirs(component_output_dir, exist_ok=True)

        # Training configuration
        self.training_args.output_dir = component_output_dir
        self.training_args.logging_dir = component_output_dir

        # Load dataset
        reward_dataset = RewardDataset(
            system=self.system,
            hf_repo_or_local_dir=hf_repo_or_local_dir, 
            original_dataset=self.train_dataset + self.val_dataset,
        )
        # Format the dataset
        ds = reward_dataset.to_preference_dataset(eval_ratio=self.training_args.eval_ratio, add_margin=self.training_args.add_margin) 

        if per_iteration_rm_train_size != -1:
            # shuffle and take the first per_iteration_rm_train_size samples
            random.seed(42)
            train_list = ds["train"].to_list()
            random.shuffle(train_list)
            train_list = train_list[:per_iteration_rm_train_size]
            ds["train"] = Dataset.from_list(train_list)
        # Otherwise use the full reward dataset
        
        # Train the reward model
        trainer = run_finetune(
            ds,
            self.reward_model,
            self.tokenizer,
            self.training_args,
            train_last_layer=True,
            component_to_idx={
                component_name: self.system.optimizable_component_to_idx[component_name]
            }
        )

        return trainer.model
    
    def _wait_for_everyone(self):
        """Wait for all processes to reach this point"""
        if self.is_distributed:
            dist.barrier()
    
    def _gather_scalar(self, scalar_value, src_rank=0):
        """Gather scalar from main process and broadcast to all"""
        if self.is_distributed:
            if scalar_value is not None:
                tensor = torch.tensor([scalar_value], dtype=torch.float32, device=torch.cuda.current_device())
            else:
                tensor = torch.tensor([0.0], dtype=torch.float32, device=torch.cuda.current_device())
            
            dist.broadcast(tensor, src_rank)
            return tensor.item()
        return scalar_value
    
    def _gather_object(self, obj, src_rank=0):
        """Gather object from main process and broadcast to all"""
        if self.is_distributed:
            gathered = [None] * self.world_size
            dist.all_gather_object(gathered, obj if self.rank == src_rank else None)
            return gathered[src_rank]
        return obj
    
    def _sync_state_dict(self, state_dict, src_rank=0):
        """Synchronize state dict across all processes"""
        if self.is_distributed:
            if self.rank == src_rank:
                # Main process has the state dict
                state_list = [state_dict]
            else:
                state_list = [None]
            
            dist.broadcast_object_list(state_list, src_rank)
            return state_list[0]
        return state_dict

    def optimize(self):
        """
        Main optimization loop using the existing preference dataset to select components.
        Following TRL's distributed training patterns.
        """
        # Store initial system state dict for comparison
        original_state_dict = self.system.state_dict()
        original_adapter_state = self._get_current_adapter_state()

        # Optimization history
        history = {"iterations": [], "overall_performance": []}
        
        # Set reward_model at the beginning
        local_hyper_param_search = any([component.variable_search_space for component in self.system.components.values()])
        if not self.args.global_hyper_param_search and local_hyper_param_search:
            self.system.register_rm(
                RewardModel(self.reward_model, self.tokenizer, self.system), sample_size=1
            )

        # Evaluate initial performance - ONLY MAIN PROCESS
        initial_score = None
        if self.is_main_process:
            try:
                metrics, _ = eval_system(system=self.system, testset=self.val_dataset, num_repeat=1)
                initial_score = metrics['mean']
                logger.info(f"Initial score: {initial_score:.4f}")
            except Exception as e:
                logger.warning(f"Error evaluating initial system: {e}. Setting initial score to 0.")
                initial_score = 0

        # Broadcast initial score to all processes
        initial_score = self._gather_scalar(initial_score, src_rank=0)

        history["overall_performance"].append(initial_score)

        # Best score and system state so far
        best_score = initial_score
        best_state_dict = original_state_dict
        best_adapter_state = original_adapter_state
        current_best_iteration = 0

        # Log only from main process
        if self.is_main_process:
            wandb.log({
                "iteration": 0,
                "eval/score": best_score,
                "eval/best_score": best_score
            })

        skip_data_gen = self.args.skip_data_gen
        flag_data_gen = self.preference_dataset is None

        if skip_data_gen and not self.preference_dataset:
            if self.is_main_process:
                logger.error(
                    "Skipping online data generation but no preference dataset provided. "
                    "Will not be able to optimize components."
                )
            return self.system, history

        # Log preference dataset stats - main process only
        if self.is_main_process and self.preference_dataset:
            logger.info("Using provided preference dataset")
            for component_name, dataset in self.preference_dataset.items():
                logger.info(f"  {component_name}: {len(dataset)} preference pairs")

        # Wait for all processes to be ready
        self._wait_for_everyone()
            
        for self.iteration in range(self.args.num_iterations):
            
            # Component selection - ONLY MAIN PROCESS
            component_to_optimize = None
            if self.is_main_process:
                logger.info(f"Starting iteration {self.iteration+1}/{self.args.num_iterations}")

                component_to_optimize = self.select_component_to_optimize(
                    self.preference_dataset, self.iteration
                )
            
                if not component_to_optimize:
                    logger.warning("No suitable component to optimize, ending optimization")
                    break

                logger.info(f"Selected component to optimize: {component_to_optimize}")
        
            # Broadcast selected component to all processes
            component_to_optimize = self._gather_object(component_to_optimize, src_rank=0)
                    
            # Data generation - ONLY MAIN PROCESS
            dataset_path = None
            if self.is_main_process:
                if skip_data_gen or not flag_data_gen:
                    logger.info(
                        "Skipping new training dataset generation: either skip_data_gen is enabled "
                        "or reusing existing preference_dataset until performance improves"
                    )
                    new_preference_dataset = {component_to_optimize: []}
                else:
                    subset_size = min(len(self.train_dataset), self.args.per_iteration_new_data_size)
                    dataset_subset = random.sample(self.train_dataset, subset_size)
                    new_preference_dataset = generate_reward_dataset(
                        system=self.system,
                        dataset=dataset_subset,
                        component_names=[component_to_optimize],
                        num_rollouts=self.args.num_rollouts,
                        num_samples=self.args.num_samples,
                        max_workers=self.args.max_workers,
                        num_repeats=self.args.num_repeats
                    )

                # Build training dataset for reward-model update
                if self.args.use_replay_buffer:
                    # merge new data with buffer data
                    fill_in_size = max(self.args.per_iteration_rm_train_size, self.args.per_component_train_size) - \
                                   len(new_preference_dataset[component_to_optimize])

                    buffer_data = list(self.replay_buffer[component_to_optimize])[-fill_in_size:]
                    buffer_data.extend(new_preference_dataset[component_to_optimize])
                    buffer_data = {key: [item[key] for item in buffer_data] for key in buffer_data[0].keys()}

                    training_dataset = DatasetDict({component_to_optimize: Dataset.from_dict(buffer_data)})

                    dataset_path = os.path.join(
                        self.output_dir,
                        f"buffer_ds_{self.iteration}_{component_to_optimize}"
                    )
                    training_dataset.save_to_disk(dataset_path)

                    self._fill_in_replay_buffer(new_preference_dataset)

                elif (not flag_data_gen) or skip_data_gen:
                    dataset_path = os.path.join(
                        self.output_dir,
                        f"ds_{self.iteration}_{component_to_optimize}",
                    )
                    selected_data = {component_to_optimize: self.preference_dataset[component_to_optimize]}
                    DatasetDict(selected_data).save_to_disk(dataset_path)
                else:
                    dataset_path = os.path.join(
                        self.output_dir,
                        f"new_ds_{self.iteration}_{component_to_optimize}",
                    )
                    new_preference_dataset.save_to_disk(dataset_path)

            # Wait for data preparation to complete
            self._wait_for_everyone()
            
            # Broadcast dataset path to all processes
            dataset_path = self._gather_object(dataset_path, src_rank=0)
            
            # REWARD MODEL TRAINING - ALL PROCESSES
            self.reward_model = self.train_reward_model(
                component_to_optimize, dataset_path, self.args.per_iteration_rm_train_size
            )
            
            # Wait for reward model training to complete
            self._wait_for_everyone()

            # Save current state - only main process
            current_state_dict = None
            current_adapter_state = None
            if self.is_main_process:
                current_state_dict = self.system.state_dict()
                current_adapter_state = self._get_current_adapter_state()

            # Component optimization - only main process
            optimized_system = None
            if self.is_main_process:
                optimized_system = self.optimize_component(
                    component_to_optimize, dataset_path
                )
            
            # Synchronize optimized system state to all processes
            if optimized_system is not None:
                optimized_state_dict = optimized_system.state_dict()
            else:
                optimized_state_dict = None
                
            optimized_state_dict = self._sync_state_dict(optimized_state_dict, src_rank=0)
            
            # Load synchronized state on all processes
            if optimized_state_dict is not None:
                self.system.load_state_dict(optimized_state_dict)
                optimized_system = self.system

            if not self.args.global_hyper_param_search and local_hyper_param_search:
                self.system.register_rm(
                    RewardModel(self.reward_model, self.tokenizer, self.system), sample_size=1
                )

            # EVALUATION - ONLY MAIN PROCESS
            new_score = None
            if self.is_main_process:
                try:
                    metrics, _ = eval_system(system=optimized_system, testset=self.val_dataset, num_repeat=1)
                    new_score = metrics['mean']
                    logger.info(
                        f"After optimizing {component_to_optimize}, score: {new_score:.4f} (previous: {best_score:.4f})"
                    )

                    # Record this optimization attempt
                    history["iterations"].append({
                        "iteration": self.iteration + 1,
                        "current_best_iteration": current_best_iteration,
                        "component_optimized": component_to_optimize,
                        "score_before": best_score,
                        "score_after": new_score,
                        "improvement": new_score - best_score,
                    })

                    if self.preference_dataset:
                        history["iterations"][-1]["gap_data"] = {
                            "num_pairs": len(self.preference_dataset[component_to_optimize]),
                            "avg_gap": sum(self.preference_dataset[component_to_optimize]["score_chosen"]) / len(self.preference_dataset[component_to_optimize])
                                        - sum(self.preference_dataset[component_to_optimize]["score_rejected"]) / len(self.preference_dataset[component_to_optimize]),
                        }

                    wandb.log({
                        "iteration": self.iteration + 1,
                        "component_idx": self.system.optimizable_component_to_idx[component_to_optimize],
                        "eval/score": new_score,
                        "eval/best_score": best_score
                    })

                except Exception as e:
                    logger.error(f"Error evaluating optimized system: {e}. Reverting.")
                    new_score = best_score

                # Broadcast evaluation results to all processes
                new_score = self._gather_scalar(new_score, src_rank=0)
                
                # All processes check for improvement
                improvement = new_score > best_score
                
                if improvement:
                    if self.is_main_process:
                        logger.info(f"Performance improved from {best_score:.4f} to {new_score:.4f}")
                    
                    flag_data_gen = True
                    best_score = new_score
                    best_state_dict = optimized_system.state_dict()
                    best_adapter_state = self._get_current_adapter_state()
                    current_best_iteration = self.iteration + 1

                    # Save improved system state - only main process
                    if self.is_main_process:
                        torch.save(
                            best_state_dict,
                            os.path.join(
                                self.output_dir,
                                f"system_state_iteration_{self.iteration+1}_{component_to_optimize}.pth",
                            ),
                        )

                    # Mark this component as recently optimized
                    self.recently_optimized[component_to_optimize] = self.iteration
                else:
                    if self.is_main_process:
                        logger.info(f"No improvement from optimizing {component_to_optimize}, reverting")
                    
                    # Revert to previous state - all processes
                    if self.is_main_process and current_state_dict is not None:
                        self.system.load_state_dict(current_state_dict)
                        self._restore_adapter_state(current_adapter_state)
                        
                        # Sync reverted state to all processes
                        reverted_state = self.system.state_dict()
                    else:
                        reverted_state = None
                        
                    reverted_state = self._sync_state_dict(reverted_state, src_rank=0)
                    if reverted_state is not None:
                        self.system.load_state_dict(reverted_state)

            else:
                if self.is_main_process:
                    logger.warning(f"No preference data available for component {component_to_optimize}")

            # Update to best state so far for next iteration - all processes
            self.system.load_state_dict(best_state_dict)
            self._restore_adapter_state(best_adapter_state)

            # Update performance history - only main process
            if self.is_main_process:
                history["overall_performance"].append(best_score)
                
            # Wait for all processes before next iteration
            self._wait_for_everyone()

        # Final improvement calculation - only main process
        if self.is_main_process:
            improvement = best_score - initial_score
            logger.info(f"Overall improvement: {improvement:.4f} ({initial_score:.4f} to {best_score:.4f})")
            history["overall_improvement"] = improvement

        # Ensure we return the system with the best state - all processes
        self.system.load_state_dict(best_state_dict)
        self._restore_adapter_state(best_adapter_state)

        return self.system, history
