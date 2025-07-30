import copy
import torch
from typing import Any, Dict, List
from datasets import DatasetDict
from optimas.optim.args import OptimasArguments
from optimas.arch.system import CompoundAISystem
from optimas.reward_model import RewardModel
from optimas.utils.logger import setup_logger
from optimas.utils.parallel import run_parallel_tasks
from optimas.wrappers.prediction import Prediction
from optimas.optim.opro import OPRO
from optimas.optim.ppo import train_ppo
from optimas.wrappers.example import Example
from optimas.utils.lora import *
import json
import itertools
import os
from pathlib import Path


import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict, deque
from datasets import Dataset
from optimas.collect.rollouts import generate_reward_dataset
from optimas.eval import eval_system
from optimas.reward_dataset import RewardDataset
import wandb

logger = setup_logger(__name__)
OPTIM_CNT = 0

class ComponentOptimizer:
    """
    An optimizer class to optimize the variables (e.g., prompts and parameters) of a CompoundAISystem.
    """

    def __init__(
        self,
        args: OptimasArguments,
        system: CompoundAISystem,
        reward_model: RewardModel,
        reward_dataset: DatasetDict,
        original_trainset: DatasetDict
    ):
        """
        Initialize the Optimizer.

        Args:
            system (CompoundAISystem): The system to optimize.
            reward_dataset (list): A dataset of input-output pairs for training or evaluation.
            args (OptimasArguments): Configuration arguments for the optimization process.
        """
        self.args = args
        self.system = system
        assert self.system.rm is None or self.system.sample_size == 1, "Pipeline already has a reward model. If sample size > 1, it would affect the optimization "

        self.reward_dataset = reward_dataset
        self.hyper_param_train_dataset = original_trainset[:self.args.per_component_search_size]
        # Load the reward model and tokenizer
        self.reward_model = reward_model

        optimizable_components = [
            component_name for component_name, component in system.components.items()
            if component.optimizable
        ]
        visible_components = list(system.components.keys()) if "all" in args.components_to_apply else args.components_to_apply
        self.components_to_apply = list(set(visible_components) & set(optimizable_components))

    def optimize(self) -> Any:
        """
        Optimize the CompoundAISystem.

        Returns:
            CompoundAISystem: The optimized system with updated variables.
        """
        for name, component in self.system.components.items():
            if not name in self.components_to_apply:
                continue

            if isinstance(component.variable, Path):
                assert self.args.ppo_epochs > 0, "ppo_epochs must be greater than 0"
                logger.info(f"Optimizing weights for component {name}...")
                self._optimize_weights(name)

            if isinstance(component.variable, str):
                logger.info(f"Optimizing prompt for component {name}...")
                self._optimize_prompts(name)

            if isinstance(component.variable, dict) and self.args.global_hyper_param_search:
                logger.info(f"Optimizing hyperparameters for component {name}...")
                self._optimize_hyperparameters(name)

        return self.system

    def _optimize_weights(self, component_name: str) -> None:
        """
        Run PPO on the local LLM that backs `component_name` and
        drop the resulting LoRA adapter inside
        {args.output_dir}/ppo/{component_name}

        Then update the component to use the new adapter.
        """
        logger.info(f"[PPO] optimising weights for component «{component_name}»")
        component = self.system.components[component_name]

        # -------- paths ------------
        global OPTIM_CNT
        run_name = f"ppo{OPTIM_CNT}-{component_name}"
        output_dir = os.path.join(self.args.output_dir, run_name)
        os.makedirs(output_dir, exist_ok=True)

        # -------- reward dataset → prompts for PPO ------------------------
        ds_inputs = self.reward_dataset[component_name]['input']
        random.shuffle(ds_inputs)
        dataset = Dataset.from_list(ds_inputs[:self.args.per_component_train_size])

        # -------- call PPO trainer -------
        train_ppo(
            args=self.args,
            dataset=dataset,
            system=self.system,
            component_name=component_name,
            reward_model=self.reward_model,
            lora_cfg=component.config.lora_cfg,
            run_name=run_name,
            output_dir=output_dir
        )
        OPTIM_CNT += 1

        # -------- update the system to use the new adapter -----------------------------------
        component.update(Path(output_dir))

    def _evaluate_hyperparameter(self, component_name, hyperparameter: dict, trainset: List, metric) -> float:
        """
        Evaluate the candidate prompt using the provided metric on the trainset.
        The `metric` expects (example, pred) or something similar.
        """
        cur_config = {component_name: {'variable': hyperparameter}}
        component = self.system.components[component_name]

        total_score = 0.0
        assert isinstance(component.variable, dict), "Module variable should be a dict."

        def process_single_example(component, example):
            """Process a single example through the component"""
            pred = component(**example.inputs())
            pred = Prediction(**pred)
            return pred

        # Prepare the task arguments
        task_args = [(component, example) for example in trainset]

        # Run in parallel

        with self.system.context(cur_config):
            predictions = run_parallel_tasks(
                task_func=process_single_example,
                task_args=task_args,
                max_workers=self.args.max_sample_workers,  # Adjust based on your system capabilities
                task_desc=f"Evaluate hp {hyperparameter}"
            )

        # Filter out None results (from errors) if needed
        predictions = [pred for pred in predictions if pred is not None]

        avg_score = metric(trainset, predictions)
        return avg_score

    def _optimize_hyperparameters(self, component_name):
        """
        Implement grid search on hyperparameters
        """

        component = self.system.components[component_name]
        old_variable = copy.deepcopy(component.variable)

        best_score = float('-inf')
        best_params = None

        param_keys = list(component.variable_search_space.keys())
        param_values = list(component.variable_search_space.values())
        param_combinations = list(itertools.product(*param_values))

        def metric_from_rm_or_global_metric(example, pred, trace=None):
            """
            Calculate metric for single instance or average for multiple instances.

            Args:
                example: Single example (dict/object) or list of examples
                pred: Single prediction (dict/object) or list of predictions
                trace: Optional trace information
                return_all: If True and input is list, return all scores instead of average

            Returns:
                float or list: Single score, average score, or list of all scores
            """

            # Ensure both are lists for uniform processing
            is_single = not (isinstance(example, list) or isinstance(pred, list))

            examples = [example] if not isinstance(example, list) else example
            preds = [pred] if not isinstance(pred, list) else pred

            # Validate same length
            if len(examples) == 1 and len(preds) > 1:
                examples = examples * len(preds)
            elif len(preds) == 1 and len(examples) > 1:
                preds = preds * len(examples)
            elif len(examples) != len(preds):
                raise ValueError(f"Length mismatch: {len(examples)} examples vs {len(preds)} predictions")

            # Check if using system evaluation
            use_system = all(
                all(field in p for field in self.system.final_output_fields)
                for p in preds
            )

            if use_system:
                scores = self.system.evaluate_multiple(examples, preds)
            else:
                batch_pool = [{**{key: getattr(ex, key) for key in component.input_fields},
                              **{key: getattr(pr, key) for key in component.output_fields}} for ex, pr in zip(examples, preds)]

                # Evaluate
                scores = self.reward_model.batch_evaluate(component_name, batch_pool, sigmoid=True)
                logger.info(f'Reward values from reward model (batch): {scores}')

            return scores[0] if is_single else sum(scores) / len(scores)

        trainset_per_component = [
            Example(**example['input']).with_inputs(*component.input_fields) \
            for example in self.reward_dataset[component_name]][:self.args.per_component_train_size]

        for i, param_combo in enumerate(param_combinations):
            params = dict(zip(param_keys, param_combo))
            avg_score = self._evaluate_hyperparameter(
                component_name,
                params,
                trainset_per_component,
                metric_from_rm_or_global_metric
            )

            if avg_score > best_score:
                best_score = avg_score
                best_params = params

            logger.info(
                f"Grid {i+1}/{len(param_combinations)} avg={avg_score:.4f} best={best_score:.4f}"
            )

            logger.info(f"Tried {i+1}/{len(param_combinations)} combinations. Current best: {best_score:.4f}")

        logger.info('FINISH PARAMETER SEARCH')
        logger.info(f'best score: {best_score:.4f}')
        logger.info(f'best parameters: {best_params}')
        logger.info(f"Old variable: {old_variable}")
        self.system.components[component_name].update(best_params)

    def _optimize_hyperparameters_full(self, component_name):
        """
        Implement grid search on hyperparameters
        """

        component = self.system.components[component_name]
        old_variable = copy.deepcopy(component.variable)

        best_score = float('-inf')
        best_params = None

        param_keys = list(component.variable_search_space.keys())
        param_values = list(component.variable_search_space.values())
        param_combinations = list(itertools.product(*param_values))

        for i, param_combo in enumerate(param_combinations):
            params = dict(zip(param_keys, param_combo))
            cur_config = {component_name: {'variable': params}}

            logger.info(f'{cur_config=}')
            with self.system.context(cur_config):
                scores = self.system.evaluate_multiple(self.hyper_param_train_dataset)
                avg_score = sum(scores) / len(scores)

            if avg_score > best_score:
                best_score = avg_score
                best_params = params

            logger.info(
                f"Grid {i+1}/{len(param_combinations)} avg={avg_score:.4f} best={best_score:.4f}"
            )

            logger.info(f"Tried {i+1}/{len(param_combinations)} combinations. Current best: {best_score:.4f}")

        logger.info('FINISH PARAMETER SEARCH')
        logger.info(f'best score: {best_score:.4f}')
        logger.info(f'best parameters: {best_params}')
        logger.info(f"Old variable: {old_variable}")
        self.system.components[component_name].update(best_params)

    def _optimize_prompts(self, component_name):
        """
        Optimize the prompts of the components in the system.
        """

        component = self.system.components[component_name]
        old_variable = copy.deepcopy(component.variable)

        def metric_from_rm_or_global_metric(example, pred, trace=None):
            """
            Calculate metric for single instance or average for multiple instances.

            Args:
                example: Single example (dict/object) or list of examples
                pred: Single prediction (dict/object) or list of predictions
                trace: Optional trace information
                return_all: If True and input is list, return all scores instead of average

            Returns:
                float or list: Single score, average score, or list of all scores
            """

            # Ensure both are lists for uniform processing
            is_single = not (isinstance(example, list) or isinstance(pred, list))

            examples = [example] if not isinstance(example, list) else example
            preds = [pred] if not isinstance(pred, list) else pred

            # Validate same length
            if len(examples) == 1 and len(preds) > 1:
                examples = examples * len(preds)
            elif len(preds) == 1 and len(examples) > 1:
                preds = preds * len(examples)
            elif len(examples) != len(preds):
                raise ValueError(f"Length mismatch: {len(examples)} examples vs {len(preds)} predictions")

            # Check if using system evaluation
            use_glob = all(
                all(field in p for field in self.system.final_output_fields)
                for p in preds
            )

            if use_glob:
                scores = self.system.evaluate_multiple(examples, preds)
            else:
                batch_pool = [{**{key: getattr(ex, key) for key in component.input_fields},
                              **{key: getattr(pr, key) for key in component.output_fields}} for ex, pr in zip(examples, preds)]

                # Evaluate
                scores = self.reward_model.batch_evaluate(component_name, batch_pool, sigmoid=True)
                logger.info(f'Reward values from reward model (batch): {scores}')

            return scores[0] if is_single else sum(scores) / len(scores)

        trainset_per_component = [
                Example(**example['input']).with_inputs(*component.input_fields) \
                for example in self.reward_dataset[component_name]
            ][:self.args.per_component_train_size]

        if self.args.prompt_optimizer == "opro":
            logger.info(f"Running OPRO for component {component_name} ...")

            # Construct the OPRO instance
            opro_optimizer = OPRO(
                llm_model=self.args.opro_llm_model,
                temperature=self.args.opro_temperature,
                max_new_tokens=self.args.opro_max_new_tokens,
                metric=metric_from_rm_or_global_metric,
                num_prompt_candidates=self.args.num_prompt_candidates,
                max_sample_workers=self.args.max_workers,
                meta_prompt_preamble=self.args.opro_meta_prompt_preamble_template.format(
                    component_description=component.description
                ),
            )
            # Now variable only contains the system prompt
            initial_prompt_str = component.variable

            new_variable, prompt_score_pairs = opro_optimizer.compile(
                component=component,
                initial_prompt=initial_prompt_str,
                trainset=trainset_per_component
            )

            # Debug prints
            logger.info(f"All prompt score pairs: {json.dumps(prompt_score_pairs, indent=2)}")
            logger.info(f"Old prompt: {old_variable}")
            logger.info(f"New prompt from OPRO: {new_variable}")

            # Update component with the new prompt
            component.update(new_variable)

        elif self.args.prompt_optimizer == "mipro":
            from dspy.teleprompt import MIPROv2

            logger.info(f'train size: {len(trainset_per_component)}')
            tp = MIPROv2(
                metric=metric_from_rm_or_global_metric,
                auto=self.args.auto,
                verbose=self.args.verbose,
                num_candidates=self.args.num_prompt_candidates,
                num_threads=self.args.num_threads
            )
            old_signature_cls = component.signature_cls.with_instructions(component.variable)

            new_signature = tp.compile(
                dspy.Predict(old_signature_cls),
                trainset=trainset_per_component,
                requires_permission_to_run=self.args.requires_permission_to_run
            ).signature

            new_variable = new_signature.instructions

        elif self.args.prompt_optimizer == "copro":
            from dspy.teleprompt import COPRO

            eval_kwargs = dict(num_threads=self.args.num_threads, display_progress=True)
            tp = COPRO(
                metric=metric_from_rm_or_global_metric,
                breadth=self.args.num_prompt_candidates,
                depth=self.args.copro_depth,
                verbose=self.args.verbose
            )
            old_signature_cls = component.signature_cls.with_instructions(component.variable)
            new_signature = tp.compile(
                dspy.Predict(old_signature_cls),
                trainset=trainset_per_component,
                eval_kwargs=eval_kwargs
            ).signature
            new_variable = new_signature.instructions
        else:
            raise ValueError(f"Invalid prompt optimizer: {self.args.prompt_optimizer}")

        if self.args.verbose:
            logger.info(f"Old prompt for component '{component_name}': {old_variable}")
            logger.info(f"Optimized prompt for component '{component_name}': {new_variable}")

        self.system.components[component_name].update(new_variable)

