import torch
import torch.nn as nn
from typing import Dict, List
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import os
import hashlib

from transformers import AutoTokenizer, AutoModel
from optimas.utils.template import apply_reward_template
from optimas.arch.base import BaseComponent
from optimas.arch.system import CompoundAISystem
from optimas.utils.context import get_context_from_traj
from optimas.utils.logger import setup_logger

logger = setup_logger(__name__)


class RewardModel(nn.Module):
    """
    A reward model for evaluating compound agent systems.

    This model formats prompts, tokenizes inputs, and assesses agent performance
    based on provided outputs and contextual information.
    """
    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        system: CompoundAISystem,
        batch_size: int = 2  # Add this parameter
    ):
        """
        Initializes the reward model.

        Args:
            model: The underlying model for computing rewards.
            tokenizer: Tokenizer for processing inputs.
            system: The compound agent system.
            batch_size: Size of batches for batch evaluation (default: 32).
        """
        super().__init__()
        # Initialize tokenizer
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.batch_size = batch_size

        self.system = system
        self.component_to_idx = system.optimizable_component_to_idx
        logger.info(f"Module-to-index mapping created: {self.component_to_idx}")

    def process_prompt(self, component: BaseComponent, **kwargs) -> torch.Tensor:
        """
        Tokenize the complete prompt (inputs + outputs).

        Args:
            component (BaseComponent): The component containing input_fields, output_fields, etc.
            inputs: Keyword arguments containing input/output values.

        Returns:
            torch.Tensor: The tokenized prompt as a tensor.
        """
        # Separate the inputs and outputs according to the component
        input_dict = {k: v for k, v in kwargs.items() if k in component.input_fields}
        output_dict = {k: v for k, v in kwargs.items() if k in component.output_fields}

        # Fill in the default prompt template
        text = apply_reward_template(
            input_dict=input_dict,
            output_dict=output_dict,
            desc=component.description
        )
        return text

    def evaluate(self, component_name: str, sigmoid=False, **kwargs) -> torch.Tensor:
        """
        Execute the forward pass and post-processing to compute the reward.

        Args:
            component (BaseComponent): The component to evaluate.
            inputs: Keyword arguments containing input/output values.

        Returns:
            torch.Tensor: Model outputs after post-processing.
        """
        operator = torch.sigmoid if sigmoid else lambda x: x
        component = self.system.components[component_name]

        prompt = self.process_prompt(component, **kwargs)
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model(**model_inputs)

        if output.logits.ndim > 1 and output.logits.shape[1] > 1:
            assert output.logits.numel() == len(self.component_to_idx), "Model output should match the number of components."
            return operator(output.logits[0, self.component_to_idx[component_name]]).item()

        assert output.logits.numel() == 1, "Model output should be a single value."
        return operator(output.logits).item()

    def batch_evaluate(self, component_name: str, batch_pool: list, sigmoid=False) -> list:
        """
        Evaluate multiple instances in batches.

        Args:
            component_name (str): The name of the component to evaluate.
            batch_pool (list): List of dictionaries containing kwargs for each instance.
            sigmoid (bool): Whether to apply sigmoid activation to the output.

        Returns:
            list: List of scores for each instance in the batch_pool.
        """
        operator = torch.sigmoid if sigmoid else lambda x: x
        component = self.system.components[component_name]

        scores = []

        # Process in batches
        for batch_start in range(0, len(batch_pool), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(batch_pool))
            batch = batch_pool[batch_start:batch_end]

            # Process prompts for this batch
            prompts = []
            for kwargs in batch:
                prompt = self.process_prompt(component, **kwargs)
                prompts.append(prompt)

            # Tokenize all prompts in the batch together
            model_inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                output = self.model(**model_inputs)

            # Process outputs based on the model's output shape
            if output.logits.ndim > 1 and output.logits.shape[1] > 1:
                # Multi-component output case
                assert output.logits.shape[1] == len(self.component_to_idx), \
                    "Model output should match the number of components."

                # Extract scores for the specific component for each instance in batch
                component_idx = self.component_to_idx[component_name]
                batch_scores = operator(output.logits[:, component_idx]).tolist()

            else:
                # Single output case
                assert output.logits.shape[1] == 1, "Model output should be a single value."
                batch_scores = operator(output.logits.squeeze(-1)).tolist()

            scores.extend(batch_scores)

        return scores

