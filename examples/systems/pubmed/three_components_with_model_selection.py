import os
import os.path as osp
import json
import datetime
import argparse
import random
import re
from tqdm import tqdm
from typing import Dict, List, Any
from dotenv import load_dotenv
import random
from optimas.arch.system import CompoundAISystem
from optimas.arch.base import BaseComponent
from optimas.utils.api import get_llm_output
import torch
import torch.nn.functional as F
import numpy as np
import re

# List of available models
MODELS_LIST = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo",
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-20250219",
]

class ModelSelectorModule(BaseComponent):
    """
    Module that selects the most appropriate model for a given task.
    Uses a reward model to score and rank candidate models when available.
    """
    def __init__(
        self, 
        task_type="context_analyst", 
        variable_search_space={"model_selection": MODELS_LIST}, 
        initial_variable="gpt-4o-mini", 
        models_list=MODELS_LIST, 
        model="gpt-4o-mini", 
        force_model=None, 
        max_tokens=1024, 
        temperature=0.0
    ):
        self.task_type = task_type
        super().__init__(
            description=f"Model Selector chooses the most appropriate model for the {task_type} task.",
            input_fields=["context", "question", "summary"] if task_type == "problem_solver" else ["context", "question"],
            output_fields=[f"{task_type}_model"],
            variable={"model_selection": initial_variable},
            variable_search_space=variable_search_space
        )
        self.force_model=force_model
        
    def forward(self, **inputs):
        if self.force_model:
            print(f"{self.task_type}_model using force model {self.force_model}")
            return {f"{self.task_type}_model": self.force_model} 
        return {f"{self.task_type}_model": self.variable["model_selection"]}

# Format prompt for yes/no/maybe answers
FORMAT_PROMPT_YESNO = '''Always conclude the last line of your response should be of the following format: 'Answer: $VALUE' (without quotes) where VALUE is either 'yes' or 'no' or 'maybe'.'''

# System prompt
SYS_SINGLE_SOL_PROMPT = '''You are a scientist.'''

class ContextAnalystModule(BaseComponent):
    """
    Module that extracts and summarizes key information from a given context
    to address a question.
    """
    
    def __init__(self, model="gpt-4o-mini", max_tokens=4096, temperature=0.0):
        """
        Initialize the Context Analyst Module.
        
        Args:
            model (str): Default model to use (will be overridden by selected model)
            max_tokens (int): Maximum tokens for generation
            temperature (float): Temperature for generation
        """
        instruction_prompt = "You are supposed to summarize the key information from the given context to answer the provided question."
        super().__init__(
            description="Context Analyst extracts and summarizes key information from a given context.",
            input_fields=["context", "question", "context_analyst_model"],
            output_fields=["summary"],
            variable=instruction_prompt,
            config={
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )
        
    def forward(self, **inputs):
        """
        Process the context and extract key information.
        
        Args:
            context (str): The medical context to analyze
            question (str): The question to answer
            context_analyst_model (str): The model selected for this task
            
        Returns:
            dict: Dictionary with summary of the context
        """
        context = inputs.get("context")
        question = inputs.get("question")
        model = inputs.get("context_analyst_model", self.config.model)
        
        if not context:
            raise ValueError("Context is required")
        if not question:
            raise ValueError("Question is required")
        
        # Format the prompt
        user_prompt = f'''{self.variable}

Here is the given context:
"{context}"

Problem:
"{question}"

Please summarize the relevant information from the context related to the question.'''
        
        # Call the LLM with the selected model
        response = get_llm_output(
            message=user_prompt,
            model=model,  
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system_prompt=SYS_SINGLE_SOL_PROMPT
        )
        
        return {"summary": response}
        

class ProblemSolverModule(BaseComponent):
    """
    Module that interprets the Context Analyst's summary and determines
    the correct yes/no/maybe answer based on evidence.
    """
    
    def __init__(self, model="gpt-4o-mini", max_tokens=4096, temperature=0.0):
        """
        Initialize the Problem Solver Module.
        
        Args:
            model (str): Default model to use (will be overridden by selected model)
            max_tokens (int): Maximum tokens for generation
            temperature (float): Temperature for generation
        """
        instruction_prompt = "You are supposed to provide a solution to a given problem based on the provided summary."
        super().__init__(
            description="Problem Solver determines the correct yes/no/maybe answer based on the provided summary.",
            input_fields=["question", "summary", "problem_solver_model"],
            output_fields=["answer"],
            variable=instruction_prompt,
            config={
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )
        
    def forward(self, **inputs):
        """
        Process the analyst's summary and determine the answer.
        
        Args:
            question (str): The medical question to answer
            summary (str): The summary provided by the Context Analyst
            problem_solver_model (str): The model selected for this task
            
        Returns:
            dict: Dictionary with the final answer
        """
        question = inputs.get("question")
        summary = inputs.get("summary")
        model = inputs.get("problem_solver_model", self.config.model)
        
        if not question:
            raise ValueError("Question is required")
        if not summary:
            raise ValueError("Summary is required")
        
        # Format the prompt
        user_prompt = f'''{self.variable}

Problem:
"{question}"

Here is a summary of relevant information:
"{summary}"

Please provide yes, no or maybe to the given problem. {FORMAT_PROMPT_YESNO}'''
        
        # Call the LLM with the selected model
        response = get_llm_output(
            message=user_prompt,
            model=model,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system_prompt=SYS_SINGLE_SOL_PROMPT
        )
        
        return {"answer": response}
        

def extract_answer_yesno(input_string):
    """Extract yes/no/maybe answer from model response."""
    pattern = r"(?i)\s*(yes|no|maybe|Yes|No|Maybe)"
    match = re.search(pattern, input_string)
    extracted_answer = match.group(1).lower() if match else input_string
    return extracted_answer
    

def pubmed_eval_func(answer, groundtruth):
    """
    Evaluation function for PubMedQA that uses the extract_answer_yesno function
    to maintain consistency with your existing code.
    
    Args:
        answer (str): The model's answer text
        groundtruth (str): The correct answer (yes/no/maybe)
        
    Returns:
        float: 1.0 if correct, 0.0 if incorrect
    """
    import re
    
    # Extract the answer using your existing function
    predicted = extract_answer_yesno(answer)
    
    # Normalize groundtruth
    groundtruth = groundtruth.lower().strip()
    
    # Simple exact match scoring
    if predicted.lower() == groundtruth.lower():
        return 1.0
    else:
        return 0.0


def system_engine(force_context_model=None, force_solver_model=None, *args, **kwargs):
    """
    Create and configure a PubMed system with two-stage model selection.
    
    Args:
        force_context_model (str, optional): Force a specific model for context analyst
        force_solver_model (str, optional): Force a specific model for problem solver
        *args: Positional arguments passed to CompoundAISystem
        **kwargs: Keyword arguments
        
    Returns:
        CompoundAISystem: The configured system
    """
    selector_model = kwargs.pop("selector_model", "gpt-4o-mini")
    temperature = kwargs.pop("temperature", 0.0)
    eval_func = kwargs.pop("eval_func", pubmed_eval_func)
    max_tokens = kwargs.pop("max_tokens", 4096)

    # Initialize modules
    context_model_selector = ModelSelectorModule(
        task_type="context_analyst",
        models_list=MODELS_LIST,
        model=selector_model,
        temperature=temperature,
        max_tokens=1024
    )
    solver_model_selector = ModelSelectorModule(
        task_type="problem_solver",
        models_list=MODELS_LIST,
        model=selector_model,
        temperature=temperature,
        max_tokens=1024
    )
    context_analyst = ContextAnalystModule(
        model="gpt-4o-mini",
        temperature=temperature,
        max_tokens=max_tokens
    )
    problem_solver = ProblemSolverModule(
        model="gpt-4o-mini",
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Apply forced models if provided
    if force_context_model:
        context_model_selector.config.force_model = force_context_model
    if force_solver_model:
        solver_model_selector.config.force_model = force_solver_model

    # Construct the system declaratively
    system = CompoundAISystem(
        components={
            "context_model_selector": context_model_selector,
            "context_analyst": context_analyst,
            "solver_model_selector": solver_model_selector,
            "problem_solver": problem_solver,
        },
        final_output_fields=["answer"],
        ground_fields=["groundtruth"],
        eval_func=eval_func,
        *args,
        **kwargs,
    )

    return system


if __name__ == "__main__":
    # Load environment variables 
    dotenv_path = osp.expanduser('.env')
    load_dotenv(dotenv_path)
    
    # Create the system
    system = system_engine(force_context_model="gpt-4o-mini", force_solver_model="claude-3-haiku-20240307")
    
    # Example PubMed question
    context = "Programmed cell death (PCD) is the regulated death of cells within an organism. The lace plant (Aponogeton madagascariensis) produces perforations in its leaves through PCD. The following paper elucidates the role of mitochondrial dynamics during developmentally regulated PCD in vivo in A. madagascariensis. This treatment resulted in lace plant leaves with a significantly lower number of perforations compared to controls, and that displayed mitochondrial dynamics similar to that of non-PCD cells."
    question = "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?"
    
    # Run the system
    result = system(context=context, question=question)
    
    # Extract and print the answer
    answer = extract_answer_yesno(result.answer)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
