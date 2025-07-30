import copy
import re
import random
from typing import Callable, List, Optional, Union
from tqdm import tqdm

from optimas.utils.api import get_llm_output
from optimas.arch.base import BaseComponent
from optimas.utils.logger import setup_logger
from optimas.utils.parallel import run_parallel_tasks
from optimas.wrappers.prediction import Prediction

logger = setup_logger(__name__)

# Tips inspired by DSPy (https://dspy.ai/):
TIPS = {
    "creative": "Don't be afraid to be creative when creating the new instruction!",
    "simple": "Keep the instruction clear and concise.",
    "description": "Make sure your instruction is very informative and descriptive.",
    "specific": "The instruction should include specific details such as numbers or conditions.",
}


class OPRO:
    """
    OPRO (Large Language Models as Optimizers):
    Iteratively propose improved prompts using an LLM.

    Parameters:
    -----------
    metric: Callable
        A function metric(example, prediction) -> float that
        returns a scalar reward for a given prompt.
    llm_model: str
        Name of the model to call via get_llm_output (e.g. "gpt-4o").
    num_prompt_candidates: int
        How many times to iteratively improve the prompt.
    temperature: float
        Temperature for the optimizer LLM calls.
    max_new_tokens: int
        Maximum tokens for the optimizer LLM calls.
    meta_prompt_preamble: str
        A string describing the overall task to the optimizer LLM.
    """
    def __init__(
        self,
        metric: Callable,
        llm_model: str = "gpt-4o",
        num_prompt_candidates: int = 5,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        meta_prompt_preamble: Optional[str] = None,
        max_sample_workers=4,
    ):
        self.metric = metric
        self.llm_model = llm_model
        self.num_prompt_candidates = num_prompt_candidates
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_sample_workers = max_sample_workers

        self.meta_prompt_preamble = meta_prompt_preamble or (
            "You are an AI specialized in improving prompts for a certain component.\n"
            "We have a list of (prompt, score) pairs tried so far.\n"
            "Please propose an improved new prompt that may yield a higher score.\n"
        )

    def compile(
        self,
        component: BaseComponent,
        initial_prompt: Union[str, "dspy.Signature"],
        trainset: List,
        include_initial_prompt: bool = False,
        **kwargs
    ) -> Union[str, "dspy.Signature"]:
        """
        Run the OPRO iterations to improve `initial_prompt` on `trainset`.
        Returns the best prompt found and full prompt-score history.
        """
        prompt_history = []

        if include_initial_prompt:
            initial_score = self._evaluate_prompt(component, initial_prompt, trainset)
            logger.info(f"Initial prompt: {initial_prompt}, Score: {initial_score}")
            prompt_history.append((initial_prompt, initial_score))

        for iteration in range(self.num_prompt_candidates):
            meta_prompt = self._build_meta_prompt(prompt_history)
            logger.info(f'{meta_prompt=}')

            llm_response = get_llm_output(
                message=meta_prompt,
                model=self.llm_model,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
            )
            new_prompt = self._parse_llm_solution(llm_response)
            logger.info(f"Iteration {iteration + 1}: Proposed new candidate: {new_prompt}")

            score = self._evaluate_prompt(component, new_prompt, trainset)
            prompt_history.append((new_prompt, score))

        # Sort by score descending and return the best one
        prompt_history.sort(key=lambda x: x[1], reverse=True)
        return prompt_history[0][0], prompt_history

    def _build_meta_prompt(self, prompt_history: List[tuple]) -> str:
        """Build the meta prompt to give context and feedback to the LLM optimizer."""
        history = sorted(prompt_history, key=lambda x: x[1])
        history_str = ""
        for i, (prompt, score) in enumerate(history):
            history_str += f"Prompt #{i+1}:\n{prompt}\nScore: {round(score, 3)}\n\n"

        tip = TIPS.get(random.choice(list(TIPS.keys())), "")
        return (
            f"{self.meta_prompt_preamble}\n"
            f"Below are the previous prompt attempts and their scores.\n"
            f"The prompts are arranged in ascending order based on their scores.\n\n"
            f"{history_str}"
            f"Observe the pattern carefully. Now propose a new improved prompt. {tip}\n"
            f"Format:\nSolution: <your new prompt>\n"
        )

    def _parse_llm_solution(self, llm_response: str) -> Optional[str]:
        """Extract the prompt text from the LLM's output."""
        match = re.search(r"Solution:\s*(.*)", llm_response, re.IGNORECASE)
        return match.group(1).strip() if match else llm_response.strip()

    def _evaluate_prompt(self, component, candidate_prompt: str, trainset: List) -> float:
        """Evaluate a prompt by running the component and scoring predictions."""
        assert isinstance(component.variable, str), "Component variable must be a string prompt."

        def process_single_example(component, example):
            pred = component(**example.inputs())
            return Prediction(**pred)

        task_args = [(component, ex) for ex in trainset]

        with component.context(variable=candidate_prompt):
            predictions = run_parallel_tasks(
                task_func=process_single_example,
                task_args=task_args,
                max_workers=self.max_sample_workers,
                task_desc="Evaluate prompt"
            )

        predictions = [p for p in predictions if p is not None]
        return self.metric(trainset, predictions)