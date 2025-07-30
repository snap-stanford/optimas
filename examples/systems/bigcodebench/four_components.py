from dotenv import load_dotenv
import re
import dspy

from optimas.arch.base import BaseComponent
from optimas.arch.system import CompoundAISystem
from optimas.adapt.dspy import create_component_from_dspy
from examples.metrics.pass_rate import pass_rate
from optimas.utils.api import get_llm_output
from bigcodebench.sanitize import sanitize


def make_raw_chat_prompt(
    task_prompt: str,
    split: str,
    instruction_prefix: str,
) -> str:
    # directly return prompt if it does not have a tokenizer.chat_template

    assert instruction_prefix is not None, "Instruction prefix is required!"

    if split == "complete":
        task_prompt = f"""\
{instruction_prefix}
```
{task_prompt.strip()}
```
"""
    else:
        task_prompt = f"""\
{instruction_prefix}
{task_prompt.strip()}
"""
    return task_prompt



class CodeGenerator(BaseComponent):
    """Generate a runnable Python solution for a given coding problem."""

    def __init__(self, model='claude-3-haiku-20240307', max_tokens=1572, temperature=0.6):
        super().__init__(
            description="Generate Python solution for coding problem.",
            input_fields=["question"],
            output_fields=["initial_code"],
            variable="Provide a self-contained Python solution:",
            config={"model": model, "max_tokens": max_tokens, "temperature": temperature},
        )
    
    def forward(self, **inputs):
        """
        Generate a runnable Python solution for a given coding problem.

        Args:
            question (str): A clear and concise description of the coding problem.
            **inputs: May include model, temperature, max_tokens set by system

        Returns:
            code (str): A complete, executable Python solution that solves the given problem without syntax errors.
        """
        prompt = make_raw_chat_prompt(
            inputs["question"], instruction_prefix=self.variable, split="complete"
        )

        initial_code = get_llm_output(
            message=prompt,
            model=self.config.model,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        if 'entry_point' in inputs:
            entry_point = inputs['entry_point']
        else:
            entry_point = 'task_func'
        initial_code = sanitize(initial_code, entry_point)

        return {"initial_code": initial_code}


class UnitTestGenerator(BaseComponent):
    """Generate a set of Python unit tests to validate the correctness of the generated code."""

    def __init__(self, model='claude-3-haiku-20240307', max_tokens=1572, temperature=0.6):

        UNIT_TEST_PROMPT = '''**Role**: You are a software programmer.
**Task**: As a programmer, you are required to complete the function. Use a Chain-of-Thought approach to break
down the problem, create pseudocode, and then write the code in Python language. Ensure that your code is
efficient, readable, and well-commented.

For example:
**Input Code Snippet**:
```python
from typing import 
def has_close_elements(numbers: List[float], threshold: float) -> bool:
 """
 Check if in given list of numbers, are any two numbers closer to each other than given threshold.
 >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
 False
 >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
 True
 """
 # TODO: Implement the logic to determine if any two numbers are closer than the threshold
 pass
# Add your code here to complete the function
```

**Instructions**:
1. **Understand and Clarify**: Make sure you understand the task.
2. **Algorithm/Method Selection**: Decide on the most efficient way.
3. **Pseudocode Creation**: Write down the steps you will follow in pseudocode.
4. **Code Generation**: Translate your pseudocode into executable Python code.
'''

        super().__init__(
            description="Generate a set of Python unit tests to validate the correctness of the generated code.",
            variable=UNIT_TEST_PROMPT,
            input_fields=["question"],
            output_fields=["additional_unit_tests"],
            config={
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )

    def forward(self, question):

        unit_test_prompt = self.variable + question + "\n\nUNIT TEST CODE:\n"
        unit_tests = get_llm_output(
            message=unit_test_prompt,
            model=self.config.model,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        
        match = re.search(r"```python\s*(.*?)\s*```", unit_tests, re.DOTALL)
        if match:
            unit_tests = match.group(1).strip("\n").strip()
        else:
            unit_tests = ""

        return {"additional_unit_tests": unit_tests}


class Executor(BaseComponent):
    """Execute the generated code against the generated unit tests and return the results."""

    def __init__(self):
        super().__init__(
            description="Execute the generated Python code using the provided unit tests.",
            input_fields=["initial_code", "additional_unit_tests"],
            output_fields=["execution_result"],
        )

    def forward(self, initial_code, additional_unit_tests):
        result = pass_rate(initial_code, additional_unit_tests, entry_point="task_func")
        return {"execution_result": result}


class FinalCodeGenerator(dspy.Signature):
    """Refine code based on test results."""

    question = dspy.InputField(desc="The coding problem.")
    initial_code = dspy.InputField(desc="Initial code solution.")
    additional_unit_tests = dspy.InputField(desc="Unit tests for evaluation.")
    execution_result = dspy.InputField(desc="Test execution results, including errors.")
    code = dspy.OutputField(desc="Improved code solution based on test results.")


def system_engine(*args, **kwargs):
    lm = dspy.LM(
        model='anthropic/claude-3-haiku-20240307',
        max_tokens=1572,
        temperature=0.6
    )
    dspy.settings.configure(lm=lm)

    system = CompoundAISystem(
        components={
            "code_generator": CodeGenerator(),
            "unit_test_generator": UnitTestGenerator(),
            "executor": Executor(),
            "final_code_generator": create_component_from_dspy(FinalCodeGenerator),
        },
        final_output_fields=["code"],
        ground_fields=["unit_tests", "entry_point"],
        eval_func=pass_rate,
        *args, **kwargs
    )

    return system


if __name__ == "__main__":
    import os.path as osp
    from examples.datasets.bigcodebench import dataset_engine

    dotenv_path = osp.expanduser(".env")
    load_dotenv(dotenv_path)

    trainset, valset, testset = dataset_engine()
    system = system_engine(max_workers=32)

    prediction = system(**trainset[0])
    print(prediction.code)