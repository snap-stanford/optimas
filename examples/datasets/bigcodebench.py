import random
from datasets import load_dataset
from typing import List, Tuple

from optimas.wrappers.example import Example


def dataset_engine(train_size: int = 1000, seed: int = 42, **kwargs) -> Tuple[List[Example], List[Example], List[Example]]:
    """
    Load and prepare the BigCodeBench dataset as train/val/test splits.

    Args:
        train_size (int): Number of examples to use for training. Defaults to 1000.
        seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[List[Example], List[Example], List[Example]]: trainset, valset, testset
    """
    raw_data = load_dataset("bigcode/bigcodebench", split="v0.1.3")

    # Assign split tags
    splits = ['train'] * train_size + ['test'] * (len(raw_data) - train_size)
    random.seed(seed)
    random.shuffle(splits)

    # Build Examples
    examples = [
        Example(
            question=ex['instruct_prompt'],
            code=ex['code_prompt'] + ex['canonical_solution'],
            unit_tests=ex['test'],
            task_id=ex['task_id'],
            entry_point='task_func'
        ).with_inputs("question")
        for ex in raw_data
    ]

    # Split into sets
    train_val = [ex for ex, tag in zip(examples, splits) if tag == 'train']
    testset = [ex for ex, tag in zip(examples, splits) if tag == 'test']

    split_idx = int(len(train_val) * 0.95) if len(train_val) >= 20 else len(train_val) - 1
    trainset = train_val[:split_idx]
    valset = train_val[split_idx:]

    return trainset, valset, testset


if __name__ == "__main__":
    trainset, valset, testset = dataset_engine()
    print(f"Loaded {len(trainset)} training, {len(valset)} validation, and {len(testset)} test examples.")
    print(trainset[0])
    # => 
    # Example({
    # 'question': 'Calculates the average of the sums of absolute differences between each pair of consecutive numbers ...', 
    # 'unit_tests': "import unittest\nfrom unittest.mock import patch\nfrom random import seed, shuffle\nimport itertools\nclass TestCases(unittest.TestCase):\n    ...", 
    # 'task_id': 'BigCodeBench/0', 
    # 'entry_point': 'task_func'
    # }) (input_keys={'question'})