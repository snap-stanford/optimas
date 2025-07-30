import json
import os
import os.path as osp
import random
from typing import Dict, List, Optional, Any, Tuple

from optimas.wrappers.example import Example


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.

    Args:
        file_path (str): Path to the .jsonl file.

    Returns:
        List[Dict[str, Any]]: List of parsed JSON objects.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def dataset_engine(
    train_size: Optional[int] = None,
    data_path: str = "examples/data/",
    train_file: str = "combined_PubMedQA_train.jsonl",
    test_file: str = "combined_PubMedQA_test.jsonl",
    seed: int = 42,
    *args,
    **kwargs
) -> Tuple[List[Example], List[Example], List[Example]]:
    """
    Load and prepare the PubMedQA dataset.

    Args:
        train_size (int, optional): Number of training examples to use. Defaults to full dataset.
        data_path (str): Path to dataset files. Defaults to 'examples/data/'.
        train_file (str): Name of the training file. Defaults to 'combined_PubMedQA_train.jsonl'.
        test_file (str): Name of the test file. Defaults to 'combined_PubMedQA_test.jsonl'.
        seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[List[Example], List[Example], List[Example]]: (trainset, valset, testset)
    """
    random.seed(seed)

    # Load JSONL files
    train_data = load_jsonl(osp.join(data_path, train_file))
    test_data = load_jsonl(osp.join(data_path, test_file))

    # Shuffle and trim train set if necessary
    if train_size is not None and train_size < len(train_data):
        random.shuffle(train_data)
        train_data = train_data[:train_size]

    # Convert training and validation data
    train_val_set = [
        Example(
            question=item["question"],
            context=" ".join(item["context"]) if isinstance(item["context"], list) else item["context"],
            groundtruth=item["groundtruth"]
        ).with_inputs("question", "context")
        for item in train_data
    ]

    # 95% train / 5% val split
    split_idx = int(len(train_val_set) * 0.95)
    trainset = train_val_set[:split_idx]
    valset = train_val_set[split_idx:]

    # Convert test data
    testset = [
        Example(
            question=item["question"],
            context=" ".join(item["context"]) if isinstance(item["context"], list) else item["context"],
            groundtruth=item["groundtruth"]
        ).with_inputs("question", "context")
        for item in test_data
    ]

    return trainset, valset, testset


if __name__ == "__main__":
    trainset, valset, testset = dataset_engine()
    print(f"Loaded {len(trainset)} training, {len(valset)} validation, and {len(testset)} test examples.")
    print(trainset[0])
    # => 
    # Example({
    # 'question': 'Can tailored interventions increase mammography use among HMO women?', 
    # 'context': 'Telephone counseling and tailored print communications have ...', 
    # 'groundtruth': 'yes'}
    # ) (input_keys={'question', 'context'})
