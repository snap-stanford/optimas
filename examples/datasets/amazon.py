import json
import csv
import ast
from pathlib import Path
from typing import List, Dict, Tuple
from optimas.wrappers.example import Example


def dataset_engine(**kwargs) -> Tuple[List[Example], List[Example], List[Example]]:
    """
    Load and merge two session-based recommendation datasets (primary and multilingual),
    each split 80/20, and return combined train/val/test sets.

    Returns:
        Tuple[List[Example], List[Example], List[Example]]: train, val, and test examples
    """
    data_root = Path("examples/data")
    primary_fp = data_root / "session_based_next_item_selection_dataset.csv"
    multi_fp = data_root / "multilingual_session_based_recommendation_dataset.csv"

    def load_csv(fp: Path) -> List[Dict[str, str]]:
        with fp.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def split_rows(rows: List[Dict[str, str]], frac: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        cutoff = int(len(rows) * frac)
        return rows[:cutoff], rows[cutoff:]

    def make_example(row: Dict[str, str]) -> Example:
        sequence = row["question"].split("Product Sequence:")[-1].strip()
        try:
            choices = ast.literal_eval(row["choices"])
        except Exception:
            choices = json.loads(row["choices"])
        return Example(
            sequence=sequence,
            choices=choices,
            gd_answer=str(int(row["answer"]))
        ).with_inputs("sequence", "choices")

    # Load and split datasets
    train_primary, test_primary = split_rows(load_csv(primary_fp))
    train_multi, test_multi = split_rows(load_csv(multi_fp))

    train_rows = train_primary + train_multi
    test_rows = test_primary + test_multi

    train_val = [make_example(r) for r in train_rows]
    testset = [make_example(r) for r in test_rows]

    # Train/Val split: 85/15
    split = int(len(train_val) * 0.85)
    trainset = train_val[:split]
    valset = train_val[split:]

    return trainset, valset, testset


# Sample usage and debugging
if __name__ == "__main__":
    trainset, valset, testset = dataset_engine()
    print(f"Loaded {len(trainset)} training, {len(valset)} validation, and {len(testset)} test examples.")
    print(trainset[0])
    # => 
    # Example({
    # 'sequence': 'Oral-B iO9 Electric Toothbrush with Revolutionary Magnetic Technology...', 
    # 'choices': [
    #   "Ella's Kitchen First Taste Mixed Case Selection From 4 Month 8 x 70g", 
    #   'Neviti 678382 Carnival Popcorn Box, Blue', 
    #   'Oral-B Advance Power Electric Toothbrush, 1 Toothbrush Head With Indicator Bristles, ...', 
    #   'Arteza 12 White Acrylic Paint Pens, Long Lasting Outdoor Acrylic Paint Markers with Plastic Nib, ...'], 
    # 'gd_answer': '2'}
    # ) (input_keys={'sequence', 'choices'})


