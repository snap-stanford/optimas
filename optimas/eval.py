import numpy as np
import json
import os
from tqdm import tqdm
from typing import Dict, List
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from optimas.arch.system import CompoundAISystem
from optimas.reward_model import RewardModel
from optimas.utils.logger import setup_logger
from optimas.wrappers.example import Example


logger = setup_logger(__name__)


def eval_system(
    system: CompoundAISystem,
    testset: List[Example],
    num_repeat: int = 1
) -> List:
    """
    Evaluate a compound agent system on a test set.

    Args:
        system: The compound agent system to evaluate
        testset: List of test examples
        num_repeat: Number of times to repeat evaluation

    Returns:
        Tuple of (predictions, metrics)
    """
    metrics = {}
    for i in range(num_repeat):
        results = system.evaluate_multiple(testset, return_pred=True)
        original_results_num = len(results)
        results = [result for result in results if result is not None]
        filtered_results_num = len(results)
        if original_results_num != filtered_results_num:
            logger.info(f"Original results num: {original_results_num}")
            logger.info(f"Filtered results num: {filtered_results_num}")
            logger.info(f"Filtered {original_results_num - filtered_results_num} NONE results")
            none_idx = [i for i, result in enumerate(results) if result is None]
        scores = []
        preds = []
        for result in results:
            if isinstance(result, tuple):
                scores.append(result[0])
                preds.append(result[1])
            else:
                scores.append(result)
                preds.append(None)
        
        metrics.update({
            f"trial_{i}": np.mean(scores)
        })

    metrics.update({
        "mean": np.mean([metrics[f"trial_{i}"] for i in range(num_repeat)]),
        "std": np.std([metrics[f"trial_{i}"] for i in range(num_repeat)])
    })

    return metrics, preds
