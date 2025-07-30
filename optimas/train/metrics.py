import warnings
from typing import Dict

import numpy as np
import torch
from transformers.trainer_utils import EvalPrediction


def compute_accuracy_and_margin(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute accuracy and average margin between preferred and rejected reward scores.

    Args:
        eval_pred (EvalPrediction): A tuple containing model predictions and labels. 
            Predictions should be a 2D array with shape (batch_size, 2), where the
            first column corresponds to the chosen reward and the second to the rejected one.
            Labels should be a 1D array of 0s and 1s indicating the correct option.

    Returns:
        Dict[str, float]: A dictionary with:
            - "margin": average difference between chosen and rejected scores.
            - "accuracy": fraction of instances where the preferred score is greater,
              counting ties as 0.5.
    """
    predictions, labels = eval_pred
    assert predictions.ndim == 2, "Predictions must have shape (batch_size, 2)"

    labels = np.array(labels, dtype=int)
    margins = predictions[np.arange(len(predictions)), labels] - predictions[np.arange(len(predictions)), 1 - labels]
    accuracy = (margins > 0).mean().item() + 0.5 * (margins == 0).mean().item()
    margin = margins.mean().item()

    # Warn and filter if any predictions are equal
    equal_mask = predictions[:, 0] == predictions[:, 1]
    if equal_mask.any():
        count_equal = int(equal_mask.sum())
        warnings.warn(
            f"{count_equal} out of {len(predictions)} predictions are equal for both options. "
            "These are ignored when computing accuracy.",
            UserWarning,
        )

    return {
        "margin": margin,
        "accuracy": accuracy,
    }
