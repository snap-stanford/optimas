import os
import torch
import os.path as osp
from typing import Any
from peft import PeftModel, get_peft_model, PeftConfig
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from optimas.utils.logger import setup_logger

logger = setup_logger(__name__)


def infer_output_dim(state_dict: dict) -> int:
    """Infer output dimension from classifier-related keys."""
    for k, v in state_dict.items():
        if 'score.weight' in k or 'classifier.weight' in k:
            return v.shape[0]
    raise ValueError("Unable to infer output dimension from state_dict keys.")


def load_model_and_tokenizer(
    model_name: str,
    bnb_config: Any = None,
    peft_config: PeftConfig = None,
    is_trainable: bool = True,
    device: str = "cuda",
    state_dict_path: str = None,
    **kwargs
):
    """
    Load a sequence classification model and tokenizer with optional LoRA and quantization.

    Args:
        model_name (str): HuggingFace model ID or local path.
        bnb_config (Optional): Quantization config (e.g., BitsAndBytesConfig).
        peft_config (Optional): PEFT LoRA config.
        is_trainable (bool): Whether model is for training or inference.
        device (str): Device spec string.
        state_dict_path (Optional[str]): Path to .pth file with adapter weights.
        **kwargs: Additional kwargs passed to model.from_pretrained.

    Returns:
        Tuple[model, tokenizer]
    """
    state_dict = None
    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path, weights_only=True)
        kwargs.setdefault("num_labels", infer_output_dim(state_dict))
        logger.info(f"Inferred output dimension: {kwargs['num_labels']}")

    if is_trainable:
        device = f"cuda:{os.getenv('LOCAL_RANK', 0)}"
        logger.info("Loading model in **training** mode")
    else:
        logger.info("Loading model in **inference** mode")

    if bnb_config is not None:
        kwargs["quantization_config"] = bnb_config

    logger.info(f"Loading model: {model_name}")
    if osp.exists(model_name):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map={"": device},
            local_files_only=True,
            use_cache=False,
            **kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_cache=False,
            device_map={"": device},
            **kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if peft_config is not None:
        model = get_peft_model(model, peft_config)

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    if state_dict is not None:
        logger.info(f"Loading weights from: {state_dict_path}")
        assert set(state_dict.keys()).issubset(model.state_dict().keys())
        model.load_state_dict(state_dict, strict=False)

    logger.info(f"is_trainable={is_trainable} | device={device} | padding_side={tokenizer.padding_side}")
    return model, tokenizer

