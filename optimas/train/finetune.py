import os
from datasets import DatasetDict
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from optimas.train.reward_trainer import RewardTrainer
from optimas.train.reward_config import RewardConfig
from optimas.train.metrics import compute_accuracy_and_margin
from optimas.utils.logger import setup_logger

logger = setup_logger(__name__)


def run_finetune(
    dataset: DatasetDict,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    training_args: RewardConfig,
    peft_config: LoraConfig = None,
    train_last_layer: bool = False,
    **kwargs
):
    """
    Fine-tune a reward model on a given dataset.

    Args:
        dataset (DatasetDict): Hugging Face dataset with 'train' and optionally 'test' splits.
        model (AutoModelForSequenceClassification): The model to fine-tune.
        tokenizer (AutoTokenizer): Tokenizer used for preprocessing.
        training_args (RewardConfig): Training hyperparameters.
        peft_config (Optional[LoraConfig]): PEFT LoRA configuration.
        train_last_layer (bool): If True, unfreezes only the last model layer for training.
        **kwargs: Additional arguments passed to RewardTrainer.

    Returns:
        RewardTrainer: The trainer instance after training is complete.
    """
    model.train()
    do_eval = dataset.get("test", None) is not None and training_args.eval_strategy != "no"

    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset.get("train", dataset),
        eval_dataset=dataset.get("test") if do_eval else None,
        peft_config=peft_config,
        compute_metrics=compute_accuracy_and_margin,
        **kwargs
    )

    if train_last_layer:
        # Unfreeze only the last layer of the model
        last_param_name, last_param = list(trainer.model.named_parameters())[-1]
        logger.info(f"Unfreezing layer: {last_param_name}")
        last_param.requires_grad = True

    def count_parameters(m):
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        total = sum(p.numel() for p in m.parameters())
        return trainable, total

    trainable, total = count_parameters(trainer.model)

    if os.environ.get("LOCAL_RANK", "0") == "0":
        logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")
        logger.info("--- Reward model train started ---")

    logger.info(f"Reward model train size: {len(trainer.train_dataset)}")
    trainer.train()

    return trainer