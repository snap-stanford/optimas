import os
import torch
from huggingface_hub import HfApi
from collections import OrderedDict
from optimas.utils.logger import setup_logger

logger = setup_logger(__name__)

STATE_DICT_FILE_NAME = "state_dict.pth"

def save_model_and_tokenizer(
    model,
    tokenizer,
    output_dir,
    repo_name=None,
    push_to_hub=False,
    criteria=lambda k: 'score.weight' in k.lower() or 'lora' in k.lower()
):
    """
    Save model and tokenizer to disk and optionally push to Hugging Face Hub.

    Args:
        model: torch.nn.Module with state_dict.
        tokenizer: Hugging Face tokenizer.
        output_dir (str): Directory to save model and tokenizer.
        repo_name (str): Hugging Face Hub repo name (e.g. "username/repo").
        push_to_hub (bool): Whether to push to the Hugging Face Hub.
        criteria (callable): Filter for which parameters to save.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter and save model weights
    state_dict_to_save = {
        k: v for k, v in model.state_dict().items() if criteria(k)
    }
    state_dict_to_save = OrderedDict(sorted(state_dict_to_save.items()))
    torch.save(state_dict_to_save, os.path.join(output_dir, STATE_DICT_FILE_NAME))
    logger.info(f"Saved filtered model state_dict to {output_dir}/{STATE_DICT_FILE_NAME}")

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Saved tokenizer to {output_dir}")

    if push_to_hub:
        if not repo_name:
            logger.error("`repo_name` must be specified when `push_to_hub=True`.")
            return

        try:
            if hasattr(model, "merge_and_unload"):
                merged_model = model.merge_and_unload()
                merged_model.push_to_hub(repo_name, private=True)
            else:
                logger.warning("Model has no `merge_and_unload()` method. Pushing original model.")
                model.push_to_hub(repo_name, private=True)

            tokenizer.push_to_hub(repo_name, private=True)
            logger.info(f"Successfully pushed model and tokenizer to Hub: {repo_name}")

        except Exception as e:
            logger.error(f"Error pushing to Hugging Face Hub: {e}")
