import os
import time
import json
import requests
from pathlib import Path
import torch
from typing import Optional, Dict, Any, List
from optimas.utils.logger import setup_logger
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logger = setup_logger(__name__)


def _api(host: str, port: int, path: str) -> str:
    return f"http://{host}:{port}{path}"


def initialize_adapter(
    base_model_path: str,
    adapter_path: str,
    lora_cfg: LoraConfig,
    force_recreate: bool = True,
    use_quantization: bool = True,
    device_map: str = "auto",
    torch_dtype = torch.float16,
    return_peft_model: bool = False,
    **model_kwargs: Dict[str, Any]
) -> bool:
    """
    Initialize and save a LoRA adapter from a base model.
    
    Args:
        base_model_path: Path to the base model
        adapter_path: Path where the adapter should be saved
        lora_cfg: LoRA configuration
        force_recreate: If True, recreate adapter even if it exists
        use_quantization: Whether to use 4-bit quantization
        device_map: Device mapping for model loading
        torch_dtype: Torch data type for the model
        return_peft_model: If True, return the PEFT model instead of None
        
    Returns:
        bool: True if adapter was successfully initialized, False otherwise
    """
    if not os.path.exists(adapter_path):
        os.makedirs(adapter_path, exist_ok=True)
    
    # Check if adapter already exists
    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if adapter_config_path.exists() and not force_recreate:
        logger.info(f"Adapter already exists at {adapter_path}. Skipping initialization.")
        return None
    
    logger.info(f"Initializing adapter at {adapter_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model loading arguments
    model_kwargs.update({
        "torch_dtype": torch_dtype,
        "device_map": device_map,
    })
    
    # Add quantization if requested
    if use_quantization:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True
        )
        model_kwargs["quantization_config"] = quant_cfg
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        **model_kwargs
    )
    
    # Prepare model for k-bit training if using quantization
    if use_quantization:
        base_model = prepare_model_for_kbit_training(base_model)
    
    # Apply LoRA
    peft_model = get_peft_model(base_model, lora_cfg)
    
    # Save the adapter (this saves only the LoRA weights, not the full model)
    peft_model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    
    # Clean up GPU memory
    del base_model
    if not return_peft_model:
        del peft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return peft_model if return_peft_model else None
        
        
def list_vllm_models(host: str = "localhost", port: int = 8001) -> List[str]:
    """List models currently served by the vLLM daemon (including LoRAs)."""
    try:
        r = requests.get(_api(host, port, "/v1/models"), timeout=30)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]
    except Exception as e:
        logger.error(f"[vLLM] Failed to list models: {e}")
        return []


def unload_lora_adapter(
    lora_name: str, host: str = "localhost", port: int = 8001, silent: bool = False, timeout: int = 120
) -> None:
    """Unload a LoRA adapter from vLLM server. Ignores 404 errors (not loaded)."""
    r = requests.post(
        _api(host, port, "/v1/unload_lora_adapter"),
        json={"lora_name": lora_name},
        timeout=timeout,
    )
    if r.ok:
        if not silent:
            logger.info(f"[vLLM] Unloaded LoRA: «{lora_name}»")
    elif r.status_code != 404:
        raise RuntimeError(f"Could not unload {lora_name}: {r.text}")


def load_lora_adapter(
    lora_name: str,
    lora_path: str,
    host: str = "localhost",
    port: int = 8001,
    retries: int = 3,
    timeout: int = 300,
) -> None:
    """
    Load a LoRA adapter into vLLM, ensuring only one copy is active.
    Raises RuntimeError if loading fails after retries.
    """
    if lora_name in list_vllm_models(host, port):
        unload_lora_adapter(lora_name, host, port, silent=True)

    payload = {"lora_name": lora_name, "lora_path": str(lora_path)}
    for i in range(retries):
        r = requests.post(
            _api(host, port, "/v1/load_lora_adapter"),
            json=payload,
            timeout=timeout,
        )
        if r.ok:
            logger.info(f"[vLLM] Loaded LoRA: «{lora_name}» from {lora_path}")
            return
        logger.warning(f"[vLLM] Load attempt {i + 1}/{retries} failed: {r.text}")
        time.sleep(2)

    raise RuntimeError(f"Failed to load LoRA {lora_name} after {retries} attempts")


def get_adapter_from_ppo_output(ppo_output_dir: str, component_name: str) -> Optional[str]:
    """
    Find the best available adapter from a PPO output directory.

    Returns:
        str or None: Path to the most recent valid adapter, or None if not found.
    """
    # Resolve component-specific path
    if os.path.basename(ppo_output_dir) == component_name:
        component_dir = ppo_output_dir
    else:
        component_dir = os.path.join(ppo_output_dir, "ppo", component_name)

    if not os.path.isdir(component_dir):
        logger.warning(f"No PPO output found for component: {component_dir}")
        return None

    # Check final checkpoint
    final_dir = os.path.join(component_dir, "final")
    if _is_valid_adapter_dir(final_dir):
        logger.info(f"Using final adapter: {final_dir}")
        return final_dir

    # Check step-based checkpoints
    step_dirs = []
    for name in os.listdir(component_dir):
        if name.startswith("step_"):
            path = os.path.join(component_dir, name)
            if os.path.isdir(path) and _is_valid_adapter_dir(path):
                try:
                    step_num = int(name.split("_")[-1])
                    step_dirs.append((step_num, path))
                except ValueError:
                    continue

    if step_dirs:
        step_dirs.sort(key=lambda x: x[0])
        latest = step_dirs[-1][1]
        logger.info(f"Using latest step adapter: {latest}")
        return latest

    logger.warning(f"No valid adapter found in {component_dir}")
    return None


def _is_valid_adapter_dir(path: str) -> bool:
    """Check if a directory contains the required files for a LoRA adapter."""
    if not os.path.isdir(path):
        return False

    config_path = os.path.join(path, "adapter_config.json")
    has_model = any(
        os.path.exists(os.path.join(path, fname))
        for fname in ["adapter_model.bin", "adapter_model.safetensors"]
    )
    return os.path.exists(config_path) and has_model
