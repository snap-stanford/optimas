import os
import re
import json
import shutil
import wandb
import importlib.util
from pathlib import Path
from typing import List, Optional, Callable, Dict

from transformers import TrainerCallback, AutoTokenizer

from optimas.utils.save import save_model_and_tokenizer
from optimas.utils.logger import setup_logger
from optimas.arch.system import CompoundAISystem

logger = setup_logger(__name__)

PREFIX_STATE_DICT_DIR = "full"
PREFIX_CHECKPOINT_DIR = "checkpoint"


def is_wandb_available() -> bool:
    """Check if Weights & Biases is available and not disabled via environment."""
    disabled_values = {"1", "ON", "YES", "TRUE"}
    if os.getenv("WANDB_DISABLED", "").upper() in disabled_values:
        logger.warning(
            "Using the WANDB_DISABLED environment variable is deprecated and will be removed in v5. "
            "Use --report_to instead (e.g., --report_to none)."
        )
        return False
    return importlib.util.find_spec("wandb") is not None


class PerComponentSaveCallback(TrainerCallback):
    """
    TrainerCallback that manages saving checkpoints for the full system and individual components,
    tracking best-performing models based on evaluation metrics.
    """

    def __init__(
        self,
        system: CompoundAISystem,
        tokenizer: AutoTokenizer,
        metric_for_best_model: str,
        repo_name: Optional[str] = None,
        criteria: Optional[Callable[[str], bool]] = None,
        save_model_per_component: bool = True,
    ):
        """
        Args:
            system: CompoundAISystem instance.
            tokenizer: HuggingFace tokenizer to be saved with models.
            metric_for_best_model: Evaluation metric to track best model.
            repo_name: Optional hub repo name.
            criteria: Optional function to filter which parameters to save.
            save_model_per_component: Whether to track best checkpoints per component.
        """
        self.tokenizer = tokenizer
        self.repo_name = repo_name
        self.metric_for_best_model = metric_for_best_model
        self.save_model_per_component = save_model_per_component
        self.criteria = criteria or (lambda x: "score.weight" in x.lower() or "lora" in x.lower())
        self.component_names = [
            name for name, comp in (system.components.items() if system else [])
            if comp.optimizable
        ]
        self.best_step_per_component: Dict[str, int] = {}

    def on_train_begin(self, args, state, control, model, **kwargs):
        if state.is_world_process_zero:
            self.init_model_path = os.path.join(args.output_dir, f"{PREFIX_STATE_DICT_DIR}-init")
            save_model_and_tokenizer(
                model, self.tokenizer, self.init_model_path,
                repo_name=self.repo_name, push_to_hub=False, criteria=self.criteria
            )

    def on_save(self, args, state, control, model, **kwargs):
        if not state.is_world_process_zero:
            return

        step_ckpt = os.path.join(args.output_dir, f"{PREFIX_STATE_DICT_DIR}-{state.global_step}")
        save_model_and_tokenizer(
            model, self.tokenizer, step_ckpt,
            repo_name=f"{self.repo_name}-{state.global_step}" if self.repo_name else None,
            push_to_hub=False, criteria=self.criteria
        )

        best_ckpt = self._get_best_checkpoint_path(args.output_dir, state)
        self._rotate_checkpoints(args, state, best_model_checkpoint=best_ckpt)

        if self.save_model_per_component:
            self._update_component_best_steps(state)
            self._save_per_component_checkpoints(args, state, model, best_ckpt)

    def on_train_end(self, args, state, control, model, **kwargs):
        if not state.is_world_process_zero:
            return

        last_ckpt = os.path.join(args.output_dir, f"{PREFIX_STATE_DICT_DIR}-last")
        save_model_and_tokenizer(
            model, self.tokenizer, last_ckpt,
            repo_name=self.repo_name, push_to_hub=False, criteria=self.criteria
        )

        model_info = {}

        if state.best_model_checkpoint:
            best_ckpt_name = os.path.basename(state.best_model_checkpoint).replace(PREFIX_CHECKPOINT_DIR, PREFIX_STATE_DICT_DIR)
            best_ckpt_path = os.path.join(args.output_dir, best_ckpt_name)
            best_model_path = os.path.join(args.output_dir, f"{PREFIX_STATE_DICT_DIR}-best")
            os.rename(best_ckpt_path, best_model_path)

            model_info.update({
                "best_metric": state.best_metric,
                "best_model_step": int(best_ckpt_name.split("-")[-1])
            })

            if self.save_model_per_component:
                for m in self.component_names:
                    step = self.best_step_per_component.get(m)
                    if step:
                        src = os.path.join(args.output_dir, f"{PREFIX_STATE_DICT_DIR}-{m}-{step}")
                        dst = os.path.join(args.output_dir, f"{PREFIX_STATE_DICT_DIR}-{m}-best")
                        os.rename(src, dst)
                        model_info[f"best_{m}_step"] = step
        else:
            os.rename(self.init_model_path, os.path.join(args.output_dir, f"{PREFIX_STATE_DICT_DIR}-best"))
            model_info = {"best_metric": None, "best_model_step": 0}

        if is_wandb_available():
            table = wandb.Table(columns=["Metric", "Value"])
            for k, v in model_info.items():
                table.add_data(k, v)
            wandb.log({"model_results": table})

        with open(os.path.join(args.output_dir, "model_info.json"), "w") as f:
            json.dump(model_info, f, indent=4)

    def _update_component_best_steps(self, state):
        self.best_step_per_component = {
            m: self._get_best_step_from_log_history(
                state.log_history,
                self.metric_for_best_model.replace("eval_", f"eval_{m}_"),
                higher_is_better=not "loss" in self.metric_for_best_model
            ) for m in self.component_names
        }

    def _save_per_component_checkpoints(self, args, state, model, best_ckpt):
        for m, step in self.best_step_per_component.items():
            if step == state.global_step:
                path = os.path.join(args.output_dir, f"{PREFIX_STATE_DICT_DIR}-{m}-{step}")
                save_model_and_tokenizer(
                    model, self.tokenizer, path,
                    repo_name=f"{self.repo_name}-{m}-{step}" if self.repo_name else None,
                    push_to_hub=False, criteria=self.criteria
                )
                self._rotate_checkpoints(args, state, best_model_checkpoint=best_ckpt, prefix=f"{PREFIX_STATE_DICT_DIR}-{m}")

    def _get_best_checkpoint_path(self, output_dir: str, state) -> Optional[str]:
        if state.best_model_checkpoint:
            return os.path.join(
                output_dir,
                os.path.basename(state.best_model_checkpoint).replace(PREFIX_CHECKPOINT_DIR, PREFIX_STATE_DICT_DIR)
            )
        return None

    def _rotate_checkpoints(self, args, state, best_model_checkpoint: Optional[str], prefix: str = PREFIX_STATE_DICT_DIR):
        if not args.save_total_limit or args.save_total_limit <= 0:
            return

        all_ckpts = self._sorted_checkpoints(args.output_dir, prefix, best_model_checkpoint)
        if len(all_ckpts) <= args.save_total_limit:
            return

        save_limit = args.save_total_limit
        if best_model_checkpoint and save_limit == 1 and all_ckpts[-1] != best_model_checkpoint:
            save_limit = 2

        for ckpt in all_ckpts[: max(0, len(all_ckpts) - save_limit)]:
            logger.info(f"Deleting outdated checkpoint: {ckpt}")
            shutil.rmtree(ckpt, ignore_errors=True)

    def _sorted_checkpoints(self, output_dir: str, prefix: str, best_model_checkpoint: Optional[str]) -> List[str]:
        pattern = rf"{prefix}-(\d+)"
        paths = [str(p) for p in Path(output_dir).glob(f"{prefix}-*") if p.is_dir()]
        sorted_paths = sorted(
            (int(re.search(pattern, p).group(1)), p)
            for p in paths if re.search(pattern, p)
        )
        sorted_ckpts = [p for _, p in sorted_paths]

        if best_model_checkpoint in sorted_ckpts:
            sorted_ckpts.append(sorted_ckpts.pop(sorted_ckpts.index(best_model_checkpoint)))

        return sorted_ckpts

    def _get_best_step_from_log_history(self, log_history: List[dict], metric_key: str, higher_is_better: bool) -> Optional[int]:
        best_metric, best_step = None, None
        for log in log_history:
            if metric_key in log:
                metric = log[metric_key]
                if best_metric is None or (higher_is_better and metric > best_metric) or (not higher_is_better and metric < best_metric):
                    best_metric = metric
                    best_step = log["step"]
        return best_step
