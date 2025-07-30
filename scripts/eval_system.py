import os
import os.path as osp
import sys
import json
import yaml
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from dotenv import load_dotenv
from transformers import HfArgumentParser

from examples.systems import registered_systems
from examples.datasets import registered_datasets
from optimas.optim.args import OptimasArguments
from optimas.utils.logger import setup_logger
from optimas.eval import eval_system


@dataclass
class ScriptArgs:
    """
    Configuration class for system evaluation and optimization script.
    """
    # Model and Pipeline Configuration
    run_name: str = "default"
    dataset: str = "hotpotqa"
    system_name: str = "hotpotqa_system"
    num_repeat: int = 3
    split: str = "test"  # Options: "val", "test"

    # Paths and Environment
    system_state_dict_path: Optional[str] = None
    output_dir: str = "outputs/optim"
    dotenv_path: str = ".env"

    # Worker Config (injected dynamically)
    max_sample_workers: int = 4
    max_eval_workers: int = 2

    def __post_init__(self):
        """Convert string 'None' values to None objects."""
        for field_name, field_value in vars(self).items():
            if field_value == "None":
                setattr(self, field_name, None)


def main():
    parser = HfArgumentParser(ScriptArgs)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        with open(sys.argv[1], "r") as f:
            yaml_config = yaml.safe_load(f)
        args = parser.parse_dict(yaml_config)[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]

    # Load environment variables
    load_dotenv(args.dotenv_path)

    # Setup output directory
    args.output_dir = osp.join(args.output_dir, args.dataset, args.system_name, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    logger = setup_logger(__name__, log_file=osp.join(args.output_dir, "output.log"))
    logger.info("Starting evaluation script.")
    logger.info(f"Parsed arguments:\n{json.dumps(vars(args), indent=2)}")

    # Construct system
    logger.info(f"Instantiating system: {args.system_name}")
    system_kwargs = {
        "log_dir": args.output_dir,
        "max_sample_workers": args.max_sample_workers,
        "max_eval_workers": args.max_eval_workers,
    }
    system = registered_systems[args.system_name](**system_kwargs)

    cur_system_state_dict = system.state_dict()
    logger.info(f"Saving current system state.")
    torch.save(cur_system_state_dict, osp.join(args.output_dir, "system_state_dict.pth.bak"))

    # Load evaluation datasets
    trainset, valset, testset = registered_datasets[args.dataset](**vars(args))

    # Restore state if specified
    if args.system_state_dict_path:
        logger.info(f"Loading system state dict from {args.system_state_dict_path}")
        state_dict = torch.load(args.system_state_dict_path)
        system.load_state_dict(state_dict)
        logger.info(f"Loaded state dict keys: {list(state_dict.keys())}")

    metrics_path = osp.join(args.output_dir, f"{args.split}_metrics.json")
    dataset = valset if args.split == "val" else testset
    metrics, preds = eval_system(system=system, testset=dataset, num_repeat=args.num_repeat)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    logger.info(f"Metrics written to {metrics_path}")
    logger.info(f"Final metrics:\n{json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
