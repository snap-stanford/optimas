import os
import os.path as osp
import sys
import time
import joblib
import argparse
import yaml
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

from examples.datasets import registered_datasets
from examples.systems import registered_systems
from optimas.collect.rollouts import generate_reward_dataset


if __name__ == "__main__":

    # args run_name name
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("config", type=str, help="Path to YAML config file")
        args = parser.parse_args()

        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)

        return argparse.Namespace(**config_dict)

    args = parse_args()
    dotenv_path = osp.expanduser('.env')
    load_dotenv(dotenv_path)

    ##############################################
    #           Define the system              #
    ##############################################
    system = registered_systems[args.system](
        max_sample_workers=args.max_workers, 
        max_eval_workers=args.max_workers
    )
    trainset, valset, _ = registered_datasets[args.dataset]()

    trainset = trainset[:args.train_size]
    reward_dataset = generate_reward_dataset(
        system, trainset,
        sample_temperature=args.sample_temperature,
        max_workers=args.max_workers,
        num_rollouts=args.num_rollouts,
        num_samples=args.num_samples,
        num_repeats=args.num_repeats
    )
    os.makedirs(args.output_dir, exist_ok=True)
    reward_dataset.save_to_disk(osp.join(args.output_dir, f"{args.system}-{len(trainset)}"))

    if args.push_to_hub:
        reward_dataset.push_to_hub(f'{args.system}-{len(trainset)}')
