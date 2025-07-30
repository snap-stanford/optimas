import os
import sys
import json
import numpy as np
from typing import Union
from datasets import load_dataset, Dataset, DatasetDict

from optimas.utils.template import apply_reward_template
from optimas.arch.system import CompoundAISystem
from optimas.utils.context import get_context_from_traj
from optimas.utils.logger import setup_logger
from datasets import load_from_disk

logger = setup_logger(__name__)


class RewardDataset:
    def __init__(
        self,
        hf_repo_or_local_dir: str,
        system: CompoundAISystem,
        original_dataset: DatasetDict = None,
    ):
        if os.path.exists(hf_repo_or_local_dir):
            self.reward_dataset = load_from_disk(hf_repo_or_local_dir)
        else:
            self.reward_dataset = load_dataset(hf_repo_or_local_dir)
        self.system = system
        self.component_name_lst = list(self.reward_dataset.keys())

        key = self.component_name_lst[0]
        self.columns = list(self.reward_dataset[key].features.keys())

        self.original_dataset = original_dataset
        if self.original_dataset:
            self.hash_dict_func = lambda x: hash(json.dumps(x, sort_keys=True))
            # create hash mapping for each model
            
            self.input_fields_to_gd_fields = {}
            for m in self.component_name_lst:
                inspect_eg = self.reward_dataset[m][0]
                inspect_context_dict = get_context_from_traj(json.loads(inspect_eg['context']))
                component_context_input_keys = [k for k in inspect_context_dict.keys() if k in self.system.required_input_fields]
                logger.info(f"Module: {m} | Context keys: {component_context_input_keys}")
                for example in self.original_dataset:
                    component_context_input = {k: example[k] for k in component_context_input_keys}
                    self.input_fields_to_gd_fields[self.hash_dict_func(component_context_input)] = {k: example[k] for k in self.system.ground_fields}


    def to_inputs_only_dataset(self, component_name_lst=None):
        """
        Convert the dataset to an inputs
        """
        if component_name_lst is None:
            component_name_lst = self.component_name_lst

        data_dict = {}
        for m in component_name_lst:
            data_dict[m] = {'input': []}
            # find the keys that appear in both required_input_fields and trajectory
            inspect_eg = self.reward_dataset[m][0]
            inspect_context_dict = get_context_from_traj(json.loads(inspect_eg['context']))
            component_context_input_keys = [k for k in inspect_context_dict.keys() if k in self.system.required_input_fields]

            logger.info(f"to_inputs_only_dataset: Module: {m} | Context keys: {component_context_input_keys}")

            for example in self.reward_dataset[m]:
                context_dict = get_context_from_traj(json.loads(example['context']))
                context = {k: context_dict[k] for k in self.system.components[m].input_fields}

                if self.original_dataset:
                    try:
                        component_context_input = {k: context_dict[k] for k in component_context_input_keys}
                        gd_fields = self.input_fields_to_gd_fields[self.hash_dict_func(component_context_input)]
                        context.update(gd_fields)
                    except KeyError as e:
                        logger.warning(f"Cannot match example with the original dataset. Skipping.")
                        continue

                data_dict[m]['input'].append(context)

            data_dict[m] = Dataset.from_dict(data_dict[m])

        return DatasetDict(data_dict)

    def to_preference_dataset(
        self, eval_ratio=0.1, 
        component_name_lst=None,
        add_margin=False, margin_threshold=0.0, normalize_margin=True
    ):
        """
        Convert the dataset to an implicit preference dataset containing columns:
           ["chosen", "rejected"] (+ optional "margin").
        This is typically used for pairwise preference training.
        """
        assert 'response_chosen' in self.columns
        assert 'response_rejected' in self.columns
        assert 'context' in self.columns
        assert ('score_chosen' in self.columns and 'score_rejected' in self.columns) if add_margin else True

        if component_name_lst is None:
            component_name_lst = self.component_name_lst

        data_dict = {'chosen': [], 'rejected': [], 'component_name': []}
        if add_margin:
            data_dict['margin'] = []
        if self.original_dataset:
            data_dict.update({'required_input_fields': [], 'ground_fields': []})

        # one more column: example
        def _invalid_margin(example):
            if add_margin:
                return float(example['score_chosen'] - example['score_rejected']) < margin_threshold
            return False

        for component_name in component_name_lst:
            desc = self.system.components[component_name].description
            num_valid = 0

            inspect_eg = self.reward_dataset[component_name][0]
            inspect_context_dict = get_context_from_traj(json.loads(inspect_eg['context']))
            component_context_input_keys = [k for k in inspect_context_dict.keys() if k in self.system.required_input_fields]

            for example in self.reward_dataset[component_name]:
                if margin_threshold > 0 and _invalid_margin(example):
                    continue
                num_valid += 1

                context_dict = get_context_from_traj(json.loads(example['context']))
                input_dict = {
                    k: context_dict[k]
                    for k in self.system.components[component_name].input_fields
                }

                # chosen
                chosen_text = apply_reward_template(
                    input_dict,
                    json.loads(example['response_chosen']),
                    desc=desc
                )
                data_dict['chosen'].append(chosen_text)

                if self.original_dataset:
                    # Get the ground truth fields

                    component_context_input = {k: context_dict[k] for k in component_context_input_keys}
                    gd_fields = self.input_fields_to_gd_fields[self.hash_dict_func(component_context_input)]
                    data_dict['required_input_fields'].append(component_context_input)
                    data_dict['ground_fields'].append(gd_fields)

                # rejected
                rejected_text = apply_reward_template(
                    input_dict,
                    json.loads(example['response_rejected']),
                    desc=desc
                )
                data_dict['rejected'].append(rejected_text)
                data_dict['component_name'].append(component_name)

                if add_margin:
                    data_dict['margin'].append(float(example['score_chosen']) - float(example['score_rejected']))

            logger.info(f"({component_name}) After-filtering: #{num_valid} | Ratio: {num_valid / len(self.reward_dataset[component_name])}")

        if add_margin and normalize_margin:
            margin = np.array(data_dict['margin'])
            data_dict['margin'] = ((margin - np.min(margin)) / (np.max(margin) - np.min(margin))).tolist()

        dataset = Dataset.from_dict(data_dict)

        if eval_ratio > 0:
            dataset = dataset.train_test_split(test_size=eval_ratio, seed=42)
            full_testset = dataset['test']
            dataset['test'] = {
                component_name: full_testset.filter(lambda x: x['component_name'] == component_name)
                for component_name in component_name_lst
            }
        else:
            dataset = DatasetDict({'train': dataset})
        return dataset

