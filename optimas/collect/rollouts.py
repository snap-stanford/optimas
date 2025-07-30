import copy
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm
from datasets import Dataset, DatasetDict

from optimas.arch.system import CompoundAISystem
from optimas.collect.process import process_dataset_parallel
from optimas.utils.context import get_context_from_traj
from optimas.utils.logger import setup_logger
from optimas.utils.operation import unique_objects
from optimas.wrappers.example import Example
from optimas.wrappers.prediction import Prediction


logger = setup_logger(__name__)


def generate_reward_dataset(
    system: CompoundAISystem,
    dataset: Union[List[Example], Tuple[List[Example], List[Prediction]]],
    component_names: List[str] = None,
    num_rollouts: int = 3,
    num_samples: int = 3,
    max_workers: int = 16,
    process_reward_func: Any = None,
    forward_to_component: str = None,
    sample_temperature: float = None,
    num_repeats: int = 1,
    highest_score: float = 1.0,
    **kwargs,
) -> DatasetDict:
    """
    Generate a preference-based reward dataset by perturbing and evaluating trajectories
    for each optimizable component in a compound AI system.

    Steps:
        1. Run base dataset through the system to collect predictions.
        2. For each component:
           - Sample perturbed trajectories.
           - Evaluate those trajectories.
           - Construct preference pairs (chosen vs rejected).
        3. Return a DatasetDict keyed by component name.

    Args:
        system (CompoundAISystem): The system to process and evaluate examples.
        dataset (Union[List[Example], Tuple[List[Example], List[Prediction]]]):
            Either a raw list of examples or a tuple of (examples, predictions).
        component_names (List[str], optional): Specific components to perturb. Defaults to all optimizable.
        num_rollouts (int): Number of rollouts for evaluation.
        num_samples (int): Number of samples to generate per example.
        max_workers (int): Max parallel workers for processing.
        process_reward_func (Callable, optional): Optional reward function override.
        forward_to_component (str, optional): Where to cut off forward passes.
        sample_temperature (float, optional): Sampling temperature for stochastic components.
        num_repeats (int): Repetition count for sampling robustness.
        **kwargs: Unused additional arguments.

    Returns:
        DatasetDict: Each component maps to a Dataset of preference pairs.
    """
    # Prepare system context for sampling
    system_context = {}
    for name, component in system.components.items():
        if hasattr(component, "temperature") and sample_temperature is not None:
            system_context[name] = {"temperature": sample_temperature}
        if getattr(component, "variable_search_space", None):
            system_context.setdefault(name, {})["randomize_variable"] = True

    # Process the dataset to get base predictions
    if isinstance(dataset, tuple):
        logger.info("Using pre-processed predictions.")
        examples, preds = dataset
    else:
        with system.context(system_context):
            examples, preds, _ = process_dataset_parallel(dataset, system, max_workers=max_workers)

    if component_names is None:
        component_names = [k for k, v in system.components.items() if getattr(v, "optimizable", False)]

    dataset_dict = {}

    # Iterate over each component
    for component_name in component_names:
        flat_eval_list = []

        # A. Sample trajectories
        for i, (example, pred) in enumerate(zip(examples, preds)):
            for _ in range(num_repeats):
                with system.context(system_context):
                    traj_samples = generate_samples(
                        system=system,
                        example=example,
                        traj=pred.traj,
                        components_to_perturb=[component_name],
                        num_samples=num_samples,
                        max_workers=max_workers,
                    )

                if traj_samples and len(traj_samples) > 1:
                    for traj in traj_samples:
                        flat_eval_list.append((i, example, traj))
                else:
                    logger.info(f"Skipping example {i} for component '{component_name}' due to insufficient variants.")

        # B. Evaluate all trajectories
        scores_infos = evaluate_samples(
            system=system,
            examples=[ex for _, ex, _ in flat_eval_list],
            trajs=[traj for _, _, traj in flat_eval_list],
            max_workers=max_workers,
            num_rollouts=num_rollouts,
            forward_to_component=forward_to_component,
            process_reward_func=process_reward_func,
        )

        # C. Group results by original example index
        grouped_results = defaultdict(list)
        for (idx, _, traj), (score, info) in zip(flat_eval_list, scores_infos):
            grouped_results[idx].append((score, info, traj))

        # D. Construct preference pairs
        preference_data = []
        for idx, example in enumerate(examples):
            if idx not in grouped_results or len(grouped_results[idx]) < 2:
                continue

            results = grouped_results[idx]
            scores, infos, trajs = zip(*results)

            chosen_idx = int(np.argmax(scores))
            rejected_idx = int(np.argmin(scores))

            reference_traj = trajs[0]
            context = {
                k: v for k, v in reference_traj.items() if k != component_name
            }
            context[component_name] = {
                "input": reference_traj[component_name]["input"]
            }

            # Add groundtruth if needed and available
            if component_name == list(system.components.keys())[-1]:
                try:
                    if any(k in system.ground_fields for k in ["gd_answer", "groundtruth"]):
                        key = system.final_output_fields[0]
                        ground_value = example[system.ground_fields[0]]
                        trajs += ({component_name: {"output": {key: ground_value}, "variable": "none"}},)
                        infos += ([None],)
                        scores += (highest_score,)
                    else:
                        outputs = {k: example[k] for k in system.final_output_fields}
                        trajs += ({component_name: {"output": outputs, "variable": "none"}},)
                        infos += ([None],)
                        scores += (highest_score,)
                except Exception as e:
                    logger.error(f"Could not add groundtruth to example {idx}: {e}")

            # Pairwise preferences where score[i] > score[j] and i > j
            for i in range(len(scores)):
                for j in range(len(scores)):
                    if scores[i] > scores[j] and i > j:
                        preference_data.append({
                            "context": json.dumps(context),
                            "response_chosen": json.dumps(trajs[i][component_name]["output"]),
                            "response_rejected": json.dumps(trajs[j][component_name]["output"]),
                            "score_chosen": scores[i],
                            "score_rejected": scores[j],
                            "info_chosen": json.dumps(infos[i]),
                            "info_rejected": json.dumps(infos[j]),
                            "variable_chosen": json.dumps(trajs[i][component_name]["variable"]),
                            "variable_rejected": json.dumps(trajs[j][component_name]["variable"]),
                        })

        # E. Convert to Dataset
        if preference_data:
            dataset_dict[component_name] = Dataset.from_dict({
                k: [entry[k] for entry in preference_data] for k in preference_data[0]
            })

    return DatasetDict(dataset_dict)


def generate_samples(
    system: CompoundAISystem,
    example: Example,
    components_to_perturb: List[str],
    traj: Dict,
    num_samples: int = 3,
    max_workers: int = 8,
) -> List[Dict[str, Any]]:
    """
    Generate candidate trajectories by perturbing specified components of a base trajectory.

    Args:
        system (CompoundAISystem): The system used to run subcomponents.
        example (Example): The base input example.
        components_to_perturb (List[str]): Names of components to perturb.
        traj (Dict): Original trajectory from which to generate variants.
        num_samples (int): Number of perturbed samples to generate.
        max_workers (int): Parallel worker threads for generating samples.

    Returns:
        List[Dict[str, Any]]: Unique perturbed trajectories including the original.
    """
    component_names = list(system.components.keys())
    component_indices = [component_names.index(name) for name in components_to_perturb]
    earliest_idx, latest_idx = min(component_indices), max(component_indices)
    latest_name = component_names[latest_idx]

    original_traj = {name: traj[name] for name in component_names[:latest_idx + 1]}

    # Build context from required inputs and early component outputs
    context = {
        **{key: getattr(example, key) for key in system.required_input_fields},
        **{
            k: v
            for name in component_names[:earliest_idx]
            for k, v in traj[name]["input"].items()
        },
        **{
            k: v
            for name in component_names[:earliest_idx]
            for k, v in traj[name]["output"].items()
        },
    }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        perturbed_samples = list(
            executor.map(
                lambda _: system.run_subsystem(
                    start_component=earliest_idx,
                    end_component=latest_idx,
                    **context
                ),
                range(num_samples)
            )
        )

    perturbed_trajs = [{**original_traj, **s.traj} for s in perturbed_samples]
    perturbed_trajs.append(original_traj)

    # Keep only unique outputs from the final component
    _, unique_indices = unique_objects(
        [traj[latest_name]["output"] for traj in perturbed_trajs],
        return_idx=True
    )
    return [perturbed_trajs[i] for i in unique_indices]


def evaluate_samples(
    system: CompoundAISystem,
    examples: List[Example],
    trajs: List[Dict],
    forward_to_component: Optional[str] = None,
    process_reward_func: Optional[Callable] = None,
    max_workers: int = 8,
    num_rollouts: int = 3,
) -> List[Tuple[float, List[Dict[str, Any]]]]:
    """
    Evaluate a batch of (example, trajectory) pairs in parallel.

    Args:
        system (CompoundAISystem): System used for inference and scoring.
        examples (List[Example]): Input examples.
        trajs (List[Dict]): Corresponding trajectories to evaluate.
        forward_to_component (str, optional): End component for partial forward passes.
        process_reward_func (Callable, optional): Used if ending early, to compute custom reward.
        max_workers (int): Number of parallel threads.
        num_rollouts (int): Number of stochastic rollouts per sample.

    Returns:
        List[Tuple[float, List[Dict[str, Any]]]]:
            A list of (average score, rollout info) tuples per (example, traj).
    """
    if len(examples) != len(trajs):
        raise ValueError("Examples and trajectories must have the same length.")

    component_names = list(system.components.keys())
    forward_to_component = forward_to_component or component_names[-1]

    def _evaluate_single(pair: Tuple[Example, Dict]) -> Tuple[float, List[Dict[str, Any]]]:
        example, traj = pair
        local_traj = copy.copy(traj)

        if not local_traj:
            raise ValueError("Trajectory is empty.")

        last_idx = max(component_names.index(name) for name in local_traj)

        context = get_context_from_traj(local_traj)
        context.update({k: getattr(example, k) for k in system.required_input_fields})

        scores, preds = [], []
        for _ in range(num_rollouts):
            if last_idx + 1 < len(component_names):
                pred = system.run_subsystem(
                    start_component=last_idx + 1,
                    end_component=forward_to_component,
                    **context
                )
            else:
                pred = Prediction(**context, traj=local_traj)

            score = (
                system.evaluate(example, pred)
                if forward_to_component == component_names[-1]
                else process_reward_func(system, example, pred)
            )
            scores.append(score)
            preds.append(pred)

        avg_score = sum(scores) / len(scores)
        rollout_info = [{"traj": p.traj, "score": s} for p, s in zip(preds, scores)]
        return avg_score, rollout_info

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(_evaluate_single, zip(examples, trajs)),
                total=len(examples),
                desc="Evaluating Samples"
            )
        )

    return results
