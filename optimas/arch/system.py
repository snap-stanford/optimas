import os
import random
import warnings
import datetime
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Set
from contextlib import contextmanager, nullcontext
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from optimas.arch.base import BaseComponent
from optimas.utils.display import display_system_overview
from optimas.utils.operation import unique_objects
from optimas.wrappers.example import Example
from optimas.wrappers.prediction import Prediction
from optimas.utils.parallel import run_parallel_tasks
from optimas.utils.logger import setup_logger

logger = setup_logger(__name__)


class CompoundAISystem:
    """
    A compound AI system that executes components in sequence with optional reward model filtering.
    """

    def __init__(self, 
                 components: Optional[Dict[str, BaseComponent]], 
                 final_output_fields: Optional[List[str]] = None,
                 ground_fields: Optional[List[str]] = None,
                 eval_func: Optional[Callable] = None,
                 max_sample_workers: int = 4, 
                 max_eval_workers: int = 4, 
                 log_dir: Optional[str] = None,
                 **eval_kwargs):
        """
        Initialize the compound AI system.
        
        Args:
            components: Dictionary of components (name -> component) or None to add later
            final_output_fields: Keys for final output (inferred if not provided)
            ground_fields: Keys for ground truth data (inferred if not provided)  
            eval_func: Evaluation function
            max_sample_workers: Number of threads for parallel sampling
            max_eval_workers: Number of threads for parallel evaluation
            log_dir: Directory for logging rollout data
            **eval_kwargs: Additional evaluation parameters
        """
        self.max_sample_workers = max_sample_workers
        self.max_eval_workers = max_eval_workers
        
        # Initialize components
        self.components: Dict[str, BaseComponent] = {}
        self.register_components(components)
        
        # System configuration
        self.execution_order: List[str] = list(self.components.keys()) if components else []
        self.final_output_fields: List[str] = final_output_fields or []
        self.ground_fields: List[str] = ground_fields or []
        self.eval_func: Callable = eval_func
        self.external: Dict[str, Any] = eval_kwargs
        
        # Computed fields (set during finalization)
        self.required_input_fields: List[str] = []
        self.optimizable_components: List[str] = []
        self.optimizable_component_to_idx: Dict[str, int] = {}
        
        # Reward model configuration
        self.rm = None
        self.sample_temperature = None
        self.sample_size = None
        self.components_to_apply = []
        
        # Logging setup
        self._setup_logging(log_dir)
        
        # Auto-finalize if we have enough information
        self.validate_system()

        if os.environ.get('LOCAL_RANK', '0') == '0':
            display_system_overview(self)

    def _setup_logging(self, log_dir: Optional[str]) -> None:
        """Set up logging configuration."""
        self.log_dir = log_dir
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_file = os.path.join(log_dir, f"rollouts_{timestamp}.csv")
            self.log_df = pd.DataFrame(columns=["component_name", "component_inputs", "scores", "outputs"])
        else:
            self.log_file = None
            self.log_df = None

    def state_dict(self) -> Dict[str, Any]:
        """Return the current state of all components."""
        state = {}
        for name, component in self.components.items():
            state[name] = {
                'variable': component.variable,
                'config': component.config
            }
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load component states from a state dictionary."""
        for name, state_info in state.items():
            if name not in self.components:
                logger.warning(f"Component '{name}' not found in system")
                continue
                
            component = self.components[name]
            component.update(state_info['variable'])
            if 'config' in state_info:
                component.update_config(**vars(state_info['config']))

    def _compute_execution_order(self) -> List[str]:
        """
        Compute the topological execution order based on component dependencies.
        
        Returns:
            List of component names in topological order
            
        Raises:
            ValueError: If there are circular dependencies
        """
        # Build dependency graph
        graph = defaultdict(set)  # component -> set of dependencies
        in_degree = defaultdict(int)  # component -> number of incoming edges
        all_components = set(self.components.keys())
        
        # Initialize in-degree for all components
        for component_name in all_components:
            in_degree[component_name] = 0
        
        # Build the dependency graph
        for component_name, component in self.components.items():
            component_inputs = set(getattr(component, 'input_fields', []))
            
            # Find which components provide these inputs
            for input_field in component_inputs:
                for provider_name, provider_component in self.components.items():
                    if provider_name != component_name:
                        provider_outputs = set(getattr(provider_component, 'output_fields', []))
                        if input_field in provider_outputs:
                            # provider_name must execute before component_name
                            if provider_name not in graph[component_name]:
                                graph[component_name].add(provider_name)
                                in_degree[component_name] += 1
        
        # Topological sort using Kahn's algorithm
        queue = deque([comp for comp in all_components if in_degree[comp] == 0])
        execution_order = []
        
        while queue:
            current = queue.popleft()
            execution_order.append(current)
            
            # Remove current component from graph and update in-degrees
            for dependent_comp in all_components:
                if current in graph[dependent_comp]:
                    graph[dependent_comp].remove(current)
                    in_degree[dependent_comp] -= 1
                    if in_degree[dependent_comp] == 0:
                        queue.append(dependent_comp)
        
        # Check for cycles
        if len(execution_order) != len(all_components):
            remaining = all_components - set(execution_order)
            raise ValueError(f"Circular dependency detected among components: {remaining}")
        
        return execution_order


    def _infer_final_outputs(self) -> List[str]:
        """Infer final output fields from the last component."""
        if not self.components:
            return []
        last_component = list(self.components.values())[-1]
        return list(getattr(last_component, 'output_fields', []))


    def validate_system(self) -> 'CompoundAISystem':
        """
        Validate and finalize the system configuration.
        
        Returns:
            Self for method chaining
        """
        # Auto-compute execution order from dependencies
        if self.components:
            self.execution_order = self._compute_execution_order()
        
        # Auto-infer final outputs if not set
        if not self.final_output_fields:
            self.final_output_fields = self._infer_final_outputs()
        
        # Validate that we have components
        if not self.components:
            raise ValueError("No components registered in the system")

        # Compute required input fields
        input_fields = set()
        output_fields = set()
        for component in self.components.values():
            input_fields.update(getattr(component, 'input_fields', []))
            output_fields.update(getattr(component, 'output_fields', []))
        self.required_input_fields = list(input_fields - output_fields)

        # Identify optimizable components
        self.optimizable_components = [
            name for name, component in self.components.items() 
            if getattr(component, 'optimizable', False)
        ]
        self.optimizable_component_to_idx = {
            name: idx for idx, name in enumerate(self.optimizable_components)
        }
        
        return self

    def register_component(self, name: str, component: BaseComponent) -> None:
        """Register a single component."""
        if name in self.components:
            raise ValueError(f"Component '{name}' already registered")
        self.components[name] = component

    def register_components(self, components: Dict[str, BaseComponent]) -> None:
        """Register multiple components."""
        for name, component in components.items():
            self.register_component(name, component)

    def construct_system(self, component_order: List[str], final_output_fields: List[str], 
                        ground_fields: List[str], eval_func: Optional[Callable] = None, **kwargs) -> None:
        """
        Configure the system execution order and evaluation (legacy method).
        
        Args:
            component_order: Execution order of components
            final_output_fields: Keys for final output
            ground_fields: Keys for ground truth data
            eval_func: Evaluation function
            **kwargs: External parameters for evaluation
        """
        warnings.warn("construct_system is deprecated. Use the new fluent API or constructor parameters.", 
                     DeprecationWarning, stacklevel=2)
        
        self.execution_order = component_order
        self.final_output_fields = final_output_fields
        self.ground_fields = ground_fields
        self.eval_func = eval_func
        self.external.update(kwargs)
        
        self.validate_system()

    def register_rm(self, rm: Union[Any, Dict[str, Any]], components_to_apply: List[str] = None,
                   sample_temperature: Optional[float] = None, sample_size: int = 1) -> None:
        """
        Register a reward model for filtering.
        
        Args:
            rm: Reward model or dictionary of reward models
            components_to_apply: Components to apply RM filtering to ('all' for all optimizable)
            sample_temperature: Temperature for sampling
            sample_size: Number of samples to generate
        """
        self.rm = rm
        self.sample_temperature = sample_temperature
        self.sample_size = sample_size
        self.components_to_apply = components_to_apply or []

        if 'all' in self.components_to_apply:
            if len(self.components_to_apply) > 1:
                raise ValueError("Cannot mix 'all' with specific component names")
            self.components_to_apply = set(self.optimizable_components)

        invalid_components = set(self.components_to_apply) - set(self.optimizable_components)
        if invalid_components:
            raise ValueError(f"Invalid components in components_to_apply: {invalid_components}")
        
        logger.info(f"Applying RM filtering to components: {self.components_to_apply}")

    @property
    def predecessor_map(self) -> Dict[str, List[str]]:
        """Return mapping of components to their immediate predecessors."""
        predecessor_map = {}
        for component_idx, component_name in enumerate(self.execution_order):
            predecessor_map[component_name] = []
            component = self.components[component_name]
            
            for input_field in component.input_fields:
                for prev_idx in range(component_idx - 1, -1, -1):
                    prev_name = self.execution_order[prev_idx]
                    if input_field in self.components[prev_name].output_fields:
                        predecessor_map[component_name].append(prev_name)
                        break
        return predecessor_map

    @property
    def successor_map(self) -> Dict[str, List[str]]:
        """Return mapping of components to their immediate successors."""
        successor_map = {}
        for component_name, predecessors in self.predecessor_map.items():
            for predecessor in predecessors:
                successor_map.setdefault(predecessor, []).append(component_name)
        return successor_map

    @property
    def desc(self) -> Dict[str, str]:
        """Return descriptions of all registered components."""
        return {name: component.description for name, component in self.components.items()}

    @contextmanager
    def context(self, component_configs: Optional[Dict[str, Dict]] = None):
        """
        Context manager for temporarily modifying multiple components.
        
        Args:
            component_configs: Dictionary mapping component names to their configurations
        """
        if component_configs is None:
            component_configs = {}

        managed_contexts = []
        try:
            for component_name, config in component_configs.items():
                if component_name not in self.components:
                    raise ValueError(f"Component '{component_name}' not found")

                variable_config = config.pop('variable', None)
                if variable_config and config.get("randomize_variable"):
                    raise ValueError(f"Cannot set both variable and randomize_variable for {component_name}")

                context_mgr = self.components[component_name].context(variable=variable_config, **config)
                managed_contexts.append((context_mgr, component_name))
                context_mgr.__enter__()

                if variable_config is not None:
                    config['variable'] = variable_config

            yield self

        finally:
            exceptions = []
            for ctx_mgr, component_name in reversed(managed_contexts):
                try:
                    ctx_mgr.__exit__(None, None, None)
                except Exception as e:
                    logger.error(f"Error exiting context for component '{component_name}': {e}")
                    exceptions.append(e)
            
            if exceptions:
                raise exceptions[0]

    def __call__(self, **inputs: Any) -> Prediction:
        """Execute the system in LLM-based mode."""
        pred = self.run_subsystem(self.execution_order[0], self.execution_order[-1], **inputs)
        context = self._extract_context_from_traj(pred.traj)

        final_outputs = {key: context[key] for key in self.final_output_fields if key in context}
        missing_outputs = set(self.final_output_fields) - set(final_outputs.keys())
        if missing_outputs:
            raise ValueError(f"Missing final output keys: {missing_outputs}")

        return Prediction(**final_outputs, traj=pred.traj)

    def run_subsystem(self, start_component: Union[int, str], end_component: Union[int, str], 
                          **inputs: Any) -> Prediction:
        """Execute a subsystem in LLM-based mode."""
        # Convert indices to names
        if isinstance(start_component, int):
            start_component = self.execution_order[start_component]
        if isinstance(end_component, int):
            end_component = self.execution_order[end_component]

        context = dict(inputs)
        traj = {}

        start_idx = self.execution_order.index(start_component)
        end_idx = self.execution_order.index(end_component)
        sub_execution_order = self.execution_order[start_idx:end_idx + 1]

        for component_name in sub_execution_order:
            component = self.components[component_name]
            
            # Check required inputs
            missing_inputs = set(component.input_fields) - set(context.keys())
            if missing_inputs:
                raise ValueError(f"Component '{component_name}' missing inputs: {missing_inputs}")

            component_inputs = {key: context[key] for key in component.input_fields}
            
            # Execute component with or without RM filtering
            if self.rm is not None and component_name in self.components_to_apply:
                outputs = self._sample_and_rank(component_name, component, component_inputs)
            else:
                outputs = self._single_call(component, component_inputs)

            context.update(outputs)
            traj[component_name] = component.traj

        return Prediction(**outputs, traj=traj)

    def _single_call(self, component: BaseComponent, component_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Make a single call to a component."""
        return component(**component_inputs)

    def _sample_and_rank(self, component_name: str, component: BaseComponent, 
                        component_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Sample multiple outputs and rank them using the reward model."""
        # Generate output pool
        output_pool = self._generate_output_pool(component, component_inputs)
        output_pool = unique_objects(output_pool)

        if len(output_pool) <= 1:
            best_output = output_pool[0] if output_pool else {}
            component.traj = {
                "input": component_inputs,
                "output": best_output,
                "score": 1.0
            }
            return best_output

        # Evaluate with reward model
        batch_pool = [{**component_inputs, **out} for out in output_pool]
        scores = self.rm.batch_evaluate(component_name, batch_pool)

        # Sort by score (descending)
        sorted_indices = np.argsort(scores)[::-1]
        outputs_sorted = [output_pool[i] for i in sorted_indices]
        scores_sorted = [scores[i] for i in sorted_indices]

        best_output = outputs_sorted[0]
        best_score = scores_sorted[0]

        logger.info(f"Component {component_name} - Best score: {best_score:.3f}")

        component.traj = {
            "input": component_inputs,
            "output": best_output,
            "score": best_score
        }

        self._log_rollout(component_name, component_inputs, outputs_sorted, scores_sorted)
        return best_output

    def _generate_output_pool(self, component: BaseComponent, component_inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate a pool of outputs for ranking."""
        def parallel_sample(sample_size: int) -> List[Dict[str, Any]]:
            with ThreadPoolExecutor(max_workers=self.max_sample_workers) as executor:
                futures = [executor.submit(self._single_call, component, component_inputs) 
                          for _ in range(sample_size)]
                return [f.result() for f in as_completed(futures) if f.result() is not None]

        # Sample with temperature if supported
        if hasattr(component.config, 'temperature'):
            context = (nullcontext() if self.sample_temperature is None 
                      else component.context(temperature=self.sample_temperature))
            with context:
                return parallel_sample(self.sample_size)

        # Sample with variable search space if available
        elif hasattr(component, "variable_search_space") and component.variable_search_space:
            output_pool = []
            all_combinations = list(product(*component.variable_search_space.values()))
            for values in all_combinations:
                variable = dict(zip(component.variable_search_space.keys(), values))
                with component.context(variable=variable):
                    output_pool.extend(parallel_sample(1))
            return output_pool

        # Default single sample
        else:
            return parallel_sample(1)

    def _log_rollout(self, component_name: str, component_inputs: Dict[str, Any], 
                    outputs: List[Dict[str, Any]], scores: List[float]) -> None:
        """Log rollout data to CSV file."""
        if self.log_df is None:
            return

        new_row = pd.DataFrame({
            "component_name": [component_name],
            "component_inputs": [component_inputs],
            "outputs": [outputs],
            "scores": [scores]
        })
        
        try:
            self.log_df = pd.concat([self.log_df, new_row], ignore_index=True)
            self.log_df.to_csv(self.log_file, index=False)
        except Exception as e:
            logger.error(f"Failed to log rollout data: {e}")

    def _extract_context_from_traj(self, traj: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the final context from a trajectory dictionary."""
        context = {}
        for component_traj in traj.values():
            context.update(component_traj["input"])
            context.update(component_traj["output"])
        return context

    def _pick_best_candidate(self, component_name: str, candidates: List[Dict[str, Any]], 
                           context: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Evaluate candidates and return the best one with its score."""
        if len(candidates) == 1:
            return candidates[0], -1.0

        def score_candidate(candidate: Dict[str, Any]) -> float:
            if isinstance(self.rm, dict):
                return self.rm[component_name].evaluate(component_name, **{**context, **candidate})
            return self.rm.evaluate(component_name, **{**context, **candidate})

        scores = [score_candidate(candidate) for candidate in candidates]
        best_idx = np.argmax(scores)
        return candidates[best_idx], scores[best_idx]

    def evaluate(self, example: Example, prediction: Optional[Prediction] = None, 
                return_pred: bool = False, **kwargs) -> Union[float, Tuple[float, Prediction]]:
        """
        Evaluate a single example.
        
        Args:
            example: Reference example with ground truth
            prediction: Optional pre-computed prediction
            return_pred: Whether to return the prediction along with the score
            **kwargs: Additional arguments for evaluation function
            
        Returns:
            Evaluation score, or tuple of (score, prediction) if return_pred=True
        """
        if prediction is None:
            prediction = self(**{k: getattr(example, k) for k in self.required_input_fields})

        try:
            score = self.eval_func(
                **{k: getattr(prediction, k) for k in self.final_output_fields},
                **{k: getattr(example, k) for k in self.ground_fields},
                **self.external,
                **kwargs
            )
            return (score, prediction) if return_pred else score
        except Exception as e:
            logger.error(f"Evaluation failed for example {example}: {e}")
            return (float("-inf"), prediction) if return_pred else float("-inf")

    def evaluate_multiple(self, examples: List[Example], predictions: Optional[List[Prediction]] = None,
                         return_pred: bool = False) -> List[Union[float, Tuple[float, Prediction]]]:
        """
        Evaluate multiple examples in parallel.
        
        Args:
            examples: List of reference examples
            predictions: Optional list of pre-computed predictions
            return_pred: Whether to return predictions along with scores
            
        Returns:
            List of evaluation scores or (score, prediction) tuples
        """
        task_args = (
            [(ex, pred, return_pred) for ex, pred in zip(examples, predictions)]
            if predictions is not None
            else [(ex, None, return_pred) for ex in examples]
        )
        
        # Adjust worker count for RM-based evaluation
        max_workers = self.max_eval_workers
        if (self.rm is not None and predictions is None and 
            (self.sample_size > 1 or any(hasattr(c, 'variable_search_space') and c.variable_search_space 
                                       for c in self.components.values()))):
            logger.warning("Using single worker for RM-based evaluation to avoid conflicts")
            max_workers = 1

        return run_parallel_tasks(
            self.evaluate,
            task_args,
            max_workers=max_workers,
            task_desc="Evaluating examples"
        )
    
    