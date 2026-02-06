"""Universal GEPA adapter for Optimas BaseComponent optimization.

This module provides a framework-agnostic GEPA adapter that can optimize
any Optimas BaseComponent, regardless of the underlying AI framework
(DSPy, CrewAI, OpenAI, etc.).
"""

import copy
import random
import traceback
from typing import Any, Dict, List, Optional, Protocol, TypeVar, Union
from dataclasses import dataclass

from optimas.arch.base import BaseComponent
from optimas.wrappers.example import Example
from optimas.wrappers.prediction import Prediction
from optimas.utils.logger import setup_logger
from optimas.utils.parallel import run_parallel_tasks

logger = setup_logger(__name__)

# Type variables for GEPA adapter
DataInst = TypeVar("DataInst")
Trajectory = TypeVar("Trajectory") 
RolloutOutput = TypeVar("RolloutOutput")


@dataclass
class ComponentTrace:
    """Execution trace for a single component execution."""
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    component_name: str
    variable_state: Any
    execution_time: float
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EvaluationBatch:
    """Container for batch evaluation results."""
    outputs: List[Dict[str, Any]]
    scores: List[float]
    trajectories: Optional[List[ComponentTrace]] = None


class FeedbackExtractor(Protocol):
    """Protocol for extracting feedback from component execution."""
    
    def extract_feedback(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        score: float,
        trace: Optional[ComponentTrace] = None,
        error: Optional[Exception] = None
    ) -> str:
        """Extract textual feedback from component execution.
        
        Args:
            inputs: Component inputs
            outputs: Component outputs  
            score: Evaluation score
            trace: Execution trace
            error: Any execution error
            
        Returns:
            Textual feedback string for GEPA reflection
        """
        ...


class DefaultFeedbackExtractor:
    """Default feedback extractor for BaseComponent."""
    
    def extract_feedback(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        score: float,
        trace: Optional[ComponentTrace] = None,
        error: Optional[Exception] = None
    ) -> str:
        """Extract basic feedback from component execution."""
        feedback_parts = [
            f"Score: {score:.3f}",
            f"Inputs: {self._format_inputs(inputs)}",
            f"Outputs: {self._format_outputs(outputs)}"
        ]
        
        if error:
            feedback_parts.append(f"Error: {str(error)}")
        
        if trace and trace.metadata:
            feedback_parts.append(f"Metadata: {trace.metadata}")
            
        return " | ".join(feedback_parts)
    
    def _format_inputs(self, inputs: Dict[str, Any]) -> str:
        """Format inputs for feedback."""
        formatted = []
        for key, value in inputs.items():
            value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            formatted.append(f"{key}={value_str}")
        return "{" + ", ".join(formatted) + "}"
    
    def _format_outputs(self, outputs: Dict[str, Any]) -> str:
        """Format outputs for feedback."""
        return self._format_inputs(outputs)  # Same formatting logic


class OptimasGEPAAdapter:
    """Universal GEPA adapter for Optimas BaseComponent optimization.
    
    This adapter enables GEPA optimization for any BaseComponent by:
    1. Managing component variable states during evaluation
    2. Executing components on batches of data
    3. Collecting execution traces and feedback
    4. Creating reflective datasets for GEPA optimization
    """
    
    def __init__(
        self,
        component: BaseComponent,
        metric_fn: callable,
        feedback_extractor: Optional[FeedbackExtractor] = None,
        max_workers: int = 1,
        capture_detailed_traces: bool = True,
        rng: Optional[random.Random] = None
    ):
        """Initialize the GEPA adapter.
        
        Args:
            component: BaseComponent to optimize
            metric_fn: Metric function (gold, pred, trace=None) -> float
            feedback_extractor: Custom feedback extractor
            max_workers: Number of parallel workers for evaluation
            capture_detailed_traces: Whether to capture detailed execution traces
            rng: Random number generator for reproducibility
        """
        self.component = component
        self.metric_fn = metric_fn
        self.feedback_extractor = feedback_extractor or DefaultFeedbackExtractor()
        self.max_workers = max_workers
        self.capture_detailed_traces = capture_detailed_traces
        self.rng = rng or random.Random()
        
        # GEPA requires this attribute (can be None for default behavior)
        self.propose_new_texts = None
        
        # Validate component has GEPA interface
        if not hasattr(component, 'gepa_optimizable_components'):
            logger.warning(
                f"Component {component.__class__.__name__} lacks gepa_optimizable_components. "
                f"Using fallback implementation."
            )
    
    def evaluate(
        self,
        batch: List[Example],
        candidate: Dict[str, str],
        capture_traces: bool = False
    ) -> EvaluationBatch:
        """Evaluate a candidate on a batch of examples.
        
        Args:
            batch: List of examples to evaluate
            candidate: Mapping from component_name -> component_text
            capture_traces: Whether to capture execution traces
            
        Returns:
            EvaluationBatch with outputs, scores, and optional traces
        """
        logger.debug(f"Evaluating candidate on batch of {len(batch)} examples")
        
        # Apply candidate to component
        original_state = self._backup_component_state()
        try:
            self._apply_candidate_to_component(candidate)
            
            # Prepare evaluation tasks
            task_args = [(self.component, example, capture_traces) for example in batch]
            
            # Execute in parallel
            results = run_parallel_tasks(
                task_func=self._evaluate_single_example,
                task_args=task_args,
                max_workers=self.max_workers,
                task_desc=f"Evaluating {len(batch)} examples"
            )
            
            # Process results
            outputs = []
            scores = []
            traces = [] if capture_traces else None
            
            for i, (example, result) in enumerate(zip(batch, results)):
                if result is None:
                    # Handle failed evaluation
                    outputs.append({})
                    scores.append(0.0)
                    if capture_traces:
                        traces.append(ComponentTrace(
                            inputs=example.inputs(),
                            outputs={},
                            component_name=self.component.__class__.__name__,
                            variable_state=self._get_component_variable_state(),
                            execution_time=0.0,
                            error=Exception("Evaluation failed")
                        ))
                else:
                    pred_dict, score, trace = result
                    outputs.append(pred_dict)
                    scores.append(score)
                    if capture_traces:
                        traces.append(trace)
            
            return EvaluationBatch(
                outputs=outputs,
                scores=scores,
                trajectories=traces
            )
            
        finally:
            # Restore original component state
            self._restore_component_state(original_state)
    
    def _evaluate_single_example(
        self, 
        component: BaseComponent, 
        example: Example, 
        capture_traces: bool
    ) -> Optional[tuple]:
        """Evaluate a single example and return (outputs, score, trace)."""
        import time
        
        start_time = time.time()
        trace = None
        
        try:
            # Execute component
            inputs = example.inputs()
            pred_dict = component(**inputs)
            execution_time = time.time() - start_time
            
            # Create prediction object
            pred = Prediction(**pred_dict)
            
            # Calculate score
            score = self.metric_fn(example, pred)
            if not isinstance(score, (int, float)):
                score = float(score)
            
            # Create trace if requested
            if capture_traces:
                trace = ComponentTrace(
                    inputs=inputs,
                    outputs=pred_dict,
                    component_name=component.__class__.__name__,
                    variable_state=self._get_component_variable_state(),
                    execution_time=execution_time,
                    metadata=getattr(component, 'traj', {})
                )
            
            return pred_dict, score, trace
            
        except Exception as e:
            logger.warning(f"Example evaluation failed: {e}")
            execution_time = time.time() - start_time
            
            if capture_traces:
                trace = ComponentTrace(
                    inputs=example.inputs() if hasattr(example, 'inputs') else {},
                    outputs={},
                    component_name=component.__class__.__name__,
                    variable_state=self._get_component_variable_state(),
                    execution_time=execution_time,
                    error=e
                )
                return {}, 0.0, trace
            
            return None
    
    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Create reflective dataset for GEPA optimization.
        
        Args:
            candidate: Current candidate mapping
            eval_batch: Results from evaluate() with capture_traces=True
            components_to_update: List of component names to update
            
        Returns:
            Dict mapping component_name -> list of reflective examples
        """
        logger.debug(f"Creating reflective dataset for components: {components_to_update}")
        
        reflective_data = {}
        
        for component_name in components_to_update:
            examples = []
            
            # Process each example in the batch
            for i, (output, score, trace) in enumerate(
                zip(eval_batch.outputs, eval_batch.scores, eval_batch.trajectories or [])
            ):
                # Extract feedback for this example
                feedback = self.feedback_extractor.extract_feedback(
                    inputs=trace.inputs if trace else {},
                    outputs=output,
                    score=score,
                    trace=trace,
                    error=trace.error if trace else None
                )
                
                # Create reflective example
                reflective_example = {
                    "Inputs": trace.inputs if trace else {},
                    "Generated Outputs": output,
                    "Feedback": feedback,
                    "Score": score,
                    "Component": component_name,
                    "Current Text": candidate.get(component_name, "")
                }
                
                # Add trace metadata if available
                if trace and trace.metadata:
                    reflective_example["Trace Metadata"] = trace.metadata
                
                examples.append(reflective_example)
            
            reflective_data[component_name] = examples
        
        return reflective_data
    
    def _backup_component_state(self) -> Dict[str, Any]:
        """Backup current component state."""
        return {
            'variable': copy.deepcopy(self.component._default_variable),
            'traj': copy.deepcopy(getattr(self.component, 'traj', {}))
        }
    
    def _restore_component_state(self, state: Dict[str, Any]):
        """Restore component state from backup."""
        self.component._default_variable = state['variable']
        if hasattr(self.component, 'traj'):
            self.component.traj = state['traj']
        
        # Trigger component update
        if hasattr(self.component, 'on_variable_update_end'):
            self.component.on_variable_update_end()
    
    def _apply_candidate_to_component(self, candidate: Dict[str, str]):
        """Apply candidate text to component."""
        if hasattr(self.component, 'apply_gepa_updates'):
            self.component.apply_gepa_updates(candidate)
        else:
            # Fallback: assume single optimizable variable
            if len(candidate) == 1:
                component_name, text = next(iter(candidate.items()))
                self.component.update(text)
            else:
                logger.warning(
                    f"Component {self.component.__class__.__name__} has multiple "
                    f"candidate texts but no apply_gepa_updates method"
                )
    
    def _get_component_variable_state(self) -> Any:
        """Get current component variable state."""
        return copy.deepcopy(self.component.variable)