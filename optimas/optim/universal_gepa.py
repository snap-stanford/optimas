"""Universal GEPA optimizer for any BaseComponent across frameworks.

This module provides a framework-agnostic GEPA optimizer that can optimize
any Optimas BaseComponent, automatically detecting the framework type and
applying appropriate optimization strategies.
"""

import random
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

from optimas.arch.base import BaseComponent
from optimas.wrappers.example import Example
from optimas.optim.gepa_adapter import OptimasGEPAAdapter
from optimas.optim.feedback_extractors import get_feedback_extractor
from optimas.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class GEPAOptimizationResult:
    """Result of GEPA optimization."""
    best_candidate: Dict[str, str]
    optimization_history: List[Dict[str, Any]]
    final_score: float
    total_evaluations: int
    framework_type: str
    optimized_components: List[str]


class UniversalGEPAOptimizer:
    """Universal GEPA optimizer for any BaseComponent.
    
    This optimizer automatically detects the component framework type and
    applies appropriate GEPA optimization strategies. It works with DSPy,
    CrewAI, OpenAI, LangChain, and any custom BaseComponent.
    """
    
    def __init__(
        self,
        reflection_lm: Optional[callable] = None,
        auto_budget: Optional[str] = None,
        max_metric_calls: Optional[int] = None,
        max_full_evals: Optional[int] = None,
        num_iters: Optional[int] = None,
        reflection_minibatch_size: int = 3,
        candidate_selection_strategy: str = "pareto",
        skip_perfect_score: bool = True,
        use_merge: bool = True,
        max_merge_invocations: int = 5,
        num_threads: int = 1,
        failure_score: float = 0.0,
        perfect_score: float = 1.0,
        log_dir: Optional[str] = None,
        track_stats: bool = False,
        use_wandb: bool = False,
        wandb_api_key: Optional[str] = None,
        wandb_init_kwargs: Optional[Dict] = None,
        track_best_outputs: bool = False,
        seed: int = 0,
        max_workers: int = 1
    ):
        """Initialize the Universal GEPA optimizer.
        
        Args:
            reflection_lm: Language model for reflection (required)
            auto_budget: Auto budget setting ('light', 'medium', 'heavy')
            max_metric_calls: Maximum metric calls (mutually exclusive with others)
            max_full_evals: Maximum full evaluations
            num_iters: Number of iterations
            reflection_minibatch_size: Size of reflection minibatches
            candidate_selection_strategy: 'pareto' or 'current_best'
            skip_perfect_score: Skip optimization if perfect score achieved
            use_merge: Use merge-based optimization
            max_merge_invocations: Maximum merge invocations
            num_threads: Number of threads for evaluation
            failure_score: Score for failed examples
            perfect_score: Perfect score threshold
            log_dir: Directory for logging
            track_stats: Track detailed statistics
            use_wandb: Use Weights & Biases logging
            wandb_api_key: W&B API key
            wandb_init_kwargs: W&B initialization kwargs
            track_best_outputs: Track best outputs
            seed: Random seed
            max_workers: Maximum parallel workers
        """
        # Validate budget configuration
        budget_args = [auto_budget, max_metric_calls, max_full_evals, num_iters]
        budget_count = sum(1 for arg in budget_args if arg is not None)
        
        if budget_count != 1:
            raise ValueError(
                "Exactly one budget parameter must be set: "
                f"auto_budget={auto_budget}, max_metric_calls={max_metric_calls}, "
                f"max_full_evals={max_full_evals}, num_iters={num_iters}"
            )
        
        if reflection_lm is None:
            raise ValueError("reflection_lm is required for GEPA optimization")
        
        self.reflection_lm = reflection_lm
        self.auto_budget = auto_budget
        self.max_metric_calls = max_metric_calls
        self.max_full_evals = max_full_evals
        self.num_iters = num_iters
        self.reflection_minibatch_size = reflection_minibatch_size
        self.candidate_selection_strategy = candidate_selection_strategy
        self.skip_perfect_score = skip_perfect_score
        self.use_merge = use_merge
        self.max_merge_invocations = max_merge_invocations
        self.num_threads = num_threads
        self.failure_score = failure_score
        self.perfect_score = perfect_score
        self.log_dir = log_dir
        self.track_stats = track_stats
        self.use_wandb = use_wandb
        self.wandb_api_key = wandb_api_key
        self.wandb_init_kwargs = wandb_init_kwargs or {}
        self.track_best_outputs = track_best_outputs
        self.seed = seed
        self.max_workers = max_workers
        self.rng = random.Random(seed)
    
    def optimize_component(
        self,
        component: BaseComponent,
        trainset: List[Example],
        valset: Optional[List[Example]] = None,
        metric_fn: Optional[callable] = None
    ) -> GEPAOptimizationResult:
        """Optimize a BaseComponent using GEPA.
        
        Args:
            component: BaseComponent to optimize
            trainset: Training examples
            valset: Validation examples (optional)
            metric_fn: Metric function (gold, pred, trace=None) -> float
            
        Returns:
            GEPAOptimizationResult with optimization details
        """
        logger.info(f"Starting GEPA optimization for {component.__class__.__name__}")
        
        # Detect framework type
        framework_type = self._detect_framework_type(component)
        logger.info(f"Detected framework type: {framework_type}")
        
        # Get optimizable components
        optimizable_components = component.gepa_optimizable_components
        if not optimizable_components:
            logger.warning(f"No optimizable components found for {component.__class__.__name__}")
            return GEPAOptimizationResult(
                best_candidate={},
                optimization_history=[],
                final_score=0.0,
                total_evaluations=0,
                framework_type=framework_type,
                optimized_components=[]
            )
        
        logger.info(f"Optimizable components: {list(optimizable_components.keys())}")
        
        # Use DSPy GEPA for DSPy components
        if framework_type == "dspy":
            return self._optimize_dspy_component(component, trainset, valset, metric_fn)
        
        # Use universal adapter for other frameworks
        return self._optimize_with_universal_adapter(
            component, trainset, valset, metric_fn, framework_type, optimizable_components
        )
    
    def _detect_framework_type(self, component: BaseComponent) -> str:
        """Detect the framework type of a component."""
        class_name = component.__class__.__name__.lower()
        
        if hasattr(component, 'signature_cls'):
            return "dspy"
        elif 'crewai' in class_name:
            return "crewai"
        elif 'openai' in class_name:
            return "openai"
        elif 'langchain' in class_name:
            return "langchain"
        elif hasattr(component, 'agent'):
            # More specific detection based on agent properties
            if hasattr(component.agent, 'role') and hasattr(component.agent, 'backstory'):
                return "crewai"
            elif hasattr(component.agent, 'instructions') and hasattr(component.agent, 'model'):
                return "openai"
        
        return "generic"
    
    def _optimize_dspy_component(
        self,
        component: BaseComponent,
        trainset: List[Example],
        valset: Optional[List[Example]],
        metric_fn: Optional[callable]
    ) -> GEPAOptimizationResult:
        """Optimize DSPy component using native DSPy GEPA."""
        try:
            import dspy
            from dspy.teleprompt.gepa import GEPA
        except ImportError:
            raise ImportError("DSPy must be installed to optimize DSPy components with GEPA")
        
        logger.info("Using native DSPy GEPA optimization")
        
        # Create GEPA instance with current settings
        gepa_kwargs = {
            'metric': metric_fn or self._create_default_metric(component),
            'reflection_minibatch_size': self.reflection_minibatch_size,
            'candidate_selection_strategy': self.candidate_selection_strategy,
            'reflection_lm': self._wrap_reflection_lm_for_dspy(),
            'skip_perfect_score': self.skip_perfect_score,
            'use_merge': self.use_merge,
            'max_merge_invocations': self.max_merge_invocations,
            'num_threads': self.num_threads,
            'failure_score': self.failure_score,
            'perfect_score': self.perfect_score,
            'log_dir': self.log_dir,
            'track_stats': self.track_stats,
            'use_wandb': self.use_wandb,
            'wandb_api_key': self.wandb_api_key,
            'wandb_init_kwargs': self.wandb_init_kwargs,
            'track_best_outputs': self.track_best_outputs,
            'seed': self.seed
        }
        
        # Set budget parameter
        if self.auto_budget:
            gepa_kwargs['auto'] = self.auto_budget
        elif self.max_metric_calls:
            gepa_kwargs['max_metric_calls'] = self.max_metric_calls
        elif self.max_full_evals:
            gepa_kwargs['max_full_evals'] = self.max_full_evals
        elif self.num_iters:
            gepa_kwargs['num_iters'] = self.num_iters
        
        gepa = GEPA(**gepa_kwargs)
        
        # Wrap component as DSPy module if needed
        if hasattr(component, 'signature_cls'):
            dspy_module = dspy.Predict(component.signature_cls.with_instructions(component.variable))
        else:
            # Create a simple DSPy wrapper
            raise NotImplementedError("DSPy component optimization requires signature_cls")
        
        # Run optimization
        optimized_module = gepa.compile(dspy_module, trainset=trainset, valset=valset)
        
        # Extract results
        if hasattr(optimized_module, 'detailed_results'):
            detailed_results = optimized_module.detailed_results
            best_candidate = detailed_results.best_candidate
            final_score = max(detailed_results.val_aggregate_scores)
            total_evaluations = detailed_results.total_metric_calls or 0
        else:
            best_candidate = {'instructions': optimized_module.signature.instructions}
            final_score = 0.0
            total_evaluations = 0
        
        # Apply updates to original component
        component.apply_gepa_updates(best_candidate)
        
        return GEPAOptimizationResult(
            best_candidate=best_candidate,
            optimization_history=[],
            final_score=final_score,
            total_evaluations=total_evaluations,
            framework_type="dspy",
            optimized_components=list(best_candidate.keys())
        )
    
    def _optimize_with_universal_adapter(
        self,
        component: BaseComponent,
        trainset: List[Example],
        valset: Optional[List[Example]],
        metric_fn: Optional[callable],
        framework_type: str,
        optimizable_components: Dict[str, str]
    ) -> GEPAOptimizationResult:
        """Optimize component using universal GEPA adapter."""
        try:
            import gepa
        except ImportError:
            raise ImportError("GEPA package must be installed for universal optimization")
        
        logger.info("Using universal GEPA adapter optimization")
        
        # Create metric function if not provided
        if metric_fn is None:
            metric_fn = self._create_default_metric(component)
        
        # Create feedback extractor for framework
        feedback_extractor = get_feedback_extractor(framework_type)
        
        # Create universal adapter
        adapter = OptimasGEPAAdapter(
            component=component,
            metric_fn=metric_fn,
            feedback_extractor=feedback_extractor,
            max_workers=self.max_workers,
            rng=self.rng
        )
        
        # Calculate budget
        if self.auto_budget:
            # Simple budget calculation
            budget_map = {'light': 50, 'medium': 100, 'heavy': 200}
            calculated_budget = budget_map.get(self.auto_budget, 100)
        elif self.max_metric_calls:
            calculated_budget = self.max_metric_calls
        elif self.max_full_evals:
            calculated_budget = self.max_full_evals * len(trainset)
        elif self.num_iters:
            calculated_budget = None  # Use num_iters instead
        
        # Run GEPA optimization
        gepa_kwargs = {
            'seed_candidate': optimizable_components,
            'trainset': trainset,
            'valset': valset,
            'adapter': adapter,
            'reflection_lm': self.reflection_lm,
            'candidate_selection_strategy': self.candidate_selection_strategy,
            'skip_perfect_score': self.skip_perfect_score,
            'reflection_minibatch_size': self.reflection_minibatch_size,
            'perfect_score': self.perfect_score,
            'use_merge': self.use_merge,
            'max_merge_invocations': self.max_merge_invocations,
            'logger': None,  # Use default logger
            'run_dir': self.log_dir,
            'use_wandb': self.use_wandb,
            'wandb_api_key': self.wandb_api_key,
            'wandb_init_kwargs': self.wandb_init_kwargs,
            'track_best_outputs': self.track_best_outputs,
            'seed': self.seed
        }
        
        # Set budget parameter
        if self.num_iters:
            gepa_kwargs['num_iters'] = self.num_iters
        else:
            gepa_kwargs['max_metric_calls'] = calculated_budget
        
        result = gepa.optimize(**gepa_kwargs)
        
        # Apply best candidate to component
        component.apply_gepa_updates(result.best_candidate)
        
        return GEPAOptimizationResult(
            best_candidate=result.best_candidate,
            optimization_history=[],  # Could extract from result if available
            final_score=max(result.val_aggregate_scores) if result.val_aggregate_scores else 0.0,
            total_evaluations=getattr(result, 'total_metric_calls', 0),
            framework_type=framework_type,
            optimized_components=list(result.best_candidate.keys())
        )
    
    def _create_default_metric(self, component: BaseComponent) -> callable:
        """Create a default metric function for the component."""
        def default_metric(gold: Example, pred, trace=None) -> float:
            # Simple exact match metric for demonstration
            # In practice, this should be more sophisticated
            try:
                gold_labels = gold.labels()
            except (ValueError, AttributeError):
                # Fallback: use all keys as labels
                gold_labels = gold
            
            # Compare outputs field by field
            total_score = 0.0
            field_count = 0
            
            for field in component.output_fields:
                if field in gold_labels and hasattr(pred, field):
                    gold_value = str(gold_labels[field]).strip().lower()
                    pred_value = str(getattr(pred, field)).strip().lower()
                    
                    if gold_value == pred_value:
                        total_score += 1.0
                    field_count += 1
            
            return total_score / max(field_count, 1)
        
        logger.warning("Using default exact match metric. Consider providing a custom metric function.")
        return default_metric
    
    def _wrap_reflection_lm_for_dspy(self) -> callable:
        """Wrap reflection LM for DSPy compatibility."""
        if hasattr(self.reflection_lm, '__call__'):
            def wrapped_lm(prompt):
                result = self.reflection_lm(prompt)
                # DSPy expects a list-like result
                if isinstance(result, str):
                    return [result]
                return result
            return wrapped_lm
        else:
            return self.reflection_lm