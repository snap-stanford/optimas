import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from contextlib import contextmanager
import random
import time

import ast
import traceback
import inspect
import functools

from optimas.utils.logger import setup_logger
from optimas.wrappers.prediction import Prediction

logger = setup_logger(__name__)


class BaseComponent:
    """
    A reusable, thread-safe base class for defining components in compound AI systems.

    Supports modular configuration, optimizable variables, context-aware overrides,
    and trajectory logging. Designed to integrate into end-to-end optimization pipelines.
    """

    def __init__(
        self,
        description: str,
        input_fields: Optional[List[str]] = None,
        output_fields: Optional[List[str]] = None,
        variable: Optional[Any] = None,
        variable_search_space: Optional[Dict[str, List[Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
        num_retry: int = 100,
        retry_after: int = 3
    ):
        """
        Initialize a BaseComponent instance.

        Args:
            description: Human-readable description of the component.
            input_fields: List of expected input field names.
            output_fields: List of output field names this component produces.
            variable: Optimizable variable (e.g., prompt, parameters, or policy).
            variable_search_space: Search space for randomized optimization.
            config: Configuration parameters (e.g., model, temperature).
            num_retry: Max retry attempts on forward failure (currently unused).
        """
        self.description = description
        self.input_fields = input_fields or []
        self.output_fields = output_fields or []
        self.variable_search_space = variable_search_space
        self.traj = {}

        self.default_config = SimpleNamespace(**(config or {}))
        self.default_config = SimpleNamespace(
            **{**vars(self.default_config), "randomize_variable": False}
        )
        self.config_keys = list(vars(self.default_config).keys())
        self.num_retry = num_retry
        self.retry_after = retry_after
        
        # Initialize variable (random if unspecified and search space is provided)
        if self.variable_search_space and variable is None:
            self._default_variable = {
                key: random.choice(value)
                for key, value in self.variable_search_space.items()
            }
            logger.debug(f"Initialized variable from search space.")
        else:
            self._default_variable = variable

        self._thread_local = threading.local()
        self._lock = threading.Lock()
        self._validation_local = threading.local()

        # Automatically execute variable update when the component is initialized
        self.on_variable_update_end()

    def forward(self, **inputs: Any) -> Dict[str, Any]:
        """Override this method to implement component-specific logic."""
        raise NotImplementedError("Subclasses must implement the forward() method.")

    @property
    def optimizable(self) -> bool:
        """Returns True if the component has an optimizable variable."""
        return self._default_variable is not None

    @property
    def variable(self):
        """Thread-local access to the current variable with usage tracking."""
        if not hasattr(self._thread_local, "variable"):
            with self._lock:
                self._thread_local.variable = self._default_variable
        
        # Track access if we're in forward() method
        if (hasattr(self, '_validation_local') and 
            hasattr(self._validation_local, 'tracking') and
            self._validation_local.tracking.get('in_forward', False)):
            self._validation_local.tracking['variable_accessed'] = True
        
        return self._thread_local.variable

    @property
    def config(self) -> SimpleNamespace:
        """Thread-local access to the current configuration with usage tracking."""
        if not hasattr(self._thread_local, "config"):
            with self._lock:
                self._thread_local.config = SimpleNamespace(**vars(self.default_config))
        
        # Track access if we're in forward() method
        if (hasattr(self, '_validation_local') and 
            hasattr(self._validation_local, 'tracking') and
            self._validation_local.tracking.get('in_forward', False)):
            self._validation_local.tracking['config_accessed'] = True
        
        return self._thread_local.config

    def on_variable_update_begin(self, new_variable):
        """Optional callback before variable update."""
        pass

    def on_variable_update_end(self):
        """Optional callback after variable update."""
        pass
        
    def update(self, new_variable: Any) -> None:
        """Replace the variable used in the component."""
        with self._lock:
            original_variable = self._default_variable
            self.on_variable_update_begin(new_variable)

            self._default_variable = new_variable
            if hasattr(self._thread_local, "variable"):
                self._thread_local.variable = new_variable
            logger.info(f"{self.__class__.__name__} variable updated: {original_variable} => {new_variable}")

            self.on_variable_update_end()

    def update_config(self, **kwargs) -> None:
        """Update configuration values for the current thread."""
        with self._lock:
            config_dict = vars(self.config).copy()
            for key, value in kwargs.items():
                if key in self.config_keys:
                    config_dict[key] = value
                    logger.debug(f"Updated config: {key} = {value}")
                else:
                    logger.error(f"Invalid config key: {key}")
                    raise ValueError(f"Invalid config key: {key}")
            self._thread_local.config = SimpleNamespace(**config_dict)

    @contextmanager
    def context(self, variable=None, **kwargs):
        """
        Temporarily override variable/config values within a context.

        Args:
            variable: Optional override for the variable.
            **kwargs: Config overrides (e.g., temperature=0.8).
        """
        original_config = vars(self.config).copy()
        original_variable = getattr(self._thread_local, "variable", None)
        original_default_config = vars(self.default_config).copy()

        # Variable override
        if variable is not None and self._default_variable is not None:
            with self._lock:
                if isinstance(variable, dict) and isinstance(self._default_variable, dict):
                    var_copy = original_variable.copy() if isinstance(original_variable, dict) else {}
                    var_copy.update(variable)
                    self._thread_local.variable = var_copy
                    self._default_variable = var_copy
                else:
                    self._thread_local.variable = variable
                    self._default_variable = variable
                logger.debug(f"Context variable override applied.")

        # Config override
        self.update_config(**kwargs)
        if kwargs:
            with self._lock:
                new_config = vars(self.default_config).copy()
                for key, value in kwargs.items():
                    if key in self.config_keys:
                        new_config[key] = value
                self.default_config = SimpleNamespace(**new_config)

        try:
            yield self
        finally:
            with self._lock:
                self._thread_local.config = SimpleNamespace(**original_config)
                if original_variable is not None:
                    self._thread_local.variable = original_variable
                self._default_variable = original_variable or self._default_variable
                self.default_config = SimpleNamespace(**original_default_config)
                logger.debug(f"Context exited and state restored.")
    
    def _validated_forward(self, **inputs):
        """Wrapper for forward() that includes validation."""
        decorated_forward = self._validate_component_usage(self.forward)
        return decorated_forward(**inputs)

    def _validate_component_usage(self, forward_func):
        """
        Decorator that validates self.variable and self.config are used in forward() method.
        Thread-safe and handles race conditions using thread-local storage.
        """
        @functools.wraps(forward_func)
        def wrapper(**inputs):
            # Thread-local tracking to avoid race conditions
            if not hasattr(self._validation_local, 'tracking'):
                self._validation_local.tracking = {
                    'variable_accessed': False,
                    'config_accessed': False,
                    'in_forward': False
                }
            
            # Reset tracking for this call
            tracking = self._validation_local.tracking
            tracking['variable_accessed'] = False
            tracking['config_accessed'] = False
            tracking['in_forward'] = True
            
            try:
                # Execute the forward method
                result = forward_func(**inputs)
                
                # Validate usage after forward completes (merged logic)
                warnings = []
                
                # Check if variable should have been accessed
                if (self._default_variable is not None and 
                    not tracking.get('variable_accessed', False)):
                    warnings.append("self.variable is defined but was not accessed in forward()")
                
                # Check if config should have been accessed (only warn if config has meaningful values)
                config_dict = vars(self.default_config)
                has_meaningful_config = any(
                    key != 'randomize_variable' and value is not None 
                    for key, value in config_dict.items()
                )
                
                if (has_meaningful_config and 
                    not tracking.get('config_accessed', False)):
                    warnings.append("self.config contains values but was not accessed in forward()")
                
                # Issue warnings
                if warnings:
                    for warning in warnings:
                        logger.warning(f"Component {self.__class__.__name__}: {warning}")
                
                return result
            finally:
                # Clean up tracking state
                tracking['in_forward'] = False
        
        return wrapper
        
    def __call__(self, **inputs: Any) -> Dict[str, Any]:
        if not isinstance(inputs, dict):
            logger.error(f"[{self.__class__.__name__}] Inputs must be a dictionary, got {type(inputs)}")
            raise ValueError("Inputs must be a dictionary.")

        current_config = self.config
        current_variable = self.variable
        temp_variable = None
        random_variable = None

        # Use consistent attribute name
        if getattr(current_config, 'randomize_variable', False) and self.variable_search_space:
            random_variable = {
                key: random.choice(value)
                for key, value in self.variable_search_space.items()
            }
            temp_variable = current_variable
            with self._lock:
                self._thread_local.variable = random_variable
            logger.info(f"Sampled random variable {random_variable} for this call.")

        outputs = None
        exception = None
        logger.debug(f"[CALL] {self.__class__.__name__}: Current variable: {self.variable}, config: {self.config}")
        
        try:
            for retry_count in range(self.num_retry):
                try:
                    outputs = self._validated_forward(**inputs)
                    break
                except Exception as e:
                    time.sleep(self.retry_after)
                    logger.error(f"[Retry={retry_count}] Error executing component: {e}\n{traceback.format_exc()}")
                    exception = e
                    if retry_count == self.num_retry - 1:
                        logger.error(f"Max retries reached. Unable to execute component.")
        finally:
            # Single restoration point - no duplicates
            if random_variable is not None and temp_variable is not None:
                with self._lock:
                    self._thread_local.variable = temp_variable

        if outputs is None and exception:
            raise exception

        # Determine which variable was actually used
        used_variable = random_variable if random_variable is not None else current_variable
        
        with self._lock:
            self.traj = {
                "input": inputs,
                "output": outputs,
                "variable": used_variable if isinstance(used_variable, dict) else None,
            }

        return outputs

    # ======================= GEPA Interface Methods =======================
    
    @property
    def gepa_optimizable_components(self) -> Dict[str, str]:
        """Return mapping of component_name -> optimizable_text for GEPA.
        
        This method identifies the text components that can be optimized by GEPA.
        Default implementation handles simple string variables and some dict cases.
        Override in subclasses for framework-specific text extraction.
        
        Returns:
            Dict mapping component names to their current text values
        """
        if self._default_variable is None:
            return {}
        
        if isinstance(self._default_variable, str):
            # Simple case: single text variable
            component_name = f"{self.__class__.__name__}_text"
            return {component_name: self._default_variable}
        
        elif isinstance(self._default_variable, dict):
            # Dict case: extract string values
            text_components = {}
            for key, value in self._default_variable.items():
                if isinstance(value, str):
                    text_components[key] = value
            return text_components
        
        else:
            # Fallback: convert to string representation
            component_name = f"{self.__class__.__name__}_variable"
            return {component_name: str(self._default_variable)}
    
    def apply_gepa_updates(self, updates: Dict[str, str]) -> None:
        """Apply GEPA-optimized text updates to component.
        
        This method receives optimized text from GEPA and applies it to the component.
        Default implementation handles simple cases. Override for framework-specific logic.
        
        Args:
            updates: Dict mapping component names to optimized text
        """
        if not updates:
            return
        
        logger.info(f"Applying GEPA updates to {self.__class__.__name__}: {list(updates.keys())}")
        
        current_components = self.gepa_optimizable_components
        
        if isinstance(self._default_variable, str):
            # Simple case: single text variable
            if len(updates) == 1:
                new_text = next(iter(updates.values()))
                self.update(new_text)
            else:
                logger.warning(f"Multiple updates provided for single-text component: {updates}")
                
        elif isinstance(self._default_variable, dict):
            # Dict case: update matching keys
            new_variable = self._default_variable.copy()
            updated_keys = []
            
            for component_name, new_text in updates.items():
                if component_name in new_variable:
                    new_variable[component_name] = new_text
                    updated_keys.append(component_name)
                else:
                    logger.warning(f"Unknown component '{component_name}' in updates")
            
            if updated_keys:
                self.update(new_variable)
                logger.info(f"Updated dict components: {updated_keys}")
        
        else:
            # Fallback: replace entire variable with first update
            if updates:
                new_text = next(iter(updates.values()))
                self.update(new_text)
                logger.warning(f"Fallback update applied to non-text variable")
    
    def extract_execution_trace(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract execution traces for GEPA reflection.
        
        This method extracts meaningful information from component execution
        that can be used for GEPA's reflection-based optimization.
        Override in subclasses to provide framework-specific trace data.
        
        Args:
            inputs: Component inputs
            outputs: Component outputs
            
        Returns:
            Dict containing trace information for reflection
        """
        trace_info = {
            "component_name": self.__class__.__name__,
            "variable_used": self.variable,
            "inputs_summary": self._summarize_data(inputs),
            "outputs_summary": self._summarize_data(outputs),
            "trajectory": getattr(self, 'traj', {})
        }
        
        # Add config information if meaningful
        config_dict = vars(self.config)
        meaningful_config = {
            k: v for k, v in config_dict.items() 
            if k != 'randomize_variable' and v is not None
        }
        if meaningful_config:
            trace_info["config"] = meaningful_config
        
        return trace_info
    
    def _summarize_data(self, data: Dict[str, Any], max_length: int = 200) -> Dict[str, str]:
        """Summarize data for trace logging."""
        summary = {}
        for key, value in data.items():
            value_str = str(value)
            if len(value_str) > max_length:
                summary[key] = value_str[:max_length] + "..."
            else:
                summary[key] = value_str
        return summary
    