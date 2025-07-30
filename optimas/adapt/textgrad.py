"""TextGrad adapter for optimas framework.

This provides functionality to create optimas BaseComponent instances
from TextGrad BlackboxLLM instances.
"""

import warnings
from typing import List

from optimas.arch.base import BaseComponent
from optimas.adapt.utils import format_input_fields

# Attempt to import textgrad as an optional dependency
try:
    import textgrad as tg
except ImportError as e:
    tg = None
    warnings.warn(
        f"Optional dependency missing: {e.name}. "
        "TextGrad support will be disabled. "
        "Install via: pip install textgrad",
        ImportWarning,
    )


def create_component_from_textgrad(
    blackboxllm: tg.BlackboxLLM,
    input_fields: List[str],
    output_fields: List[str]
) -> BaseComponent:
    """Create a BaseComponent from a TextGrad BlackboxLLM.

    This function wraps a TextGrad BlackboxLLM instance as a BaseComponent,
    enabling the use of TextGrad models within the optimas framework system.

    Args:
        blackboxllm: A TextGrad BlackboxLLM instance to be wrapped.
        input_fields: List of input field names that the component will accept.
        output_fields: List of output field names that the component will produce.
            Currently must contain exactly one field name.

    Returns:
        An initialized BaseComponent instance that wraps the TextGrad agent.

    Raises:
        ImportError: If TextGrad dependencies are not installed.
        ValueError: If output_fields contains more than one field (current limitation).

    Example:
        >>> import textgrad as tg
        >>> blackboxllm = tg.BlackboxLLM(
        ...     engine="gpt-4o",
        ...     system_prompt="You are a helpful assistant."
        ... )
        >>> component = create_component_from_textgrad(blackboxllm, ["input"], ["output"])
    """
    if tg is None:
        raise ImportError(
            "TextGrad support requires the `textgrad` package. "
            "Please install it with: pip install textgrad"
        )

    if len(output_fields) != 1:
        raise ValueError(
            f"TextGrad adapter currently supports exactly one output field, "
            f"but {len(output_fields)} were provided: {output_fields}"
        )

    # Extract system prompt information
    system_prompt = getattr(blackboxllm, 'system_prompt', None)
    prompt_value = system_prompt.value if system_prompt else None
    description = prompt_value.strip() if prompt_value else "No description provided."

    class TextGradModule(BaseComponent):
        """Dynamic BaseComponent implementation for TextGrad BlackboxLLM."""
        
        name = f"{type(blackboxllm).__name__}Module"
        qualname = name

        def __init__(self):
            """Initialize the component with TextGrad agent configuration."""
            # Initialize parent BaseComponent
            super().__init__(
                description=description,
                input_fields=input_fields,
                output_fields=output_fields,
                variable=prompt_value,
            )
            self.blackboxllm = blackboxllm

        def forward(self, **inputs) -> dict:
            """Execute the TextGrad agent with the given inputs.
            
            Args:
                **inputs: Input arguments matching the component's input fields.
                
            Returns:
                dict: Output dictionary with the agent's response.
            """
            # Format inputs into a task string
            task_str = format_input_fields(**inputs)
            
            # Update system prompt with current variable if available
            if hasattr(self.blackboxllm.llm_call, 'system_prompt'):
                self.blackboxllm.llm_call.system_prompt.value = self.variable

            # Create TextGrad Variable for input
            input_variable = tg.Variable(
                task_str,
                role_description="input to the LLM",
                requires_grad=False
            )
            
            # Execute blackboxllm and get response
            output_variable = self.blackboxllm(input_variable)
            
            # Return response mapped to the specified output field
            return {output_fields[0]: output_variable.value}

    # Return initialized component instance
    return TextGradModule()