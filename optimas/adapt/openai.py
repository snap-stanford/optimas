"""OpenAIAgent adapter for optimas framework.

This provides functionality to create optimas BaseComponent instances
from OpenAI Agent instances.
"""

import asyncio
import warnings
from typing import List

from optimas.arch.base import BaseComponent
from optimas.adapt.utils import format_input_fields

# Attempt to import agents as an optional dependency
try:
    import agents
    from agents import Agent, Runner
except ImportError as e:
    agents = None
    warnings.warn(
        f"Optional dependency missing: {e.name}. "
        "OpenAI Agent SDK support will be disabled. "
        "Install via: pip install openai-agents",
        ImportWarning,
    )


def create_component_from_openai(
    agent: Agent,
    input_fields: List[str],
    output_fields: List[str]
) -> BaseComponent:
    """Create a BaseComponent from a OpenAIAgent Agent.

    This function wraps a OpenAIAgent Agent instance as a BaseComponent,
    enabling the use of OpenAIAgent agents within the optimas framework system.

    Args:
        agent: A OpenAIAgent Agent instance to be wrapped.
        input_fields: List of input field names that the component will accept.
        output_fields: List of output field names that the component will produce.
            Currently must contain exactly one field name.

    Returns:
        An initialized BaseComponent instance that wraps the OpenAIAgent agent.

    Raises:
        ImportError: If OpenAIAgent dependencies are not installed.
        ValueError: If output_fields contains more than one field (current limitation).

    Example:
        >>> import agents
        >>> agent = agents.Agent(
        ...     name="MyAgent",
        ...     model="gpt-4o",
        ...     instructions="You are a helpful assistant."
        ... )
        >>> component = create_component_from_openai(agent, ["input"], ["output"])
    """
    if agents is None:
        raise ImportError(
            "OpenAIAgent support requires the `agents` package. "
            "Please install it with: pip install agents"
        )

    if len(output_fields) != 1:
        raise ValueError(
            f"OpenAIAgent adapter currently supports exactly one output field, "
            f"but {len(output_fields)} were provided: {output_fields}"
        )

    async def _run_agent(agent: Agent, task: str) -> str:
        """Execute the agent asynchronously and extract response content.
        
        Args:
            agent: The OpenAI agent to run.
            task: The task string to send to the agent.
            
        Returns:
            The content of the agent's response message.
        """
        result = await Runner.run(agent, input=task)
        return result.final_output

    class OpenAIAgentModule(BaseComponent):
        """Dynamic BaseComponent implementation for OpenAIAgent agents."""
        
        name = f"{getattr(agent, 'name', 'OpenAIAgentAgent')}Module"
        qualname = name

        def __init__(self):
            """Initialize the component with OpenAIAgent agent configuration."""
            # Extract agent properties for component initialization
            instructions = getattr(agent, 'instructions', f"OpenAIAgent agent: {agent.name}")
            model_config = {"model": getattr(agent, 'model', None)}

            # Initialize parent BaseComponent
            super().__init__(
                description=instructions,
                input_fields=input_fields,
                output_fields=output_fields,
                variable=instructions,
                config=model_config,
            )
            self.agent = agent

        def forward(self, **inputs) -> dict:
            """Execute the OpenAIAgent agent with the given inputs.
            
            Args:
                **inputs: Input arguments matching the component's input fields.
                
            Returns:
                dict: Output dictionary with the agent's response.
            """
            # Format inputs into a task string
            task_str = format_input_fields(**inputs)
            
            # Update agent's instructions with current variable
            self.agent.instructions = self.variable

            # Run the agent asynchronously and get response
            output_content = asyncio.run(_run_agent(self.agent, task_str))
            
            # Return response mapped to the specified output field
            return {output_fields[0]: output_content}

    # Return initialized component instance
    return OpenAIAgentModule()