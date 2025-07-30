"""AutoGen adapter for optimas framework.

This provides functionality to create optimas BaseComponent instances
from AutoGen AssistantAgent instances.
"""

import asyncio
import warnings
from typing import List, Any

from optimas.arch.base import BaseComponent
from optimas.adapt.utils import format_input_fields

# Attempt to import autogen as optional dependencies
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import OpenAIChatCompletionClient
except ImportError as e:
    AssistantAgent = None
    OpenAIChatCompletionClient = None
    warnings.warn(
        f"Optional dependency missing: {e.name}. "
        "AutoGen support will be disabled. "
        "Install via: pip install autogen_agentchat autogen_ext",
        ImportWarning,
    )


def create_component_from_autogen(
    agent: AssistantAgent, 
    input_fields: List[str], 
    output_fields: List[str]
) -> BaseComponent:
    """Create a BaseComponent from an AutoGen AssistantAgent.

    This function wraps an AutoGen AssistantAgent instance as a BaseComponent,
    enabling the use of AutoGen agents within the optimas framework system.

    Args:
        agent: An AutoGen AssistantAgent instance to be wrapped.
        input_fields: List of input field names that the component will accept.
        output_fields: List of output field names that the component will produce.
            Currently must contain exactly one field name.

    Returns:
        An initialized BaseComponent instance that wraps the AutoGen agent.

    Raises:
        ImportError: If AutoGen dependencies are not installed.
        ValueError: If output_fields contains more than one field (current limitation).

    Example:
        >>> from autogen_agentchat.agents import AssistantAgent
        >>> from autogen_ext.models.openai import OpenAIChatCompletionClient
        >>> 
        >>> model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
        >>> agent = AssistantAgent(
        ...     name="MyAgent",
        ...     model_client=model_client,
        ...     description="A helpful assistant",
        ...     system_message="You are a helpful assistant."
        ... )
        >>> component = create_component_from_autogen(agent, ["input"], ["output"])
    """
    if AssistantAgent is None:
        raise ImportError(
            "AutoGen support requires autogen_agentchat and autogen_ext. "
            "Please install them with: pip install autogen_agentchat autogen_ext"
        )

    if len(output_fields) != 1:
        raise ValueError(
            f"AutoGen adapter currently supports exactly one output field, "
            f"but {len(output_fields)} were provided: {output_fields}"
        )

    async def _run_agent(agent: AssistantAgent, task: str) -> str:
        """Execute the agent asynchronously and extract response content.
        
        Args:
            agent: The AutoGen agent to run.
            task: The task string to send to the agent.
            
        Returns:
            The content of the agent's response message.
        """
        result = await agent.run(task=task)
        return result.messages[-1].content

    class AutoGenModule(BaseComponent):
        """Dynamic BaseComponent implementation for AutoGen agents."""
        
        name = f"{agent.name}Module"
        qualname = name

        def __init__(self):
            """Initialize the component with AutoGen agent configuration."""
            
            # Get system message for variable initialization
            system_message = ""
            if agent._system_messages:
                system_message = agent._system_messages[0].content

            # Initialize parent BaseComponent
            super().__init__(
                description=agent.description or f"AutoGen agent: {agent.name}",
                input_fields=input_fields,
                output_fields=output_fields,
                variable=system_message
            )
            self.agent = agent

        def forward(self, **inputs) -> dict:
            """Execute the AutoGen agent with the given inputs.
            
            Args:
                **inputs: Input arguments matching the component's input fields.
                
            Returns:
                dict: Output dictionary with the agent's response.
            """
            # Format inputs into a task string
            task_str = format_input_fields(**inputs)
            
            # Update agent's system message with current variable
            if self.agent._system_messages:
                self.agent._system_messages[0].content = self.variable
            
            # Run the agent asynchronously and get response
            response = asyncio.run(_run_agent(self.agent, task_str))
            
            # Return response mapped to the specified output field
            return {output_fields[0]: response}

    # Return initialized component instance
    return AutoGenModule()