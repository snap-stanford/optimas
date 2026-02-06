"""OpenAIAgent adapter for optimas framework.

This provides functionality to create optimas BaseComponent instances
from OpenAI Agent instances.
"""

import asyncio
import warnings
from typing import List, Dict, Any

from optimas.arch.base import BaseComponent
from optimas.adapt.utils import format_input_fields
from optimas.utils.logger import setup_logger

logger = setup_logger(__name__)

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

        # ======================= GEPA Interface Methods =======================
        
        @property
        def gepa_optimizable_components(self) -> Dict[str, str]:
            """Return OpenAI Agent-specific optimizable components."""
            components = {}
            
            # Add agent instructions as primary optimizable component
            if hasattr(self.agent, 'instructions') and self.agent.instructions:
                components['instructions'] = self.agent.instructions
            
            # Add model-specific prompts if available
            if hasattr(self.agent, 'system_prompt') and self.agent.system_prompt:
                components['system_prompt'] = self.agent.system_prompt
                
            # Add function descriptions if available
            if hasattr(self.agent, 'functions') and self.agent.functions:
                function_descriptions = []
                for func in self.agent.functions:
                    if hasattr(func, 'description'):
                        function_descriptions.append(func.description)
                if function_descriptions:
                    components['function_descriptions'] = '\n'.join(function_descriptions)
                    
            return components
        
        def apply_gepa_updates(self, updates: Dict[str, str]) -> None:
            """Apply GEPA updates to OpenAI Agent components."""
            if not updates:
                return
                
            logger.info(f"Applying GEPA updates to OpenAI agent: {list(updates.keys())}")
            
            # Update instructions (primary variable)
            if 'instructions' in updates:
                self.agent.instructions = updates['instructions']
                self.update(updates['instructions'])  # Update base component variable
                logger.info(f"Updated agent instructions")
            
            # Update system prompt
            if 'system_prompt' in updates:
                if hasattr(self.agent, 'system_prompt'):
                    self.agent.system_prompt = updates['system_prompt']
                    logger.info(f"Updated agent system prompt")
                    
            # Update function descriptions (more complex - would need framework support)
            if 'function_descriptions' in updates:
                logger.info(f"Function description update requested (may require manual implementation)")
        
        def extract_execution_trace(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
            """Extract OpenAI Agent-specific execution traces."""
            trace_info = super().extract_execution_trace(inputs, outputs)
            
            # Add OpenAI-specific trace information
            trace_info.update({
                "framework": "openai",
                "agent_name": getattr(self.agent, 'name', ''),
                "agent_model": getattr(self.agent, 'model', ''),
                "agent_instructions": getattr(self.agent, 'instructions', ''),
            })
            
            # Add function information if available
            if hasattr(self.agent, 'functions') and self.agent.functions:
                trace_info["available_functions"] = [
                    getattr(func, 'name', str(func)) for func in self.agent.functions
                ]
            
            # Add model configuration if available
            if hasattr(self.agent, 'model_config'):
                trace_info["model_config"] = self.agent.model_config
                
            return trace_info

    # Return initialized component instance
    return OpenAIAgentModule()