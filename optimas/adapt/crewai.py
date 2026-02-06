"""CrewAI adapter for optimas framework.

This provides functionality to create optimas BaseComponent instances
from CrewAI Agent instances.
"""

import warnings
from typing import List, Any, Type, Dict

from pydantic import BaseModel, create_model
from optimas.arch.base import BaseComponent
from optimas.adapt.utils import format_input_fields
from optimas.utils.logger import setup_logger

logger = setup_logger(__name__)

# Attempt to import crewai as an optional dependency
try:
    import crewai
except ImportError as e:
    crewai = None
    warnings.warn(
        f"Optional dependency missing: {e.name}. "
        "CrewAI support will be disabled. "
        "Install via: pip install crewai",
        ImportWarning,
    )


def create_component_from_crewai(
    agent: crewai.Agent,
    input_fields: List[str],
    output_fields: List[str]
) -> BaseComponent:
    """Create a BaseComponent from a CrewAI Agent.

    This function wraps a CrewAI Agent instance as a BaseComponent,
    enabling the use of CrewAI agents within the optimas framework system.
    The agent's responses are structured using Pydantic models for reliable parsing.

    Args:
        agent: A CrewAI Agent instance to be wrapped.
        input_fields: List of input field names that the component will accept.
        output_fields: List of output field names that the component will produce.

    Returns:
        An initialized BaseComponent instance that wraps the CrewAI agent.

    Raises:
        ImportError: If CrewAI dependencies are not installed.

    Example:
        >>> import crewai
        >>> agent = crewai.Agent(
        ...     role="Assistant",
        ...     goal="Help users with their queries",
        ...     backstory="You are a helpful assistant",
        ...     llm="gpt-4"
        ... )
        >>> component = create_component_from_crewai(agent, ["input"], ["output"])
    """
    if crewai is None:
        raise ImportError(
            "CrewAI support requires the `crewai` package. "
            "Please install it with: pip install crewai"
        )

    def _create_response_model(fields: List[str]) -> Type[BaseModel]:
        """Create a Pydantic model for structured response parsing.
        
        Args:
            fields: List of field names for the response model.
            
        Returns:
            A dynamically created Pydantic BaseModel class.
        """
        return create_model(
            "CrewAIResponse", 
            **{field: (str, ...) for field in fields}
        )

    # Create response model for output parsing
    response_model = _create_response_model(output_fields)

    class CrewAIModule(BaseComponent):
        """Dynamic BaseComponent implementation for CrewAI agents."""
        
        name = f"{getattr(agent, 'role', 'CrewAIAgent')}Module"
        qualname = name

        def __init__(self):
            """Initialize the component with CrewAI agent configuration."""
            # Extract agent properties for component initialization
            description = getattr(agent, 'goal')
            backstory = getattr(agent, 'backstory')

            # Initialize parent BaseComponent
            super().__init__(
                description=description,
                input_fields=input_fields,
                output_fields=output_fields,
                variable=backstory
            )
            self.agent = agent

        def forward(self, **inputs) -> dict:
            """Execute the CrewAI agent with the given inputs.
            
            Args:
                **inputs: Input arguments matching the component's input fields.
                
            Returns:
                dict: Output dictionary with structured agent response.
            """
            # Format inputs into a task string
            task_str = format_input_fields(**inputs)
            
            # Update agent's backstory with current variable
            self.agent.backstory = self.variable
            
            # Execute agent with structured response format
            result = self.agent.kickoff(
                messages=[{"role": "user", "content": task_str}],
                response_format=response_model,
            )
            
            # Extract and return structured data
            data = result.pydantic.model_dump()
            return {field: data.get(field) for field in output_fields}

        # ======================= GEPA Interface Methods =======================
        
        @property
        def gepa_optimizable_components(self) -> Dict[str, str]:
            """Return CrewAI-specific optimizable components."""
            components = {}
            
            # Add agent backstory as primary optimizable component
            if hasattr(self.agent, 'backstory') and self.agent.backstory:
                components['backstory'] = self.agent.backstory
            
            # Add agent goal if different from description
            if hasattr(self.agent, 'goal') and self.agent.goal:
                components['goal'] = self.agent.goal
                
            # Add agent role
            if hasattr(self.agent, 'role') and self.agent.role:
                components['role'] = self.agent.role
            
            # Add system message if available
            if hasattr(self.agent, 'system_message') and self.agent.system_message:
                components['system_message'] = self.agent.system_message
                
            return components
        
        def apply_gepa_updates(self, updates: Dict[str, str]) -> None:
            """Apply GEPA updates to CrewAI agent components."""
            if not updates:
                return
                
            logger.info(f"Applying GEPA updates to CrewAI agent: {list(updates.keys())}")
            
            # Update backstory (primary variable)
            if 'backstory' in updates:
                self.agent.backstory = updates['backstory']
                self.update(updates['backstory'])  # Update base component variable
                logger.info(f"Updated agent backstory")
            
            # Update goal
            if 'goal' in updates:
                self.agent.goal = updates['goal']
                logger.info(f"Updated agent goal")
            
            # Update role
            if 'role' in updates:
                self.agent.role = updates['role']
                logger.info(f"Updated agent role")
                
            # Update system message
            if 'system_message' in updates:
                if hasattr(self.agent, 'system_message'):
                    self.agent.system_message = updates['system_message']
                    logger.info(f"Updated agent system message")
        
        def extract_execution_trace(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
            """Extract CrewAI-specific execution traces."""
            trace_info = super().extract_execution_trace(inputs, outputs)
            
            # Add CrewAI-specific trace information
            trace_info.update({
                "framework": "crewai",
                "agent_role": getattr(self.agent, 'role', ''),
                "agent_goal": getattr(self.agent, 'goal', ''),
                "agent_backstory": getattr(self.agent, 'backstory', ''),
            })
            
            # Add tools information if available
            if hasattr(self.agent, 'tools') and self.agent.tools:
                trace_info["available_tools"] = [
                    getattr(tool, 'name', str(tool)) for tool in self.agent.tools
                ]
            
            # Add memory information if available  
            if hasattr(self.agent, 'memory') and self.agent.memory:
                trace_info["has_memory"] = True
                
            return trace_info

    # Return initialized component instance
    return CrewAIModule()
