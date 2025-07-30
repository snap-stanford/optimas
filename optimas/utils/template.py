import json
from typing import Dict, Any


def apply_reward_template(input_dict: Dict[str, Any], output_dict: Dict[str, Any], desc: str) -> str:
    """
    Format a reward evaluation prompt based on input/output dictionaries and a component description.

    Args:
        input_dict (Dict[str, Any]): Input key-value pairs to the component.
        output_dict (Dict[str, Any]): Output key-value pairs from the component.
        desc (str): Description of the component task.

    Returns:
        str: A formatted evaluation prompt string.

    Example:
        >>> input_dict = {"input1": "value1", "input2": "value2"}
        >>> output_dict = {"output1": "value1", "output2": {"output2_1": "value1", "output2_2": "value2"}}
        >>> desc = "Do something"
        >>> apply_reward_template(input_dict, output_dict, desc)
        'You are an evaluator for a module.\\nModule task: Do something\\n\\nGiven the inputs:\\n- input1: value1\\n- input2: value2\\n\\nEvaluate its outputs:\\n- output1: value1\\n- output2: {...'
    """

    def format_io_dict(io_dict: Dict[str, Any]) -> str:
        """
        Convert a dictionary into a multi-line string for readability.

        Args:
            io_dict (Dict[str, Any]): Dictionary to format.

        Returns:
            str: Multi-line string with formatted key-value pairs.
        """
        formatted = []
        for key, value in io_dict.items():
            if isinstance(value, (dict, list)):
                try:
                    value = json.dumps(value, indent=2)
                except Exception:
                    value = str(value)
            formatted.append(f"- {key}: {value}")
        return "\n".join(formatted)

    return (
        f"You are an evaluator for a module.\n"
        f"Module task: {desc}\n\n"
        f"Given the inputs:\n{format_io_dict(input_dict)}\n\n"
        f"Evaluate its outputs:\n{format_io_dict(output_dict)}"
    )
