from typing import Dict, Any


def get_context_from_traj(traj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a flattened context dictionary from a trajectory consisting of component inputs and outputs.

    Args:
        traj (Dict[str, Any]): A trajectory dictionary where each key is a component name and each value
            contains optional 'input' and 'output' dictionaries.

    Returns:
        Dict[str, Any]: A single dictionary containing all key-value pairs from inputs and outputs
                        of all components in the trajectory. Later outputs override earlier inputs if keys overlap.
    """
    context: Dict[str, Any] = {}

    # First gather inputs
    for component_name, record in traj.items():
        inputs = record.get('input', {})
        context.update(inputs)

    # Then gather outputs (overrides inputs if keys overlap)
    for component_name, record in traj.items():
        outputs = record.get('output', {})
        context.update(outputs)

    return context
