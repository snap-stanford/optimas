"""DSPy adapter for optimas framework.

This provides functionality to create optimas BaseComponent instances
from DSPy signature classes, enabling seamless integration between the
two frameworks.
"""

import copy
import warnings
from typing import Type, Any

from optimas.arch.base import BaseComponent

# Attempt to import dspy as an optional dependency
try:
    import dspy
except ImportError:
    dspy = None
    warnings.warn(
        "Optional dependency `dspy` not found. "
        "For full functionality install it via:\n"
        "    pip install dspy",
        ImportWarning,
    )

def create_component_from_dspy(signature_cls: Type[dspy.Signature]) -> Type[BaseComponent]:
    """
    Dynamically create a BaseComponent subclass from a dspy.Signature class.
    
    The resulting component integrates with the OPTIMAS pipeline, automatically extracting
    input/output fields and LLM configuration.

    Args:
        signature_cls (Type[dspy.Signature]): 
            A subclass of dspy.Signature that defines the component's interface.

    Returns:
        Type[BaseComponent]: A dynamically created subclass of BaseComponent.
    """
    description = signature_cls.__doc__.strip() if signature_cls.__doc__ else "No description provided."
    input_fields = list(signature_cls.input_fields.keys())
    output_fields = list(signature_cls.output_fields.keys())

    class DSPyModule(BaseComponent):
        def __init__(self, signature_cls: Type[dspy.Signature]):
            """
            Initialize the DSPyModule with input/output fields and LLM config.
            """
            try:
                config = dict(dspy.settings.config['lm'].kwargs)
                if hasattr(dspy.settings.config['lm'], 'model'):
                    config['model'] = dspy.settings.config['lm'].model
            except Exception:
                raise ValueError(
                    "No LLM config found in dspy.settings.config. "
                    "If you're using an LLM, please ensure `dspy.settings.configure(...)` is called."
                )

            self.signature_cls = signature_cls
            super().__init__(
                description=description,
                input_fields=input_fields,
                output_fields=output_fields,
                variable=signature_cls.instructions,
                config=config,
            )

        def forward(self, **inputs) -> dict:
            """
            Run a forward pass using the DSPy signature.

            Args:
                **inputs: Input arguments including any LLM config parameters.

            Returns:
                dict: Model outputs mapped by output field names.
            """
            config = copy.deepcopy(vars(self.config))
            config.pop('randomize_variable', None)

            signature = dspy.Predict(self.signature_cls.with_instructions(self.variable))
            with dspy.settings.context(lm=dspy.LM(**config, cache=False)):
                outputs = signature(**inputs, dspy_cache=False)
                outputs_dict = {k: v for k, v in outputs.items()}

                for key, field in signature.signature.output_fields.items():
                    prefix = field.json_schema_extra.get('prefix', '')
                    if isinstance(outputs_dict.get(key), str) and prefix in outputs_dict[key]:
                        outputs_dict[key] = outputs_dict[key].split(prefix)[-1]

            return outputs_dict

    DSPyModule.__name__ = f"{signature_cls.__name__}Module"
    DSPyModule.__qualname__ = DSPyModule.__name__

    return DSPyModule(signature_cls)
