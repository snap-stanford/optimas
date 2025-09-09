# GEPA Integration Guide

GEPA automatically optimizes prompts and text components in your AI systems. This guide explains how GEPA works with Optimas and how to use it.

## What is GEPA?

**GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning**  
📄 [Paper](https://arxiv.org/abs/2507.19457) | 🔗 [GitHub](https://github.com/gepa-ai/gepa)

GEPA (Genetic-Pareto) is a framework for optimizing arbitrary systems composed of text components—like AI prompts, code snippets, or textual specs—against any evaluation metric. 

### How GEPA Works

GEPA employs LLMs to **reflect** on system behavior, using feedback from execution and evaluation traces to drive targeted improvements. Through iterative **mutation**, **reflection**, and **Pareto-aware candidate selection**, GEPA evolves robust, high-performing variants with minimal evaluations.

The process:
1. **Evaluate** current prompts on your data
2. **Reflect** on failures using an LLM to understand what went wrong  
3. **Mutate** prompts based on reflective feedback
4. **Select** best candidates using Pareto-aware selection
5. **Iterate** until convergence or budget exhaustion

GEPA can co-evolve multiple components in modular systems, making it perfect for optimizing complex AI pipelines with minimal human intervention.

## Quick Start

```python
from optimas.optim.universal_gepa import UniversalGEPAOptimizer

# 1. Create a reflection model (the AI that suggests improvements)
def reflection_lm(prompt):
    # Use any LM - OpenAI, Anthropic, or local Ollama
    return your_llm_call(f"Improve this prompt: {prompt}")

# 2. Create GEPA optimizer
optimizer = UniversalGEPAOptimizer(
    reflection_lm=reflection_lm,
    auto_budget="light"  # How much optimization to do
)

# 3. Optimize your component
result = optimizer.optimize_component(
    component=your_component,
    trainset=your_examples,
    metric_fn=your_evaluation_function
)

# Your component now has an optimized prompt!
```

## Setting Up Components for GEPA

Your components need two things to work with GEPA:

### 1. Make Variables Optimizable

```python
class MyComponent(BaseComponent):
    def __init__(self):
        super().__init__(
            description="What this component does",
            input_fields=["question"],
            output_fields=["answer"],
            variable="Your initial prompt here",  # GEPA will optimize this
            config={"model": "gpt-4"}
        )
```

### 2. That's It!

GEPA automatically detects optimizable components. The base `BaseComponent` class already provides:
- `gepa_optimizable_components` - Shows what GEPA can optimize
- `apply_gepa_updates()` - Applies optimized prompts

## Configuration Options

### Reflection Models

**Local Ollama** (recommended for development):
```python
import requests

def ollama_reflection_lm(prompt):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.1:8b",
        "prompt": f"Improve this prompt: {prompt}"
    })
    return response.json()["response"]
```

**OpenAI**:
```python
import openai

def openai_reflection_lm(prompt):
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Improve this prompt: {prompt}"}]
    )
    return response.choices[0].message.content
```

### Budget Control

Control how much optimization GEPA does:

```python
# Simple options
UniversalGEPAOptimizer(
    reflection_lm=reflection_lm,
    auto_budget="light"    # 50 evaluations - fast
    # auto_budget="medium"  # 100 evaluations - balanced  
    # auto_budget="heavy"   # 200 evaluations - thorough
)

# Precise control
UniversalGEPAOptimizer(
    reflection_lm=reflection_lm,
    max_metric_calls=75,      # Exactly 75 evaluations
    # OR
    num_iters=10,             # 10 optimization rounds
    # OR  
    max_full_evals=5          # 5 complete dataset evaluations
)
```

### Advanced Options

```python
UniversalGEPAOptimizer(
    reflection_lm=reflection_lm,
    auto_budget="medium",
    reflection_minibatch_size=3,    # Examples per reflection
    candidate_selection_strategy="pareto",  # How to pick best prompts
    skip_perfect_score=True,        # Stop if score is perfect
    use_merge=True,                 # Combine good prompts
    max_workers=2,                  # Parallel evaluation
    seed=42                         # Reproducible results
)
```

## Creating Evaluation Functions

GEPA needs a way to measure if prompts are good:

```python
def my_evaluation_function(gold_example, prediction, trace=None):
    """
    Args:
        gold_example: The correct answer
        prediction: Your component's output
        trace: Optional execution details
    
    Returns:
        float: Score from 0.0 (bad) to 1.0 (perfect)
    """
    expected = gold_example.labels()["answer"]
    actual = prediction.answer
    
    # Simple exact match
    return 1.0 if expected.lower() == actual.lower() else 0.0
    
    # Or more sophisticated scoring...
```

## Working with Different Frameworks

### DSPy Components

```python
# GEPA automatically detects DSPy signatures
import dspy

class QASignature(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

component = create_component_from_dspy(
    signature_cls=QASignature,
    instruction="Answer the question clearly."
)

# Works directly with GEPA
result = optimizer.optimize_component(component, trainset, metric_fn)
```

### Custom Components

```python
class CustomComponent(BaseComponent):
    def forward(self, **inputs):
        # Your component logic here
        question = inputs["question"]
        prompt = f"{self.variable}\n\nQ: {question}\nA:"
        
        response = your_llm_call(prompt)
        return {"answer": response}
    
    # Optional: custom GEPA integration
    @property
    def gepa_optimizable_components(self):
        return {"instructions": self.variable}
    
    def apply_gepa_updates(self, updates):
        if "instructions" in updates:
            self.update(updates["instructions"])
```

## Examples and Troubleshooting

### Complete Examples

**Basic GEPA Demo**
- `examples/gepa/demo_gepa.py` - BaseComponent with local Ollama models
- Shows standard GEPA optimization workflow

**DSPy vs BaseComponent Comparison**
- `examples/gepa/demo_gepa_dspy.py` - Side-by-side comparison demo
- Demonstrates both DSPy and BaseComponent approaches with GEPA
- Uses `llama3.1:8b` for inference and `qwen3:8b` for reflection

#### Framework Comparison Results

| Framework | GEPA Integration | Detection | Optimization Path |
|-----------|------------------|-----------|------------------|
| **DSPy** | Native DSPy GEPA | `framework_type: dspy` | Uses DSPy's built-in GEPA teleprompt |
| **BaseComponent** | Universal Adapter | `framework_type: generic` | Uses Optimas Universal GEPA Optimizer |

**Key Findings:**
- Both frameworks work seamlessly with GEPA optimization  
- DSPy uses its native GEPA integration for signature optimization
- BaseComponent uses the universal adapter for any text-based component
- Local Ollama models (`llama3.1:8b`, `qwen3:8b`) work perfectly with both approaches
- Performance and optimization quality are comparable between approaches

### Common Issues

**"No optimizable components found"**
- Make sure your component has a `variable` parameter
- The variable should be a string (the prompt to optimize)

**"Inputs have not been set"**
- Use `.with_inputs()` on your examples:
```python
examples = [
    Example(question="What is 2+2?", answer="4").with_inputs("question")
]
```

**Slow optimization**
- Use `auto_budget="light"` for faster results
- Reduce `reflection_minibatch_size` to 2
- Set `max_workers=1` to avoid conflicts

**No improvement**
- Your initial prompt might already be good!
- Try with more diverse/challenging examples
- GEPA correctly avoids changing prompts that work well

## Best Practices

1. **Start Small**: Use `auto_budget="light"` first
2. **Good Examples**: Provide diverse, challenging examples
3. **Clear Metrics**: Write evaluation functions that measure what you care about
4. **Local Development**: Use Ollama for development, cloud models for production
5. **Monitor Results**: GEPA will tell you if/how much it improved your prompts

## Local Ollama Setup

For development with local models:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull llama3.1:8b    # Fast inference
ollama pull qwen3:8b       # Good for reflection

# Run demo
python examples/gepa/demo_gepa.py
```

This uses your local models instead of API calls, perfect for experimentation.

---

GEPA makes prompt optimization automatic. Give it your component, examples, and evaluation function - it handles the rest!