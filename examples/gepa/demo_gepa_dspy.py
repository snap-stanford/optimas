#!/usr/bin/env python3
"""
GEPA Demo with DSPy Components

Demonstrates GEPA prompt optimization using DSPy signatures and components.
Shows side-by-side comparison with BaseComponent approach.

Usage: python examples/gepa/demo_gepa_dspy.py
"""

import sys
import os
import requests
from typing import Dict, Any

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    import dspy
except ImportError:
    print("❌ DSPy not installed. Install with: pip install dspy-ai")
    sys.exit(1)

from optimas.arch.base import BaseComponent
from optimas.adapt.dspy import create_component_from_dspy
from optimas.optim.universal_gepa import UniversalGEPAOptimizer
from optimas.wrappers.example import Example
from optimas.wrappers.prediction import Prediction


# DSPy Signature for Question Answering  
class QuestionAnswerSignature(dspy.Signature):
    """Answer questions accurately with brief, factual responses."""
    
    question: str = dspy.InputField(desc="The question to answer")
    answer: str = dspy.OutputField(desc="A clear, concise answer")


# Custom Ollama LM for DSPy
class OllamaLM(dspy.LM):
    """Custom DSPy language model using local Ollama"""
    
    def __init__(self, model="llama3.1:8b", **kwargs):
        super().__init__(model=model, **kwargs)
        self.model = model
        self.history = []
    
    def __call__(self, prompt, **kwargs):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": str(prompt),
                    "stream": False,
                    "options": {"temperature": kwargs.get("temperature", 0.1)}
                },
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                
                # DSPy expects a list of choices
                choice = dspy.Prediction(answer=answer)
                self.history.append({"prompt": prompt, "response": answer})
                return [choice]
            else:
                return [dspy.Prediction(answer=f"Error: {response.status_code}")]
                
        except Exception as e:
            return [dspy.Prediction(answer=f"Error: {e}")]


# Regular BaseComponent for comparison
class RegularQAComponent(BaseComponent):
    """Regular BaseComponent Q&A for comparison"""
    
    def __init__(self):
        super().__init__(
            description="Answer questions using regular BaseComponent",
            input_fields=["question"],
            output_fields=["answer"],
            variable="Answer the question clearly and concisely.",
            config={"model": "llama3.1:8b"}
        )
    
    def forward(self, **inputs) -> Dict[str, Any]:
        question = inputs.get("question", "")
        prompt = f"{self.variable}\n\nQuestion: {question}\nAnswer:"
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
            else:
                answer = f"Error: {response.status_code}"
        except Exception as e:
            answer = f"Error: {e}"
        
        return {"answer": answer}


def create_reflection_lm():
    """Create reflection model using qwen3:8b"""
    def reflection_lm(prompt: str) -> str:
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen3:8b",
                    "prompt": f"Analyze this prompt optimization task and suggest improvements:\n{prompt}\n\nSuggestion:",
                    "stream": False,
                    "options": {"temperature": 0.7}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            return "Make the prompt more specific and clear."
        except Exception:
            return "Be more specific in your instructions."
    
    return reflection_lm


def qa_metric(gold: Example, pred: Prediction, trace=None) -> float:
    """Evaluation metric for Q&A"""
    try:
        expected = gold.labels().get("answer", "").lower()
        actual = pred.answer.lower()
        
        # Exact match gets full score
        if expected in actual or actual in expected:
            return 1.0
        
        # Check for key words
        expected_words = set(expected.split())
        actual_words = set(actual.split())
        overlap = len(expected_words & actual_words)
        
        if overlap > 0:
            return overlap / max(len(expected_words), 1) * 0.7
        
        return 0.0
        
    except Exception:
        return 0.0


def test_component(component, examples, name):
    """Test a component and return average score"""
    print(f"\n📋 Testing {name}:")
    scores = []
    
    for ex in examples:
        if hasattr(component, 'forward'):
            # BaseComponent
            result = component(question=ex.question)
            pred = Prediction(answer=result["answer"])
        else:
            # DSPy component
            try:
                result = component(question=ex.question)
                pred = Prediction(answer=result.answer)
            except Exception as e:
                print(f"  Error with DSPy component: {e}")
                pred = Prediction(answer="Error")
        
        score = qa_metric(ex, pred)
        scores.append(score)
        
        print(f"  Q: {ex.question}")
        print(f"  A: {pred.answer[:50]}...")
        print(f"  Score: {score:.2f}")
    
    avg_score = sum(scores) / len(scores)
    print(f"Average score: {avg_score:.2f}")
    return avg_score


def main():
    """Run the DSPy vs BaseComponent GEPA demo"""
    print("🧠 GEPA Demo: DSPy vs BaseComponent")
    print("=" * 50)
    
    # Check Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in response.json().get("models", [])]
        required = ["llama3.1:8b", "qwen3:8b"]
        missing = [m for m in required if m not in models]
        if missing:
            print(f"❌ Missing models: {missing}")
            return
        print("✅ Ollama models ready")
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return
    
    # Create dataset
    examples = [
        Example(question="What is the capital of Japan?", answer="Tokyo").with_inputs("question"),
        Example(question="How many sides does a triangle have?", answer="3").with_inputs("question"),
        Example(question="What gas do plants absorb from the air?", answer="carbon dioxide").with_inputs("question"),
    ]
    
    print(f"Dataset: {len(examples)} questions")
    
    # Setup DSPy with local Ollama
    print("\n🔧 Setting up DSPy with Ollama...")
    dspy_available = True
    try:
        # Use DSPy's built-in Ollama support
        ollama_lm = dspy.LM("ollama/llama3.1:8b", api_base="http://localhost:11434")
        dspy.configure(lm=ollama_lm)
    except Exception as e:
        print(f"⚠️  DSPy setup failed: {e}")
        print("Skipping DSPy optimization, will only test BaseComponent")
        dspy_available = False
    
    # Create components
    print("Creating components...")
    
    # 1. DSPy Component (if setup succeeded)
    dspy_component = None
    if dspy_available:
        try:
            dspy_component = create_component_from_dspy(
                signature_cls=QuestionAnswerSignature
            )
            print(f"DSPy initial instruction: '{dspy_component.variable}'")
        except Exception as e:
            print(f"⚠️  DSPy component creation failed: {e}")
            dspy_component = None
    
    # 2. Regular BaseComponent  
    regular_component = RegularQAComponent()
    print(f"BaseComponent initial prompt: '{regular_component.variable}'")
    
    # Test both components before optimization
    if dspy_component is not None:
        dspy_before = test_component(dspy_component, examples, "DSPy Component (before)")
    else:
        dspy_before = 0.0
        print("⚠️  Skipping DSPy component test")
        
    regular_before = test_component(regular_component, examples, "BaseComponent (before)")
    
    # Create GEPA optimizer
    print("\n⚙️ Setting up GEPA optimization...")
    reflection_lm = create_reflection_lm()
    
    optimizer = UniversalGEPAOptimizer(
        reflection_lm=reflection_lm,
        auto_budget="light",
        reflection_minibatch_size=2,
        max_workers=1,
        seed=42
    )
    
    # Optimize DSPy component (if available)
    if dspy_component is not None:
        print("\n🔄 Optimizing DSPy component...")
        try:
            dspy_result = optimizer.optimize_component(
                component=dspy_component,
                trainset=examples[:2],
                valset=examples[2:],
                metric_fn=qa_metric
            )
            
            print(f"DSPy optimization completed!")
            print(f"Framework detected: {dspy_result.framework_type}")
            print(f"Final score: {dspy_result.final_score:.3f}")
            print(f"Total evaluations: {dspy_result.total_evaluations}")
            
            if dspy_result.best_candidate:
                for name, text in dspy_result.best_candidate.items():
                    print(f"Optimized {name}: '{text}'")
        
        except Exception as e:
            print(f"DSPy optimization failed: {e}")
            print("Continuing with BaseComponent optimization...")
    else:
        print("\n⚠️  Skipping DSPy optimization (component not available)")
    
    # Optimize BaseComponent
    print("\n🔄 Optimizing BaseComponent...")
    try:
        regular_result = optimizer.optimize_component(
            component=regular_component,
            trainset=examples[:2],
            valset=examples[2:],
            metric_fn=qa_metric
        )
        
        print(f"BaseComponent optimization completed!")
        print(f"Framework detected: {regular_result.framework_type}")
        print(f"Final score: {regular_result.final_score:.3f}")
        print(f"Total evaluations: {regular_result.total_evaluations}")
        
        if regular_result.best_candidate:
            for name, text in regular_result.best_candidate.items():
                print(f"Optimized {name}: '{text}'")
    
    except Exception as e:
        print(f"BaseComponent optimization failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test both components after optimization
    print("\n📊 Final Results:")
    if dspy_component is not None:
        dspy_after = test_component(dspy_component, examples, "DSPy Component (after)")
    else:
        dspy_after = 0.0
        print("⚠️  Skipping DSPy component final test")
        
    regular_after = test_component(regular_component, examples, "BaseComponent (after)")
    
    # Summary comparison
    print("\n🏆 Summary Comparison:")
    if dspy_component is not None:
        print(f"DSPy Component:")
        print(f"  Before: {dspy_before:.2f}")
        print(f"  After:  {dspy_after:.2f}")
        print(f"  Change: {dspy_after - dspy_before:+.2f}")
    else:
        print("DSPy Component: Not available")
    
    print(f"\nBaseComponent:")
    print(f"  Before: {regular_before:.2f}")
    print(f"  After:  {regular_after:.2f}")
    print(f"  Change: {regular_after - regular_before:+.2f}")
    
    # Determine winner (if DSPy was available)
    if dspy_component is not None:
        dspy_improvement = dspy_after - dspy_before
        regular_improvement = regular_after - regular_before
        
        print(f"\n🎯 Best Approach:")
        if dspy_improvement > regular_improvement:
            print("🥇 DSPy + GEPA performed better!")
        elif regular_improvement > dspy_improvement:
            print("🥇 BaseComponent + GEPA performed better!")
        else:
            print("🤝 Both approaches performed equally well!")
    else:
        print(f"\n🎯 Result:")
        print("✅ BaseComponent + GEPA optimization demonstrated successfully!")
    
    print("\n✨ Demo completed!")
    print("Both DSPy and BaseComponent work seamlessly with GEPA optimization.")


if __name__ == "__main__":
    main()