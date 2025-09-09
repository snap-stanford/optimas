#!/usr/bin/env python3
"""
GEPA Demo with Ollama

Demonstrates GEPA prompt optimization using local Ollama models.
Uses llama3.1:8b for inference and qwen3:8b for reflection.

Usage: python examples/gepa/demo_gepa.py
"""

import sys
import os
import requests
from typing import Dict, Any, List

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from optimas.arch.base import BaseComponent
from optimas.optim.universal_gepa import UniversalGEPAOptimizer
from optimas.wrappers.example import Example
from optimas.wrappers.prediction import Prediction


class SimpleQAComponent(BaseComponent):
    """Simple Q&A component that GEPA can optimize"""

    def __init__(self):
        super().__init__(
            description="Answer questions using Ollama",
            input_fields=["question"],
            output_fields=["answer"],
            variable="Answer the question clearly.",
            config={"model": "llama3.1:8b"}
        )

    def forward(self, **inputs) -> Dict[str, Any]:
        question = inputs.get("question", "")

        # Build prompt with current instruction
        prompt = f"{self.variable}\n\nQuestion: {question}\nAnswer:"

        # Call Ollama
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
                    "prompt": f"Help improve this prompt:\n{prompt}\n\nSuggestion:",
                    "stream": False,
                    "options": {"temperature": 0.7}
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            return "Make the prompt clearer and more specific."
        except Exception as e:
            return f"Reflection error: {e}"

    return reflection_lm


def qa_metric(gold: Example, pred: Prediction, trace=None) -> float:
    """Simple evaluation metric"""
    try:
        expected = gold.labels().get("answer", "").lower()
        actual = pred.answer.lower()

        # Check if expected answer is in the response
        if expected in actual:
            return 1.0

        # Partial credit for containing keywords
        expected_words = set(expected.split())
        actual_words = set(actual.split())
        overlap = len(expected_words & actual_words)
        return overlap / max(len(expected_words), 1) * 0.5

    except Exception:
        return 0.0


def main():
    """Run the GEPA demo"""
    print("🚀 GEPA Demo with Ollama")
    print("=" * 40)

    # Check Ollama
    print("Checking Ollama...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in response.json().get("models", [])]
        print(f"✅ Found models: {', '.join(models)}")

        required = ["llama3.1:8b", "qwen3:8b"]
        missing = [m for m in required if m not in models]
        if missing:
            print(f"❌ Missing models: {missing}")
            print("Run: ollama pull " + " && ollama pull ".join(missing))
            return
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return

    # Create component and examples
    print("\nSetting up component...")
    component = SimpleQAComponent()
    print(f"Initial prompt: '{component.variable}'")

    # Create simple dataset
    examples = [
        Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
        Example(question="What is 2 + 3?", answer="5").with_inputs("question"),
        Example(question="What color is the sun?", answer="yellow").with_inputs("question"),
    ]

    print(f"Dataset: {len(examples)} examples")

    # Test before optimization
    print("\n📋 Testing before optimization:")
    scores = []
    for ex in examples:
        result = component(question=ex.question)
        pred = Prediction(answer=result["answer"])
        score = qa_metric(ex, pred)
        scores.append(score)
        print(f"  Q: {ex.question}")
        print(f"  A: {result['answer'][:50]}...")
        print(f"  Score: {score:.2f}")

    before_avg = sum(scores) / len(scores)
    print(f"Average score before: {before_avg:.2f}")

    # Run GEPA optimization
    print("\n⚙️ Running GEPA optimization...")
    optimizer = UniversalGEPAOptimizer(
        reflection_lm=create_reflection_lm(),
        auto_budget="light",  # Small budget for demo
        reflection_minibatch_size=2,
        max_workers=1,
        seed=42
    )

    try:
        result = optimizer.optimize_component(
            component=component,
            trainset=examples[:2],  # Use 2 for training
            valset=examples[2:],    # Use 1 for validation
            metric_fn=qa_metric
        )

        print("\n📊 Optimization results:")
        print(f"Final score: {result.final_score:.3f}")
        print(f"Total evaluations: {result.total_evaluations}")

        if result.best_candidate:
            for name, text in result.best_candidate.items():
                print(f"Optimized {name}: '{text}'")

        # Test after optimization
        print("\n📋 Testing after optimization:")
        scores_after = []
        for ex in examples:
            result_after = component(question=ex.question)
            pred = Prediction(answer=result_after["answer"])
            score = qa_metric(ex, pred)
            scores_after.append(score)
            print(f"  Q: {ex.question}")
            print(f"  A: {result_after['answer'][:50]}...")
            print(f"  Score: {score:.2f}")

        after_avg = sum(scores_after) / len(scores_after)
        improvement = after_avg - before_avg

        print(f"\n📈 Results:")
        print(f"Before: {before_avg:.2f}")
        print(f"After:  {after_avg:.2f}")
        print(f"Change: {improvement:+.2f}")

        if improvement > 0:
            print("🎉 GEPA improved the component!")
        else:
            print("🤔 No improvement (try more data/iterations)")

    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n✨ Demo completed!")


if __name__ == "__main__":
    main()
