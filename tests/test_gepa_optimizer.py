import pytest
import os

pytest.importorskip("dspy")
pytest.importorskip("gepa")

import dspy
from dspy.teleprompt.gepa import GEPA

def test_gepa_optimizer_runs():
    class DummySignature(dspy.Signature):
        question: str = dspy.InputField(desc="The question")
        answer: str = dspy.OutputField(desc="The answer")
        __doc__ = "Answer the question."

    class DummyModule(dspy.Module):
        signature = DummySignature
        def forward(self, question):
            return {"answer": "42"}

    module = DummyModule()
    trainset = [dspy.Example(question="What is 6*7?", answer="42")]
    valset = [dspy.Example(question="What is 2*21?", answer="42")]
    def metric(gold, pred, *args, **kwargs):
        return 1.0 if pred.answer == gold.answer else 0.0
    reflection_lm = lambda prompt: "Try 42."
    gepa = GEPA(
        metric=metric,
        max_metric_calls=1,
        reflection_lm=reflection_lm,
        candidate_selection_strategy="pareto",
        skip_perfect_score=True,
        use_merge=False,
        track_stats=False,
        use_wandb=False,
        log_dir=None,
    )
    result = gepa.compile(module, trainset=trainset, valset=valset)
    assert hasattr(result, "signature")
