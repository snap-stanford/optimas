import os
import os.path as osp
import json
import dspy
import numpy as np

from optimas.arch.base import BaseComponent
from optimas.arch.system import CompoundAISystem
from optimas.adapt.dspy import create_component_from_dspy
from optimas.utils.api import get_llm_output
from examples.metrics.mrr import mrr


class RelationScorerSignature(dspy.Signature):
    """Given a question and a list of 5 entities with their relational information, assign each entity a relevance score (between 0 and 1) based on how well its relations match the information in the question."""
    question: str = dspy.InputField(
        prefix="Query: "
    )
    relation_info: str = dspy.InputField(
        prefix="Relational Information: "
    )
    relation_scores: str = dspy.OutputField(
        prefix="Json Scores \{1: score1, ..., 5: score5\}: "
    )


class TextScorerSignature(dspy.Signature):
    """Given a question and a list of 5 entities with their property information, assign each entity a relevance score between 0 and 1 based on how well its properties match the requirements described in the question."""
    question: str = dspy.InputField(
        prefix="Query: "
    )
    text_info: str = dspy.InputField(
        prefix="Property Information: "
    )
    text_scores: str = dspy.OutputField(
        prefix="Json Scores \{1: score1, ..., 5: score5\}: "
    )


class FinalScorer(BaseComponent):
    def __init__(self):
        super().__init__(
            description="Given a question, assess the importance of textual properties, relational cues, and general semantics in retrieving an entity. Combine the three score lists into a final score list using weighted aggregation.",
            input_fields=["question", "emb_scores", "relation_scores", "text_scores"],
            output_fields=["final_scores"],
            variable={
                'relation_weight': 0.1,
                'text_weight': 0.1
            },
            variable_search_space={
                'relation_weight': [0.1, 1.0],
                'text_weight': [0.1, 1.0]
            }
        )

    def forward(self, **inputs):
        question = inputs.get("question")
        emb_scores = inputs.get("emb_scores")
        relation_scores = inputs.get("relation_scores")
        text_scores = inputs.get("text_scores")
        
        try:
            relation_scores = json.loads(relation_scores).values()
            relation_scores = [float(x) for x in relation_scores]
            assert len(relation_scores) == 5
        except:
            relation_scores = [0 for _ in range(5)]
        try:
            text_scores = json.loads(text_scores).values()
            text_scores = [float(x) for x in text_scores]
            assert len(text_scores) == 5
        except:
            text_scores = [0 for _ in range(5)]

        relation_weight = self.variable['relation_weight']
        text_weight = self.variable['text_weight']

        final_scores = [relation_weight * r + text_weight * t + e for r, t, e in zip(relation_scores, text_scores, emb_scores)]
        return {"final_scores": [round(x, 2) for x in final_scores]}


def system_engine(*args, **kwargs):
    # Configure the LLM
    lm = dspy.LM(
        model="anthropic/claude-3-haiku-20240307",
        max_tokens=256,
        temperature=0.6,
    )
    dspy.settings.configure(lm=lm)

    dataset = kwargs.get("dataset", None)  # Optional, may be used externally

    # Initialize the system with declarative component definition
    system = CompoundAISystem(
        components={
            "relation_scorer": create_component_from_dspy(RelationScorerSignature),
            "text_scorer": create_component_from_dspy(TextScorerSignature),
            "final_scorer": FinalScorer(),
        },
        final_output_fields=["final_scores"],
        ground_fields=["candidate_ids", "answer_ids"],
        eval_func=mrr,
        *args,
        **kwargs,
    )

    return system


if __name__ == "__main__":

    from dotenv import load_dotenv
    from examples.datasets.stark_prime import dataset_engine
    
    dotenv_path = osp.expanduser(".env")
    load_dotenv(dotenv_path)

    trainset, valset, testset = dataset_engine()
    system = system_engine()
    
    pred = system(**testset[0])
    metric = system.evaluate(example=testset[0], prediction=pred)
    print('final_scores:', pred.final_scores, 'candidate_ids:', testset[0].candidate_ids)
    print('answer_ids:', testset[0].answer_ids, 'metric', metric)