import os
import dspy
import os.path as osp
from dotenv import load_dotenv

from optimas.arch.system import CompoundAISystem
from optimas.arch.base import BaseComponent
from optimas.adapt.dspy import create_component_from_dspy
from examples.metrics.f1_score import f1_score


class QuestionRewriter(dspy.Signature):
    """Rephrase the question."""
    question: str = dspy.InputField(
        prefix="Query: "
    )
    rewritten_query: str = dspy.OutputField(
        prefix="Rewritten Query: "
    )

class InfoExtractor(dspy.Signature):
    """Extract keywords from the query to retrieve relevant content."""

    rewritten_query: str = dspy.InputField(
        prefix="Query: "
    )
    search_keywords: str = dspy.OutputField(
        prefix="Search Keywords: "
    )


class WikipediaRetriever(BaseComponent):
    def __init__(self, k, variable_search_space):
        super().__init__(
            description="Retrieve content from Wikipedia.",
            input_fields=["search_keywords"],
            output_fields=["retrieve_content"],
            variable={"k": k},
            variable_search_space=variable_search_space
        )
        colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
        dspy.settings.configure(rm=colbertv2_wiki17_abstracts)

    def forward(self, **inputs):
        search_keywords = inputs.get("search_keywords")
        if not search_keywords:
            raise ValueError("Missing required input: 'search_keywords'")

        topk_passages = dspy.Retrieve(k=self.variable["k"])(search_keywords).passages
        retrieve_content = "\n".join(topk_passages)
        return {"retrieve_content": retrieve_content}


class HintGenerator(dspy.Signature):
    """Generate useful hints to answer the query."""

    rewritten_query: str = dspy.InputField(
        prefix="Query: "
    )
    retrieve_content: str = dspy.InputField(
        prefix="Retrieved Information: "
    )
    hints: str = dspy.OutputField(
        prefix="Hints: "
    )


class AnswerGenerator(dspy.Signature):
    """Given some hints, directly answer the query with a short answer for the query."""
    rewritten_query: str = dspy.InputField(
        prefix="Query: "
    )
    hints: str = dspy.InputField(
        prefix="Hints: "
    )
    answer: str = dspy.OutputField(prefix="Short Answer: ")


def system_engine(*args, **kwargs):
    lm = dspy.LM(model='openai/gpt-4o-mini', max_tokens=1024, temperature=0.6)
    dspy.settings.configure(lm=lm)

    # System automatically logs summary and Mermaid graph on initialization
    system = CompoundAISystem(
        components={
            "question_rewriter": create_component_from_dspy(QuestionRewriter),
            "info_extractor": create_component_from_dspy(InfoExtractor),
            "wikipedia_retriever": WikipediaRetriever(k=1, variable_search_space={"k": [1, 5, 10, 25]}),
            "hint_generator": create_component_from_dspy(HintGenerator),
            "answer_generator": create_component_from_dspy(AnswerGenerator),
        },
        final_output_fields=["answer"],
        ground_fields=["gd_answer"],
        eval_func=f1_score,
        *args, **kwargs
    )
    
    return system


if __name__ == "__main__":

    from examples.datasets.hotpotqa import dataset_engine
    trainset, valset, testset = dataset_engine()
    dotenv_path = osp.expanduser('.env')
    load_dotenv(dotenv_path)

    system = system_engine()

    with system.components['wikipedia_retriever'].context(variable={"k": 50}):
        score = system.evaluate(testset[0])
        print(score)

    with system.components['question_rewriter'].context(model="openai/gpt-4o", max_tokens=1024):
        prediction = system(**testset[0])
        print(prediction)
        print('prediction:', prediction.answer, 'answer:', testset[0].gd_answer)
        score = system.evaluate(testset[0], prediction)
        print(score)
                


