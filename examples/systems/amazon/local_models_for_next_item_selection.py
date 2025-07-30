"""
huggingface-cli download Qwen/Qwen2-1.5B --local-dir local_lm/qwen-1_5b/base

CUDA_VISIBLE_DEVICES=6,7 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True python -m vllm.entrypoints.openai.api_server \
    --port 8877 --host localhost --trust-remote-code \
    --enable-lora \
    --max-lora-rank 32 \
    --max-loras 2 \
    --model local_lm/qwen-1_5b/base \
    --tensor-parallel-size 2
"""
import copy
import dspy
from dotenv import load_dotenv
from datasets import Dataset

import os
import os.path as osp
import json
import requests
from typing import Dict, Any, List, Optional
from peft import LoraConfig
from pathlib import Path

from optimas.arch.system import CompoundAISystem
from optimas.arch.base import BaseComponent
from optimas.adapt.dspy import create_component_from_dspy
from optimas.reward_model import RewardModel
from optimas.utils.lora import load_lora_adapter, initialize_adapter


# change input path to env variable
BASE_MODEL_PATH = os.getenv(
    "BASE_MODEL_PATH", "local_lm/qwen-1_5b/base",
)


# Helper functions
def accuracy(answer: str, gd_answer: str) -> float:
    """Exact-match accuracy metric."""
    return 1.0 if str(answer) == str(gd_answer) else 0.0


def post_http_request(
    prompt: str,
    api_url: str,
    headers: Dict[str, str],
    base_model: str,
    adapter_id: str = None,
    *,
    n: int = 1,
    temperature: float = 0.6,
    stream: bool = False,
    max_new_tokens: int = 512,
) -> requests.Response:
    """
    Send a completion request to *local* vLLM in OpenAI-compatible mode.
    """
    payload: Dict[str, Any] = {
        "model": base_model,
        "prompt": prompt,
        "n": n,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "stream": stream,
        
    }
    if adapter_id is not None:
        payload["model"] = adapter_id

    return requests.post(
        api_url,
        headers=headers,
        json=payload,
        stream=stream,
        timeout=180
    )


def get_response(response: requests.Response) -> str:
    """
    Extract the first completion string from an OpenAI response.
    """
    response.raise_for_status()
    data = response.json()
    # OpenAI schema → choices[0].text
    return data["choices"][0]["text"].strip()

# --------------------------------------------------------------------------- #
#  Modules                                                                    #
# --------------------------------------------------------------------------- #
class SessionAnalyzerModule(BaseComponent):
    """Summarise a user's session into a compact context string (local LLM)."""

    def __init__(
        self,
        base_model_path,
        adapter_path,
        lora_cfg: LoraConfig,
        host: str = "localhost",
        port: int = 8877,
    ):
        self.host = host
        self.port = port
        self.api_url = f"http://{host}:{port}/v1/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('VLLM_API_KEY', 'dummy')}",
            "User-Agent": "SessionAnalyzerClient",
        }

        self.adapter_id = self.__class__.__name__
        initialize_adapter(base_model_path, adapter_path, lora_cfg)
        super().__init__(
            description="Summarise session into context using local VLLM",
            input_fields=["sequence"],
            output_fields=["context"],
            variable=Path(adapter_path),
            config={
                "base_model_path": base_model_path,
                "lora_cfg": lora_cfg,
                "temperature": 0.6,
                "max_new_tokens": 512
            }
        )

    def on_variable_update_end(self):
        load_lora_adapter(self.adapter_id, self.variable, self.host, self.port)

    def forward(self, **inputs):
        sequence = inputs["sequence"]
        prompt = (
            "You are an e-commerce behaviour analyst.\n\n"
            "Session sequence:\n"
            f"{sequence}\n\n"
            "Provide a 2-3 sentence summary of the user's browsing intent."
        )
        response = post_http_request(
            prompt,
            self.api_url,
            headers=self.headers,
            temperature=self.config.temperature,
            base_model=self.config.base_model_path,
            max_new_tokens=self.config.max_new_tokens,
            adapter_id=self.adapter_id,
        )
        summary = get_response(response)
        return {"context": summary}


class CandidateProfilerModule(BaseComponent):
    """Give line-by-line feedback on each candidate item (local LLM)."""

    def __init__(
        self,
        base_model_path,
        adapter_path,
        lora_cfg: LoraConfig,
        host: str = "localhost",
        port: int = 8877,
    ):
        self.host = host
        self.port = port
        self.api_url = f"http://{host}:{port}/v1/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('VLLM_API_KEY', 'dummy')}",
            "User-Agent": "CandidateProfilerClient",
        }
        initialize_adapter(base_model_path, adapter_path, lora_cfg)
        self.adapter_id = self.__class__.__name__
        super().__init__(
            description="Generate feedback for each candidate using local VLLM",
            input_fields=["context", "choices"],
            output_fields=["feedback"],
            variable=Path(adapter_path),
            config={
                "base_model_path": base_model_path,
                "lora_cfg": lora_cfg,
                "temperature": 0.6,
                "max_new_tokens": 512
            }
        )

    def on_variable_update_end(self):
        load_lora_adapter(self.adapter_id, self.variable, self.host, self.port)

    def forward(self, **inputs):
        context = inputs["context"]
        choices = inputs["choices"]
        prompt = (
            "You are an e-commerce candidate profiler.\n\n"
            "Session summary:\n"
            f"{context}\n\n"
            "Candidate items:\n"
            f"{json.dumps(choices, indent=2)}\n\n"
            "For each item, on its own line, write a brief (1-2 sentence) "
            "comment on why the user might or might not choose it next."
        )

        # Use the current adapter_id (which might be an adapter) or the base model

        response = post_http_request(
            prompt,
            self.api_url,
            headers=self.headers,
            temperature=self.config.temperature,
            base_model=self.config.base_model_path,
            max_new_tokens=self.config.max_new_tokens,
            adapter_id=self.adapter_id,
        )
        feedback = get_response(response)
        return {"feedback": feedback}


class NextItemDecider(dspy.Signature):
    """Select the next item by considering both the summary and the provided feedback carefully."""
    context: str = dspy.InputField(prefix="Context: ", desc="Summary of behaviour")
    feedback: str = dspy.InputField(prefix="Feedback: ", desc="Comments per option")
    answer: str = dspy.OutputField(prefix="Answer: ", desc="Index of item chosen")


# --------------------------------------------------------------------------- #
#  Pipeline factory                                                           #
# --------------------------------------------------------------------------- #
def system_engine(*args, **kwargs):
    from dotenv import load_dotenv
    import os.path as osp
    import os

    # Load .env secrets
    dotenv_path = osp.expanduser(".env")
    load_dotenv(dotenv_path)

    # Configure LLM
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=1024,
        temperature=0.3,
    )
    dspy.settings.configure(lm=lm)

    # Host and port setup for VLLM adapters
    host = os.getenv("VLLM_HOST", "localhost")
    port = int(os.getenv("VLLM_PORT", "8877"))

    lora_cfg = LoraConfig(
        r=32, 
        lora_alpha=32, 
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Initialize system using constructor-based configuration
    system = CompoundAISystem(
        components={
            "session_analyzer": SessionAnalyzerModule(
                lora_cfg=lora_cfg, host=host, port=port, base_model_path=BASE_MODEL_PATH,
                adapter_path=osp.join(BASE_MODEL_PATH, "session_analyzer_adapter")
            ),
            "candidate_profiler": CandidateProfilerModule(
                lora_cfg=lora_cfg, host=host, port=port, base_model_path=BASE_MODEL_PATH,
                adapter_path=osp.join(BASE_MODEL_PATH, "candidate_profiler_adapter")
            ),
            "next_item_decider": create_component_from_dspy(NextItemDecider),
        },
        final_output_fields=["answer"],
        ground_fields=["gd_answer"],
        eval_func=accuracy,
        *args,
        **kwargs,
    )

    return system


if __name__ == "__main__":

    from examples.datasets.amazon import dataset_engine
    trainset, valset, testset = dataset_engine()

    system = system_engine()

    pred = system(
        sequence=trainset[0].sequence, 
        choices=trainset[0].choices
    )
    print("Sample prediction:", pred.answer)
    print("Sample answer:", trainset[0].gd_answer)

    print("Evaluating on testset...")
    scores = system.evaluate_multiple(testset)
    print("Average score:", sum(scores) / len(scores))
    
    comp = system.components['session_analyzer']
    result1 = comp(sequence=trainset[0].sequence)
    comp.update(Path('path/to/your/lora_adapter'))
    result2 = comp(sequence=trainset[0].sequence)
    print("Result 1:", result1)
    print("Result 2:", result2)