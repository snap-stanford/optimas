from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OptimasArguments:
    """
    Configuration for the optimization process.
    """
    # ----------- Common -----------
    device: str = field(
        default="cuda:0",
        metadata={"help": "Device for training and evaluation (e.g., 'cuda:0', 'cpu')."}
    )
    skip_data_gen: bool = field(
        default=False,
        metadata={"help": "Skip new data generation."}
    )
    num_iterations: int = field(
        default=10,
        metadata={"help": "Number of optimization iterations."}
    )
    components_to_apply: List[str] = field(
        default_factory=lambda: ["all"],
        metadata={"help": "Modules to apply optimization."}
    )
    per_component_train_size: int = field(
        default=50, metadata={"help": "Train size for component optimization."}
    )
    cooldown_period: int = field(
        default=1,
        metadata={"help": "Number of iterations before a component can be optimized again."}
    )
    # ----------- Reward model training -----------
    per_iteration_rm_train_size: int = field(
        default=-1,
        metadata={"help": "Train size for reward model per iteration."}
    )

    # ------------ Fresh reward dataset generation -----------
    per_iteration_new_data_size: int = field(
        default=20,
        metadata={"help": "Fresh data size for each iteration."}
    )
    max_workers: int = field(
        default=8,
        metadata={"help": "Maximum number of workers for data generation."}
    )
    num_rollouts: int = field(
        default=3,
        metadata={"help": "Number of forward passes for reward estimation."}
    )
    num_samples: int = field(
        default=3,
        metadata={"help": "Number of times to repeat each sample."}
    )
    num_repeats: int = field(
        default=1,
        metadata={"help": "Number of times to repeat the entire dataset."}
    )

    # ------------ Replay Buffer -----------
    replay_buffer_size: int = field(
        default=200,
        metadata={"help": "Size of the replay buffer."}
    )

    use_replay_buffer: bool = field(
        default=False,
        metadata={"help": "Use replay buffer for optimization."}
    )

    # -------------- Prompt Optimization --------------
    prompt_optimizer: str = field(
        default="opro",
        metadata={
            "help": "Prompt optimization method.",
            "choices": ["opro", "mipro", "copro"],
            },
    )
    num_threads: int = field(
        default=1, 
        metadata={"help": "Number of threads for parallel ops."}
    )
    num_prompt_candidates: int = field(
        default=10, 
        metadata={"help": "Number of candidate prompts."}
    ) 
    
    # ----------- OPRO -----------
    opro_llm_model: str = field(
        default="gpt-4o-mini",
        metadata={"help": "LLM model for OPRO optimization"}
    )
    opro_temperature: float = field(
        default=0.7,
        metadata={"help": "Temperature for OPRO LLM"}
    )
    opro_max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum new tokens for OPRO LLM"}
    )
    opro_meta_prompt_preamble_template: str = field(
        default="This component is meant to handle the task:\n{component_description}\nWe want to improve its prompt based on prior attempts.\n",
        metadata={"help": "Meta prompt preamble template for OPRO (use {component_description} placeholder)"}
    )

    # ------ COPRO -----
    copro_depth: int = field(default=2, metadata={"help": "Number of optimization iterations per prompt."})

    # ----- MIPRO ------
    requires_permission_to_run: bool = field(default=False, metadata={"help": "Requires permission to run dspy prompt optimization."})
    verbose: bool = field(default=True, metadata={"help": "Verbose mode."})

    auto: str = field(
        default=None, metadata={"help": "Must be one of {'light', 'medium', 'heavy', None}"}
    )

    # ------- Hyperparameter $ Model Router Optimization ---------
    per_component_search_size: int = field(
        default=20, metadata={"help": "Hyper parameter search size."}
    )
    global_hyper_param_search: bool = field(
        default=False,
        metadata={"help": "Enable hyperparameter search."}
    )

    # -------------- Parameter Optimization --------------
    ppo_epochs: int = field(
        default=4,
        metadata={"help": "Number of epochs for PPO."}
    )
    ppo_mini_batch_size: int = field(
        default=4,
        metadata={"help": "Mini batch size for PPO."}
    )
    ppo_batch_size: int = field(
        default=2,
        metadata={"help": "Batch size for PPO."}
    )
    ppo_learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for PPO."}
    )
    ppo_save_steps: int = field(
        default=10,
        metadata={"help": "Save adapter every N steps."}
    )
    ppo_gradient_accumulation_steps: int = field(
        default=2,
        metadata={"help": "Gradient accumulation steps for PPO."}
    )
    ppo_resume_adapter: Optional[str] = field(
        default=None,
        metadata={"help": "Resume from a specific adapter."}
    )
    weight_optimizer: str = field(
        default="none",
        metadata={
            "help": "Weight optimizer for PPO.",
            "choices": ["ppo"],
        },
    )
    ppo_base_model_name: str = field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        metadata={"help": "Base model name for PPO."}
    )
    emb_sim_reward: bool = field(
        default=False,
        metadata={"help": "Use embedding similarity reward."}
    )

    # ---------------- validation settings -------------
    val_every_ppo_ratio: float = field(
        default=0.25,
        metadata={"help": "Validation ratio for PPO."}
    )
    val_every_prompt_iter: int = field(
        default=5,
        metadata={"help": "Validation interval for prompt optimization."}
    )