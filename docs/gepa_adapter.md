# Using the GEPA Adapter in Optimas

## What is GEPA?
GEPA (Genetic-Pareto) is an evolutionary optimizer for text-based components (e.g., prompts, instructions, code snippets) in AI systems. It uses LLM-based reflection and Pareto-aware search to evolve robust, high-performing variants with minimal evaluations. See the [GEPA paper](https://arxiv.org/abs/2507.19457) and [GEPA repo](https://github.com/gepa-ai/gepa) for details.

## When to Use GEPA in Optimas
- You want to optimize prompts or other text components in a modular AI system.
- You want to leverage LLM-based reflection and feedback for prompt evolution.
- You are using DSPy modules (recommended, easiest integration), or you are an advanced user with a custom text-based system.

## Using GEPA with DSPy (Default Integration)
Optimas natively supports GEPA as a prompt optimizer for DSPy-based components. To use it:

1. Set `prompt_optimizer: gepa` in your config or CLI arguments.
2. (Optional) Configure GEPA-specific options, e.g.:
   ```yaml
   prompt_optimizer: gepa
   gepa_auto: medium  # or set gepa_max_metric_calls, gepa_num_iters, etc.
   gepa_reflection_minibatch_size: 5
   gepa_log_dir: ./gepa_logs
   gepa_use_wandb: true
   gepa_wandb_api_key: "your_wandb_api_key"
   gepa_wandb_init_kwargs:
     project: "my-gepa-project"
     entity: "my-wandb-entity"
   ```
3. Run your Optimas pipeline as usual. GEPA will optimize all DSPy-based components.

## Using GEPA with a Custom Adapter (Advanced)
If you want to optimize a non-DSPy system (e.g., your own text-based pipeline), you can implement a custom `GEPAAdapter`:

1. Implement the `GEPAAdapter` interface (see [gepa/core/adapter.py](https://github.com/gepa-ai/gepa/blob/main/src/gepa/core/adapter.py)). Your adapter must provide:
   - `evaluate(batch, candidate, capture_traces)`
   - `make_reflective_dataset(candidate, eval_batch, components_to_update)`
   - (Optional) `propose_new_texts(...)`
2. Modify or subclass the Optimas optimizer to inject your adapter instance when calling GEPA.
3. Pass your adapter and config as needed. Example (Python):
   ```python
   from gepa.adapters.default_adapter import DefaultAdapter
   my_adapter = DefaultAdapter(model="openai/gpt-4o")
   # ...
   # In your optimizer logic:
   gepa_result = gepa.optimize(
       seed_candidate=seed,
       trainset=trainset,
       valset=valset,
       adapter=my_adapter,
       # ...other config...
   )
   ```
4. See the [GEPA documentation](https://github.com/gepa-ai/gepa) for more on adapters.

## Configuring Logging and Experiment Tracking
- Use `gepa_logger` to pass a custom logger instance (advanced).
- Use `gepa_use_wandb`, `gepa_wandb_api_key`, and `gepa_wandb_init_kwargs` to control Weights & Biases logging.
- Example YAML:
  ```yaml
  gepa_use_wandb: true
  gepa_wandb_api_key: "your_wandb_api_key"
  gepa_wandb_init_kwargs:
    project: "my-gepa-project"
    entity: "my-wandb-entity"
  ```

## Example: Minimal DSPy GEPA Config
```yaml
prompt_optimizer: gepa
gepa_auto: light
gepa_log_dir: ./gepa_logs
```

## Further Reading
- [GEPA Documentation](https://github.com/gepa-ai/gepa)
- [DSPy GEPAAdapter Example](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/gepa/gepa_utils.py)
- [Optimas README](../README.md)
