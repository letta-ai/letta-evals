# Letta Evals

Letta Evals provides a framework for evaluating AI agents built with [Letta](https://github.com/letta-ai/letta). We offer a flexible evaluation system to test different dimensions of agent behavior and the ability to write your own custom evals for use cases you care about. You can use your own datasets to build private evals that represent common patterns in your agentic workflows.

<img width="596" src="https://github.com/user-attachments/assets/4471f0b0-8353-48b7-8f52-b51bbf0482cb" alt="Letta Evals running an evaluation suite with real-time progress tracking" width="800">

If you are building with agentic systems, creating high quality evals is one of the most impactful things you can do. Without evals, it can be very difficult and time intensive to understand how agent configurations, model versions, or prompt changes might affect your use case. In the words of [OpenAI's President Greg Brockman](https://twitter.com/gdb/status/1733553161884127435):

<img width="596" alt="https://x.com/gdb/status/1733553161884127435?s=20" src="https://github.com/openai/evals/assets/35577566/ce7840ff-43a8-4d88-bb2f-6b207410333b">

## Setup

To run evals against Letta agents, you will need a running Letta server. You can either:

* **Self-hosted**: Follow the [Letta installation guide](https://docs.letta.com/guides/ade/desktop#self-hosted-server-mode-recommended) to get started with self-hosting your server.
* **Letta Cloud**: Create an account at [app.letta.com](https://app.letta.com) and configure your environment:
  ```bash
  export LETTA_API_KEY=your-api-key        # Get from Letta Cloud dashboard
  export LETTA_PROJECT_ID=your-project-id  # Get from Letta Cloud dashboard

  Then set `base_url: https://api.letta.com/` in your suite YAML.

If you plan to use LLM-based grading (rubric graders), you'll also need to configure API keys for your chosen provider (e.g., `OPENAI_API_KEY`).

**Minimum Required Version: Python 3.9**

### Installing Letta Evals

If you are going to be creating custom evals or contributing to this repository, clone the repo directly from GitHub and install using:

```bash
# we recommend uv
uv sync --extra dev
```

Using the editable install, changes you make to your evals will be reflected immediately without having to reinstall.

### Running Evals Only

If you simply want to run existing evals locally, you can install the package via pip:

```bash
pip install letta-evals
```

## Quick Start

1. **Create a test dataset** (`dataset.jsonl`):
```jsonl
{"input": "What's the capital of France?", "ground_truth": "Paris"}
{"input": "Calculate 2+2", "ground_truth": "4"}
```

2. **Write a suite configuration** (`suite.yaml`):
```yaml
name: my-eval-suite
dataset: dataset.jsonl
target:
  kind: letta_agent
  agent_file: my_agent.af  # or use agent_id for existing agents
  base_url: http://localhost:8283
graders:
  quality:
    kind: tool
    function: contains  # or exact_match
    extractor: last_assistant
gate:
  kind: simple
  metric_key: quality
  aggregation: avg_score
  op: gte
  value: 0.75  # require average score >= 0.75
```

3. **Run the evaluation**:
```bash
letta-evals run suite.yaml
```

## Running Evals

The core evaluation flow is:

**Dataset → Target (Agent) → Extractor → Grader → Gate → Result**

```bash
# run an evaluation suite with real-time progress
letta-evals run suite.yaml

# save results to a directory (header.json, summary.json, results.jsonl)
letta-evals run suite.yaml --output results

# run multiple times for statistical analysis
letta-evals run suite.yaml --num-runs 5

# validate suite configuration before running
letta-evals validate suite.yaml

# list available components
letta-evals list-extractors
letta-evals list-graders
```

See the [`examples/`](examples/) directory for complete working examples of different eval types.

## Writing Evals

Letta Evals supports multiple approaches for creating evaluations, from simple YAML-based configs to fully custom Python implementations.

### Getting Started

We suggest getting started with these examples:

- **Basic tool grading**: [`examples/simple-tool-grader/`](examples/simple-tool-grader/) - Simple string matching with `exact_match` and `contains` functions
- **LLM-as-judge grading**: [`examples/simple-rubric-grader/`](examples/simple-rubric-grader/) - Using rubric graders with custom prompts for nuanced evaluation
- **Agent-as-judge grading**: [`examples/letta-agent-rubric-grader/`](examples/letta-agent-rubric-grader/) - Using a Letta agent as an LLM judge (no API keys required!)
- **Multi-grader gates**: [`examples/multi-grader-gate/`](examples/multi-grader-gate/) - Combining multiple graders with logical AND/OR gates, weighted averages, and advanced aggregation functions
- **Memory block extraction**: [`examples/multiturn-memory-block-extractor/`](examples/multiturn-memory-block-extractor/) - Extracting and evaluating agent memory across multiturn conversations
- **Multi-model evaluation**: [`examples/multi-model-simple-rubric-grader/`](examples/multi-model-simple-rubric-grader/) - Testing across multiple LLM configurations
- **Programmatic agent creation**: [`examples/programmatic-agent-creation/`](examples/programmatic-agent-creation/) - Using agent factories to create agents dynamically per sample
- **Custom graders and extractors**: [`examples/custom-tool-grader-and-extractor/`](examples/custom-tool-grader-and-extractor/) - Implementing custom evaluation logic with Python decorators
- **Letta Code CLI evaluation**: [`examples/letta-code-simple-edit/`](examples/letta-code-simple-edit/) - Testing autonomous coding agents with async graders and subprocess execution, including multi-model evaluation support

### Writing Custom Components

Letta Evals provides Python decorators for extending the framework:

- **@grader**: Register custom scoring functions for domain-specific evaluation logic
- **@extractor**: Create custom extractors to parse agent responses in specialized ways
- **@agent_factory**: Define programmatic agent creation for dynamic instantiation per sample
- **@suite_setup**: Run initialization code before evaluation starts. Supports three signatures:
  - `() -> None` - Run once at the start with no parameters
  - `(client: AsyncLetta) -> None` - Run once at the start with client access
  - `(client: AsyncLetta, model_name: str) -> None` - Run once per model when evaluating multiple models (useful for model-specific setup like creating isolated working directories)

See [`examples/custom-tool-grader-and-extractor/`](examples/custom-tool-grader-and-extractor/) for implementation examples.

## FAQ

**Do you have examples of different eval types?**

* Yes! See the [`examples/`](examples/) directory. Each subdirectory contains a complete working example with dataset, suite config, and any custom components.

**Can I use this without writing any Python code?**

* Absolutely! You can create powerful evals using just YAML configs and JSONL datasets. See [`examples/simple-tool-grader/`](examples/simple-tool-grader/) or [`examples/simple-rubric-grader/`](examples/simple-rubric-grader/) for code-free examples.

**How do I evaluate multi-turn agent interactions?**

* Letta Evals natively supports multiturn conversations! Simply provide `input` as a list of strings in your dataset instead of a single string. The framework will send each message sequentially and capture the full trajectory. Use extractors like `last_turn`, `all_assistant`, or `memory_block` to evaluate different aspects of the multiturn interaction. See [`examples/multiturn-memory-block-extractor/`](examples/multiturn-memory-block-extractor/) for a complete example testing memory updates across conversation turns.

**Can I test the same agent with different LLM models?**

* Yes! Use the multi-model configuration feature. See [`examples/multi-model-simple-rubric-grader/`](examples/multi-model-simple-rubric-grader/) for an example that tests one agent with multiple model configurations.

**Can I run evaluations multiple times to measure consistency?**

* Yes! Run evaluations multiple times to measure consistency and variance. See [`examples/simple-tool-grader/multi_run_tool_output_suite.yaml`](examples/simple-tool-grader/multi_run_tool_output_suite.yaml) for an example.

  ```bash
  # run 5 times and get mean/std dev statistics
  letta-evals run suite.yaml --num-runs 5 --output results/
  ```

  Results include aggregate statistics across runs with mean and standard deviation for all metrics.

**Can I monitor long-running evaluations in real-time?**

* Yes! Results are written incrementally as JSONL, allowing you to monitor evaluations in real-time and resume interrupted runs.

**Can I reuse agent trajectories when testing different graders?**

* Yes! Use `--cached-results` to reuse agent trajectories across evaluations, avoiding redundant agent runs when testing different graders.

**Can I evaluate Letta Code agents across different models?**

* Yes! The Letta Code target supports evaluating multiple models. In your suite YAML, specify multiple model handles:
  ```yaml
  target:
    kind: letta_code
    model_handles:
      - anthropic/claude-sonnet-4-5-20250929
      - gpt-5-low
  ```
  The framework automatically creates isolated working directories for each model to prevent interference between concurrent evaluations. When combined with `@suite_setup` functions that accept `model_name`, you can perform model-specific initialization for each evaluation run.

**Can I use this in CI/CD pipelines?**

* Absolutely! Letta Evals is designed to integrate seamlessly into continuous integration workflows. Check out our [`.github/workflows/e2e-tests.yml`](.github/workflows/e2e-tests.yml) for an example of running evaluations in GitHub Actions. The workflow automatically discovers and runs all suite files, making it easy to gate releases or validate changes to your agents.

**I don't have access to LLM provider API keys - can I still use LLM-as-judge / rubric grading?**

* Yes! Use the **agent-as-judge** feature instead of the standard rubric grader. With agent-as-judge, you configure a Letta agent (with its own LLM access) to act as the evaluator. This is perfect for:
  - Teams without direct LLM API access (using Letta Cloud or managed instances)
  - Scenarios where you want the judge to use tools (e.g., web search, database queries) during evaluation
  - Organizations with centralized LLM access through Letta

  See [`examples/letta-agent-rubric-grader/`](examples/letta-agent-rubric-grader/) for a complete working example. The judge agent just needs a `submit_grade(score: float, rationale: str)` tool, and the framework handles the rest!

## Contributing

Contributions are welcome! If you have an interesting eval or feature, please submit an issue or contact us on [Discord](https://discord.gg/letta).

## License

This project is licensed under the MIT License. By contributing to evals, you are agreeing to make your evaluation logic and data under the same MIT license as this repository. You must have adequate rights to upload any data used in an eval. Letta reserves the right to use this data in future service improvements to our product.
