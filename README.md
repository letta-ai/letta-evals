# Letta Evals

Letta Evals provides a framework for evaluating AI agents built with [Letta](https://github.com/letta-ai/letta). We offer a flexible evaluation system to test different dimensions of agent behavior and the ability to write your own custom evals for use cases you care about. You can use your own datasets to build private evals that represent common patterns in your agentic workflows.

<img width="596" src="https://raw.githubusercontent.com/letta-ai/letta-evals/refs/heads/main/docs/assets/evaluation-progress.png?token=GHSAT0AAAAAACYIKB3XHC4RELKFG7VSECAM2HX7NJA" alt="Letta Evals running an evaluation suite with real-time progress tracking" width="800">

If you are building with agentic systems, creating high quality evals is one of the most impactful things you can do. Without evals, it can be very difficult and time intensive to understand how agent configurations, model versions, or prompt changes might affect your use case. In the words of [OpenAI's President Greg Brockman](https://twitter.com/gdb/status/1733553161884127435):

<img width="596" alt="https://x.com/gdb/status/1733553161884127435?s=20" src="https://github.com/openai/evals/assets/35577566/ce7840ff-43a8-4d88-bb2f-6b207410333b">

## Setup

To run evals against Letta agents, you will need a running Letta server. You can either:
* Follow the [Letta installation guide](https://docs.letta.com/guides/ade/desktop#self-hosted-server-mode-recommended) to get started with self-hosting your server.
* Use [Letta Cloud](https://app.letta.com) - create an account and configure your suite with the Letta Cloud API endpoint and your API key.

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
  kind: agent
  agent_file: my_agent.af  # or use agent_id for existing agents
  base_url: http://localhost:8283
graders:
  quality:
    kind: tool
    function: contains  # or exact_match
    extractor: last_assistant
gate:
  metric_key: quality
  op: gte
  value: 0.75  # require 75% pass threshold
```

3. **Run the evaluation**:
```bash
letta-evals run suite.yaml
```

## Running Evals

You can find the full evaluation flow documentation in [`CLAUDE.md`](CLAUDE.md). The core evaluation flow is:

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
- **Multi-metric evaluation**: [`examples/simple-rubric-grader/suite.two-metrics.yaml`](examples/simple-rubric-grader/suite.two-metrics.yaml) - Combining multiple graders (rubric + tool) in one suite
- **Custom extractors**: [`examples/simple-memory-block-extractor/`](examples/simple-memory-block-extractor/) - Extracting specific content from agent memory blocks
- **Multi-model evaluation**: [`examples/multi-model-simple-rubric-grader/`](examples/multi-model-simple-rubric-grader/) - Testing across multiple LLM configurations
- **Programmatic agent creation**: [`examples/programmatic-agent-creation/`](examples/programmatic-agent-creation/) - Using agent factories to create agents dynamically per sample
- **Custom graders and extractors**: [`examples/custom-tool-grader-and-extractor/`](examples/custom-tool-grader-and-extractor/) - Implementing custom evaluation logic with Python decorators

### Writing Custom Components

Letta Evals provides Python decorators for extending the framework:

- **@grader**: Register custom scoring functions for domain-specific evaluation logic
- **@extractor**: Create custom extractors to parse agent responses in specialized ways
- **@agent_factory**: Define programmatic agent creation for dynamic instantiation per sample
- **@suite_setup**: Run initialization code before evaluation starts

See [`examples/custom-tool-grader-and-extractor/`](examples/custom-tool-grader-and-extractor/) for implementation examples.

## FAQ

**Do you have examples of different eval types?**

> Yes! See the [`examples/`](examples/) directory. Each subdirectory contains a complete working example with dataset, suite config, and any custom components.

**Can I use this without writing any Python code?**

> Absolutely! You can create powerful evals using just YAML configs and JSONL datasets. See [`examples/simple-tool-grader/`](examples/simple-tool-grader/) or [`examples/simple-rubric-grader/`](examples/simple-rubric-grader/) for code-free examples.

**How do I evaluate multi-turn agent interactions?**

> Letta agents inherently support multi-turn conversations. Use extractors like `all_messages` or `tool_calls` to capture the full interaction trajectory, not just the final response.

**Can I test the same agent with different LLM models?**

> Yes! Use the multi-model configuration feature. See [`examples/multi-model-simple-rubric-grader/`](examples/multi-model-simple-rubric-grader/) for an example that tests one agent with multiple model configurations.

**Can I run evaluations multiple times to measure consistency?**

> Yes! Run evaluations multiple times to measure consistency and variance. See [`examples/simple-tool-grader/multi_run_tool_output_suite.yaml`](examples/simple-tool-grader/multi_run_tool_output_suite.yaml) for an example.
>
> ```bash
> # run 5 times and get mean/std dev statistics
> letta-evals run suite.yaml --num-runs 5 --output results/
> ```
>
> Results include aggregate statistics across runs with mean and standard deviation for all metrics.

**Can I monitor long-running evaluations in real-time?**

> Yes! Results are written incrementally as JSONL, allowing you to monitor evaluations in real-time and resume interrupted runs.

**Can I reuse agent trajectories when testing different graders?**

> Yes! Use `--cached-results` to reuse agent trajectories across evaluations, avoiding redundant agent runs when testing different graders.

**Can I use this in CI/CD pipelines?**

> Absolutely! Letta Evals is designed to integrate seamlessly into continuous integration workflows. Check out our [`.github/workflows/e2e-tests.yml`](.github/workflows/e2e-tests.yml) for an example of running evaluations in GitHub Actions. The workflow automatically discovers and runs all suite files, making it easy to gate releases or validate changes to your agents.

## Contributing

Contributions are welcome! If you have an interesting eval or feature, please submit an issue or contact us on [Discord](https://discord.gg/letta).

## License

This project is licensed under the MIT License. By contributing to evals, you are agreeing to make your evaluation logic and data under the same MIT license as this repository. You must have adequate rights to upload any data used in an eval. Letta reserves the right to use this data in future service improvements to our product.
