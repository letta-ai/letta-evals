# Core Memory Read Evaluation

This directory contains a core memory reading evaluation implemented using the letta-evals-kit framework.

## Structure

```
core-memory-read-agent/
├── datasets/
│   └── core_memory_read.jsonl          # Converted dataset from letta_bench_gen_200.jsonl
├── suites/
│   ├── core-memory-read.yaml           # Main evaluation suite config
│   └── core-memory-read-simple.yaml    # Simplified version for testing
├── core-memory-read-agent.af           # Agent configuration file
├── core_memory_evaluators.py           # Custom graders and extractors
├── setup_agent.py                      # Agent setup script for memory population
├── convert_dataset.py                  # Script to convert original dataset
└── README.md                           # This file
```

## Dataset

The dataset contains 1100 samples converted from the original letta_bench_gen_200.jsonl format. Each sample includes:
- `input`: A question about people and facts
- `ground_truth`: The expected answer
- `metadata`: Contains the supporting facts that should be stored in core memory

## Agent Configuration

The agent is configured with:
- A "Supporting Facts" core memory block
- Tools for memory operations and conversation
- Instructions to read from core memory to answer questions

## Custom Evaluators

- `CoreMemoryResponseExtractor`: Extracts the final assistant response
- `grade_core_memory_read`: Flexible grader that allows partial matches
- `grade_core_memory_read_strict`: Strict grader requiring exact matches

## Commands to Run

### Prerequisites

1. Make sure you have the letta-evals-kit installed and set up
2. Ensure a Letta server is running at `http://localhost:8283`
3. Set up your OpenAI API key for the agent

### Basic Commands

1. **Validate the suite configuration:**
   ```bash
   # From the letta-evals-kit directory
   cd /Users/kevinlin/Documents/letta-evals-kit
   letta-evals validate ../letta-leaderboard/core-memory-read-agent/suites/core-memory-read-simple.yaml
   ```

2. **Run the simple evaluation (10 samples):**
   ```bash
   letta-evals run ../letta-leaderboard/core-memory-read-agent/suites/core-memory-read-simple.yaml
   ```

3. **Run the full evaluation (100 samples):**
   ```bash
   letta-evals run ../letta-leaderboard/core-memory-read-agent/suites/core-memory-read.yaml
   ```

4. **Run with output saved to file:**
   ```bash
   letta-evals run ../letta-leaderboard/core-memory-read-agent/suites/core-memory-read-simple.yaml --output results.json
   ```

5. **Run in quiet mode (only show pass/fail):**
   ```bash
   letta-evals run ../letta-leaderboard/core-memory-read-agent/suites/core-memory-read-simple.yaml --quiet
   ```

### Advanced Usage

- Modify `max_samples` in the YAML file to test different numbers of samples
- Adjust the `gate.value` threshold to change the pass/fail criteria
- Switch between the flexible and strict graders by changing the grader function in the YAML

## Example Sample

```json
{
  "input": "Which person collaborated with Eliza Woodley on a charity event that combined music and literature?",
  "ground_truth": "Nora Fishel",
  "metadata": {
    "facts": [
      "Nora Fishel is an accomplished violinist who has performed in orchestras across Europe.",
      "Eliza Woodley is a published author known for her historical fiction novels.",
      "Nora Fishel and Eliza Woodley collaborated on a charity event that combined music and literature."
    ]
  }
}
```

The agent should read the facts from its core memory and correctly answer "Nora Fishel".