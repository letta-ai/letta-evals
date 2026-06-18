# Multi-Model Rubric Grader Example

This example demonstrates evaluating the same tasks across multiple models to compare performance.

## What This Example Shows

- Using `model_handles` to test multiple models in a single suite
- Comparing performance between different model capabilities (e.g., GPT-4 vs GPT-4-mini)
- Applying the same rubric consistently across models
- A/B testing for model selection or regression testing

## Key Takeaway

By specifying multiple `model_handles`, you can compare how different models perform on the same evaluation tasks. The framework runs each (sample, model) pair separately and tracks results per model.

## Running This Example

### Local Setup (Default)

Start your local Letta server:
```bash
letta server
```

Set your OpenAI API key (required for both agent models and rubric grader):
```bash
export OPENAI_API_KEY=your-key
```

Then run the evaluation:
```bash
cd examples/multi-model-simple-rubric-grader
letta-evals run suite.yaml
```

### Letta Cloud Setup

Set these environment variables:
```bash
export LETTA_API_KEY=your-api-key
export LETTA_PROJECT_ID=your-project-id
export OPENAI_API_KEY=your-openai-key
```

Update `base_url` in `suite.yaml`:
```yaml
target:
  base_url: https://api.letta.com/
```

Then run the evaluation as above.

## Configuration Details

### Model Handles Setup

The `model_handles` field in `suite.yaml` specifies which models to test:

```yaml
target:
  kind: letta_code
  base_url: http://localhost:8283
  model_handles:
    - openai/gpt-4.1
    - openai/gpt-4.1-mini
```

**Key points:**
- Each model in `model_handles` creates separate evaluation runs
- Same evaluation suite, different underlying model
- Format: `provider/model-name` (e.g., `openai/gpt-4.1`, `anthropic/claude-3-5-sonnet-20241022`)
- All samples run against all models (N samples × M models = N×M evaluations)

### How It Works

1. The framework reads each model from `model_handles`
2. For each model, it runs all samples through the Letta Code target
3. Results include per-model metrics for comparison

### Results Output

Each model gets its own per-sample results file (`<model>.jsonl`), and `summary.json` contains a `models` array with one entry per model:

```json
{
  "suite": "multi-model-rubric-grader",
  "models": [
    {
      "model": "openai/gpt-4.1",
      "n_total": 20,
      "n_attempted": 20,
      "reward": 0.82,
      "per_metric": { "quality": 0.82 }
    },
    {
      "model": "openai/gpt-4.1-mini",
      "n_total": 20,
      "n_attempted": 20,
      "reward": 0.71,
      "per_metric": { "quality": 0.71 }
    }
  ]
}
```

This makes it easy to compare model performance and choose the best model for your use case.
