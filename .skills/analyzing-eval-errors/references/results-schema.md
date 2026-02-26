# Eval Results Schema

## File Layout

A results directory from `letta_evals` contains:

```
results/
├── header.json      # Suite config, target config, model handles
├── results.jsonl    # One JSON line per sample result (streamed during eval)
└── summary.json     # Aggregate metrics, per-model breakdowns, error summaries
```

## results.jsonl

Each line is a JSON object with `{"type": "result", "result": {...}}`. The `result` object is a `SampleResult`:

```python
{
    "type": "result",
    "result": {
        "sample": {
            "id": 3,                    # int, 0-indexed sample ID
            "input": "...",             # str or list[str], the prompt(s)
            "ground_truth": "...",      # str, expected answer
        },
        "agent_id": "agent-xxxx",       # str, Letta agent ID
        "model_name": "minimax-m2.5",   # str, model handle
        "submission": "...",            # str, extracted agent response (empty if errored)
        "trajectory": [[...]],          # list of list of messages (usually empty in JSONL)
        "grade": {                      # present if grading succeeded
            "score": 0.0-1.0,
            "rationale": "..."
        },
        "error": null | {               # null if success, ErrorInfo if failed
            "category": "target",       # "target" | "extraction" | "grading"
            "exception_type": "RuntimeError",
            "message": "Letta command failed with return code 1. Stderr: "
        },
        # Token usage fields (nullable):
        "cost": 0.05,
        "prompt_tokens": 50000,
        "completion_tokens": 1200,
        "cached_input_tokens": 30000,
    }
}
```

### ErrorInfo categories

- **`target`**: The target (CLI subprocess) failed. Includes CLI crashes (rc != 0), timeouts, and connection errors.
- **`extraction`**: Agent ran but the extractor couldn't find a submission (e.g., no `assistant_message`).
- **`grading`**: Extraction succeeded but the grader failed.

### Common error messages

| Pattern | Meaning |
|---------|---------|
| `"Letta command failed with return code 1. Stderr: "` | CLI exited code 1, empty stderr — likely approval/streaming bug |
| `"Letta command timed out after N seconds"` | Hit the timeout limit |
| `"Empty submission - extractor found no content"` | Agent produced no assistant_message |

## summary.json

```python
{
    "type": "summary",
    "metrics": {
        "total": 149,
        "total_attempted": 111,
        "avg_score_attempted": 0.44,
        "avg_score_total": 0.33,
        "per_model": [
            {
                "model_name": "minimax-m2.5",
                "total": 49,
                "total_attempted": 19,
                "error_summary": {
                    "total_errors": 30,
                    "by_category": {"target": 30},
                    "by_exception_type": {"RuntimeError": 30},
                    "failed_sample_ids": [0, 3, 5, ...]
                }
            }
        ]
    },
    "gates_passed": false
}
```

## header.json

Contains the suite config and target config. Key fields:

```python
{
    "type": "header",
    "config": {
        "suite": "filesystem-code",
        "target": {
            "kind": "letta_code",          # or "letta_agent"
            "base_url": "https://api.letta.com/",
            "model_handles": ["minimax-m2.5", "glm-5"],
            "timeout": 600,
            "working_dir": "files",
        }
    }
}
```
