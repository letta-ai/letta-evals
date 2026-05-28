# Custom Tool Grader and Extractor Example

This example demonstrates custom Python extractors and graders with the `letta_code` target.

The task asks the agent to classify support tickets as JSON. The custom extractor pulls the JSON object out of the last assistant response, and the custom grader compares the parsed `category` and `priority` fields against the dataset ground truth.

```bash
cd examples/custom-tool-grader-and-extractor
letta-evals run suite.yaml
```

Key files:

- `custom_evaluators.py` — `@extractor` and `@grader` implementations
- `dataset.jsonl` — ticket prompts plus structured ground truth
- `suite.yaml` — references custom functions with `file.py:function_name`
