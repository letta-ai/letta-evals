# Simple Rubric Grader Example

This example demonstrates LLM-as-judge evaluation using detailed rubric criteria.

## What This Example Shows

- Using rubric graders for subjective quality evaluation
- Multi-dimensional scoring criteria (instruction compliance, accuracy, craftsmanship, effort)
- Running multiple graders in a single suite (`suite.two-metrics.yaml`)
- Combining rubric grading with tool grading for comprehensive evaluation

## Key Takeaway

Rubric graders use an LLM to evaluate agent responses against detailed criteria, making them ideal for subjective qualities like creativity, quality, or compliance. The rubric in `rubric.txt` defines clear scoring dimensions that sum to 1.0.

## Running This Example

### Local Setup (Default)

Start your local Letta server:
```bash
letta server
```

Set your API keys (required for rubric grader):
```bash
# For suite.yaml (uses Anthropic Claude)
export ANTHROPIC_API_KEY=your-key

# For suite.two-metrics.yaml and suite.ascii-only-accuracy.yaml (use OpenAI GPT)
export OPENAI_API_KEY=your-key
```

Then run the evaluation:
```bash
cd examples/simple-rubric-grader
letta-evals run suite.yaml
```

### Letta Cloud Setup

Set these environment variables:
```bash
export LETTA_API_KEY=your-api-key
export LETTA_PROJECT_ID=your-project-id

# Depending on which suite you're running:
export ANTHROPIC_API_KEY=your-anthropic-key  # for suite.yaml
export OPENAI_API_KEY=your-openai-key  # for suite.two-metrics.yaml and suite.ascii-only-accuracy.yaml
```

Update `base_url` in `suite.yaml`:
```yaml
target:
  base_url: https://api.letta.com/
```

Then run the evaluation as above.

## Custom Rubric Variables

This example demonstrates using **custom rubric variables** to pass per-sample context into the rubric. Each sample in `dataset.jsonl` includes a `rubric_vars` field with a `reference_ascii` containing high-quality reference ASCII art:

```json
{
  "input": "Send me your best version of ASCII art representing cat...",
  "rubric_vars": {
    "reference_ascii": " /\\_/\\\n( o.o )\n > ^ <"
  }
}
```

The suite configuration declares which variables are required:

```yaml
graders:
  quality:
    kind: rubric
    prompt_path: rubric.txt
    rubric_vars:
      - reference_ascii
```

These variables are then substituted into the rubric template using `{variable_name}` syntax, allowing the LLM judge to compare submissions against reference examples.

## Rubric Structure

The rubric in `rubric.txt` defines multi-dimensional scoring criteria with a reference example section:

```
Here is an example of a high-quality ASCII art for this task:

{reference_ascii}

Use this as a reference for quality, detail level, and craftsmanship when evaluating the submission.

---

## Scoring dimensions (sum → 1.0)
1. Instruction compliance (0.30)
   0.30: Art wrapped in backticks as requested; exactly one send_message; no extra content
   0.15: Art present but formatting issues
   0.00: No backticks, extra commentary, or mixed content

2. Requested item match (0.30)
   0.30: Depicts the requested animal with unmistakable features
   0.15: Vaguely animal-like or generic creature
   0.00: Wrong subject or unrelated shapes

3. Recognizability and silhouette (0.20)
   0.20: Clear silhouette; core anatomy reads at a glance
   0.10: Recognizable with effort
   0.00: Unrecognizable

4. Craftsmanship and layout (0.10)
   0.10: Good alignment and spacing; holds shape in monospaced render
   0.05: Minor alignment issues
   0.00: Misaligned or broken

5. Effort / "best version" signal (0.10)
   0.10: Shows detail and clear attempt at quality
   0.05: Minimal but functional
   0.00: Trivial or careless

IMPORTANT: Make sure to clearly express your rationale and scoring.
```

**Key points:**
- Each dimension has multiple scoring tiers with specific criteria
- Dimensions should sum to 1.0 for normalized scoring
- Use `{variable_name}` placeholders for custom rubric variables declared in the suite config
- Clear rationale expectations help produce consistent LLM judgments

## Suite Configuration

### Single Rubric Grader (`suite.yaml`)

```yaml
graders:
  quality:
    kind: rubric
    prompt_path: rubric.txt
    model: claude-haiku-4-5-20251001
    temperature: 0.0
    provider: anthropic
    max_retries: 3
    timeout: 120.0
    extractor: last_assistant
    rubric_vars:
      - reference_ascii  # declares required custom variables
```

### Multiple Graders (`suite.two-metrics.yaml`)

Combine rubric and tool graders for comprehensive evaluation:

```yaml
graders:
  quality:
    kind: rubric
    display_name: "rubric score"
    prompt_path: rubric.txt
    model: gpt-5-mini
    temperature: 0.0
    provider: openai
    extractor: last_assistant
  ascii_only:
    kind: tool
    display_name: "ascii character check"
    function: ascii_printable_only
    extractor: last_assistant
```

**Key points:**
- Use `display_name` for prettier output in results
- `temperature: 0.0` for more consistent grading
- `max_retries` and `timeout` handle LLM API issues
- Gate can reference any grader by its `metric_key`

## Variants

- `suite.yaml`: Single rubric grader for quality assessment (uses Claude Haiku via Anthropic)
- `suite.two-metrics.yaml`: Combines rubric grader (quality) + tool grader (ASCII character validation) (uses GPT-5-mini via OpenAI)
- `suite.ascii-only-accuracy.yaml`: Example showing different gate configurations (uses GPT-5-mini via OpenAI)
