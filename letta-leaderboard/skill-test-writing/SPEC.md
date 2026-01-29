# Skill Test Writing Evaluation Suite

## Overview

This evaluation suite measures an agent's ability to write high-quality test cases for skills. Given a skill (SKILL.md + supporting files), the agent must produce test cases that effectively measure whether having that skill actually helps an LLM succeed at relevant tasks.

## Background

A "skill" is a document that provides an LLM with specialized knowledge about a domain (APIs, tools, workflows, gotchas). We create test cases to measure whether having such knowledge actually helps. This evaluation measures whether an agent can write such test cases.

## Reference Materials

The following reference materials are copied from `~/repos/skill-eval`:

- `reference_skills/` - Sample skills that tests can be written for
- `reference_tests/` - Example test cases showing the expected format
- `reference_judge.py` - The judge implementation from skill-eval
- `judge_prompt.txt` - The LLM judge rubric (copied verbatim)

## Test Case Format

Test cases are defined in `test.yaml` files with the following structure:

```yaml
name: my-test
skills:                   # Skills this test evaluates
  - skill-name
prompt: "Task prompt"     # What the model is asked to do
timeout: 300              # Timeout in seconds

grader:
  kind: letta_judge       # letta_judge | model_judge | tool
  prompt: "Evaluation criteria..."
  extractor: last_assistant
```

### Grader Types

1. **letta_judge** - Agent-as-judge evaluation
2. **model_judge** - LLM-as-judge with specific model
3. **tool** - Programmatic grading with functions like `contains`, `exact_match`, or custom `grader.py:func`

## Evaluation Rubric

The judge evaluates test cases on three dimensions (from `judge_prompt.txt`):

### 1. Non-Obviousness (1-10)

Does this test knowledge that practitioners genuinely need but models often lack?

Key distinction between trivia vs useful knowledge:

**TRIVIA (bad):**

- Exact script names from a specific codebase when the concept doesn't require knowing that name
- Arbitrary numeric values that could be any number
- Specific dataset identifiers when testing dataset handling concepts

**USEFUL KNOWLEDGE (good):**

- Tool-specific workflows that practitioners need to know
- Best practices even if "obvious" to experts
- Naming conventions and rules for a system
- Common patterns and syntax for tools
- Gotchas that bite practitioners repeatedly

Score guide:

- 10: Tests a genuine domain gotcha - a concept/pattern that transfers across tools and contexts
- 7: Tests useful domain knowledge that practitioners learn; specific details are incidental to the core concept
- 4: Tests something that COULD be general knowledge but the test is overly focused on specific strings/names
- 1: Tests pure trivia - the specific script name, magic number, or exact string IS the test, not the underlying concept

### 2. Clarity (1-10)

Is the grader robust? Does it accept all valid solutions?

CRITICAL: Score LOW (1-3) if ANY of these apply:

- **Fragile pattern matching**: Grader uses regex/string matching that could reject valid solutions or accept wrong ones
- **Single-path grader**: Only accepts one specific approach when multiple valid solutions exist
- **Missing environment**: Test requires files, servers, or setup that aren't provided
- **Unrunnable**: Test asks to "run" code but execution would fail

Score guide:

- 10: Grader tests the actual capability, accepts all valid approaches
- 7: Grader is mostly robust, minor edge cases possible
- 4: Grader has significant brittleness or specificity issues
- 1: Grader is fundamentally flawed - pattern matching that rejects valid solutions

### 3. Realism (1-10)

Would a practitioner actually encounter this task?

Score guide:

- 10: Common task that practitioners frequently encounter
- 7: Realistic scenario that comes up in actual work
- 4: Plausible but somewhat contrived
- 1: Artificial scenario designed to test the skill, not a real need

### 4. Evidence Quality (1-10, optional)

If evidence is provided, does it support that this tests a real capability gap?

- 10: Direct proof - shows model failure, error logs, user corrections, or clear documentation of the gotcha
- 7: Good supporting evidence that this is a real issue
- 4: Suggestive but not conclusive
- 1: Vague, irrelevant, or doesn't support the test's value

## Integration with letta-evals

### Dataset Format

The dataset is a CSV with columns:

- `sample_id` - Unique identifier
- `skill` - Which skill to write tests for
- `input` - The task prompt for the agent
- `rubric_vars` - JSON with rubric variables for grading

### Custom Extractor

The `custom_extractor.py` module provides:

- `test_files_extractor` - Extracts test.yaml and grader.py files from the sandbox
- Formats output for the LLM judge to evaluate

### Grader Configuration

```yaml
graders:
  test_quality:
    kind: model_judge
    prompt_path: judge_prompt.txt  # Copied verbatim from skill-eval
    model: claude-haiku-4-5-20251001
    extractor: custom_extractor.py:test_files_extractor
    rubric_vars:
      - skill_name
      - skill_files
```

## Task Generation

Tasks are generated by:

1. Selecting a skill from `reference_skills/`
2. Providing the skill's SKILL.md and file tree to the agent
3. Asking the agent to write comprehensive test cases
4. Evaluating the test cases against the rubric

## Success Criteria

A test case is considered good if it:

1. Tests **general domain knowledge** that practitioners in that field would know
2. Has a robust grader that accepts all valid solutions
3. Represents a capability that matters in real-world use

A test case is considered bad if it:

1. Tests **skill-specific trivia** (magic field names, arbitrary conventions only in that skill doc)
2. Has a fragile grader that rejects valid alternative approaches
3. Could only be solved by memorizing the specific skill document

## Directory Structure

```
skill-test-writing/
├── SPEC.md                    # This file
├── suite.yaml                 # Main eval configuration
├── judge_prompt.txt           # LLM judge rubric (verbatim from skill-eval)
├── custom_extractor.py        # Extracts test files for grading
├── data/
│   └── dataset.csv            # Evaluation samples
├── .skills/                   # Skills available to agents during eval
├── reference_skills/          # Sample skills (from skill-eval)
├── reference_tests/           # Example test cases (from skill-eval)
├── reference_judge.py         # Judge implementation (from skill-eval)
└── task_generator/            # Scripts to generate evaluation tasks
```
