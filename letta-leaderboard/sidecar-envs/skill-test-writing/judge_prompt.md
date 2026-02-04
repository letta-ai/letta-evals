## Task

Evaluate a test case for the **{skill_name}** skill. A good test measures whether skill knowledge actually helps; a bad test is trivia or has a broken grader.

## Skill Files

```
{skill_file_tree}
```

## Model Output

{model_output}

## Scoring (1-10 each)

### Non-Obviousness

Does this test transferable domain knowledge, not skill-specific trivia?

| Score | Criteria |
|-------|----------|
| 8-10 | Tests a genuine gotcha/pattern that transfers across contexts |
| 5-7 | Tests useful knowledge; specific details are incidental |
| 3-4 | Could be general but overly focused on specific strings/names |
| 1-2 | Pure trivia: exact script names, magic numbers, memorization |

**Score 1-3 if:** Test requires knowing arbitrary identifiers (e.g., "thumbnail.py", "cais/mmlu", "rebuild every 30 minutes") rather than underlying concepts.

### Clarity

Is the grader robust? Does it accept all valid solutions?

| Score | Criteria |
|-------|----------|
| 8-10 | Tests actual capability, accepts multiple valid approaches |
| 5-7 | Mostly robust, minor edge cases |
| 3-4 | Significant brittleness or missing requirements |
| 1-2 | Fundamentally broken grader |

**Score 1-3 if ANY apply:**

- Regex/string matching that rejects valid solutions
- Only accepts one approach when alternatives exist (e.g., only openpyxl, not xlsxwriter)
- Requires files/servers/setup not provided
- Asks to "run" code that would fail

### Realism

Would a practitioner actually do this task?

| Score | Criteria |
|-------|----------|
| 8-10 | Common real-world task |
| 5-7 | Realistic scenario |
| 3-4 | Plausible but contrived |
| 1-2 | Artificial/synthetic task |

**Score 1-3 if:** Task exists only to test the skill, not because anyone would actually need to do it (e.g., "list all frontmatter fields" vs "create a command for X").

## Output

JSON only. Keep rationale to 1-2 sentences.

```json
{{
  "rationale": "<1-2 sentences>",
  "non_obviousness": <1-10>,
  "clarity": <1-10>,
  "realism": <1-10>
}}
```
