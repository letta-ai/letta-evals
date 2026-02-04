"""Custom grader for user-memory-update eval.

Computes a text diff between initial and final user memory,
then asks an LLM judge to score against a rubric.

Uses kind: tool so the grader has access to sample.extra_vars
(initial_memory), avoiding duplication into rubric_vars.
"""

import difflib
import json
import os

from openai import AsyncOpenAI

from letta_evals.decorators import grader
from letta_evals.models import GradeResult, Sample

JUDGE_SYSTEM_PROMPT = """\
You are an evaluation judge. Return a JSON response with:
{"score": <float>, "rationale": "<explanation>"}
The score MUST be one of [0, 0.25, 0.5, 0.75, 1.0] based on how many criteria pass."""

JUDGE_PROMPT = """\
You are evaluating whether an AI agent correctly updated a user's memory block.

The agent had a memory block with structured facts about a user. The user sent a message \
with new information. The agent should have updated the memory to reflect that information \
and preserved everything else unchanged.

Each criterion is binary (0 or 0.25). Final score is one of [0, 0.25, 0.5, 0.75, 1].

Criteria:
1. **Memory was updated**: The final memory is not identical to the initial memory. \
If empty or unchanged, this fails.
2. **New information incorporated**: Key facts from the user's message appear in the \
final memory.
3. **Outdated information replaced**: Facts contradicted by the user's message have been \
removed or replaced. Old values should not remain alongside new ones.
4. **Unrelated facts preserved**: Everything not related to the update remains intact. \
No unrelated deletions, additions, or unnecessary rewording. Be strict: any unnecessary \
changes to unrelated parts (rewording, reordering, reformatting) should fail this criterion.

## User's Message
{user_message}

## Initial Memory (before)
```
{initial_memory}
```

## Final Memory (after)
```
{final_memory}
```

## Diff (- = removed, + = added)
```
{diff}
```

Return JSON: {{"score": <float>, "rationale": "<explanation>"}}"""


@grader
async def user_memory_grader(sample: Sample, submission: str) -> GradeResult:
    """Grade user memory update by computing a diff and asking an LLM judge."""
    extra = sample.extra_vars or {}
    initial_memory = extra.get("initial_memory", "")
    final_memory = submission

    diff_lines = list(
        difflib.unified_diff(
            initial_memory.splitlines(keepends=True),
            final_memory.splitlines(keepends=True),
            fromfile="before",
            tofile="after",
        )
    )
    diff_text = "".join(diff_lines) if diff_lines else "(no textual changes detected)"

    prompt = JUDGE_PROMPT.format(
        user_message=sample.input,
        initial_memory=initial_memory,
        final_memory=final_memory,
        diff=diff_text,
    )

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = await client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    score = max(0.0, min(1.0, float(result["score"])))
    rationale = result.get("rationale", "")

    return GradeResult(
        score=score,
        rationale=rationale,
        metadata={"model": "gpt-5-mini", "diff": diff_text},
    )
