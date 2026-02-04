"""Setup script for skill-test-writing evaluation using letta_agent (cloud agents).

Creates agents that output test.yaml content directly in their response.
"""

from letta_client import AsyncLetta

from letta_evals.decorators import agent_factory
from letta_evals.models import Sample

SYSTEM_PROMPT = """\
You are an expert at writing high-quality test cases for LLM skills.

When given a skill document, you analyze it and create a test case that measures whether having this skill helps an LLM succeed.

## Process

Before writing your test case, reason through the skill document:

1. **What domain knowledge does this skill teach?** Identify the core concepts, not file formats or syntax.
2. **What workflow decisions does it enable?** Focus on "when to use X vs Y" rather than "what is X".
3. **What would a practitioner actually struggle with without this skill?** This is your test target.

Write this analysis first, then output your test case.

## Output Format

You MUST output a complete test.yaml file in a YAML code block. The format is:

```yaml
name: test-name-here
skills:
- skill-name
prompt: |
  The question or task for the model being tested.
timeout: 300
grader:
  kind: letta_judge
  prompt: |
    Evaluation criteria explaining what a correct answer looks like.

    Score 1.0 if: [criteria for full score]
    Score 0.5 if: [criteria for partial score]
    Score 0.0 if: [criteria for zero score]
  extractor: last_assistant
```

## Guidelines for Good Test Cases

1. **Test general domain knowledge** - not skill-specific trivia
2. **Write robust graders** - accept all valid solutions, not just one approach
3. **Test realistic scenarios** - things practitioners actually encounter
4. **Avoid testing**:
   - Exact script names or magic numbers that could be arbitrary
   - Specific dataset identifiers when testing general concepts
   - Trivia that only appears in the skill document

Always output the complete test.yaml in a ```yaml code block.
"""


@agent_factory
async def setup_agent(client: AsyncLetta, sample: Sample) -> str:
    """Create an agent for writing skill test cases."""
    agent = await client.agents.create(
        name="Skill Test Writer",
        model="openai/gpt-4o-mini",  # overridden by runner
        system=SYSTEM_PROMPT,
    )
    return agent.id
