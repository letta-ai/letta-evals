#!/usr/bin/env python3
"""Generate dataset.jsonl for skill-test-writing evaluation.

For each skill in reference_skills/, creates a row asking the agent to write
a test case for that skill. Uses extra_vars to pass skill metadata to the judge.
"""

import json
from pathlib import Path

SKILLS_DIR = Path(__file__).parent / "reference_skills"
OUTPUT_FILE = Path(__file__).parent / "data" / "dataset.jsonl"

# Prompt template for generating test cases
PROMPT_TEMPLATE = """You are given a skill document that teaches an LLM specialized knowledge about a domain.

Your task: Write ONE high-quality test case that measures whether having this skill helps an LLM succeed.

## The Skill

**Skill Name:** {skill_name}

**SKILL.md Content:**
```markdown
{skill_content}
```

{additional_files_section}

## Test Case Format

Create a `test.yaml` file with this structure:

```yaml
name: descriptive-kebab-case-name
skills:
  - {skill_name}
prompt: |
  The task prompt that tests whether the model has the skill's knowledge.
  Should be a realistic task a practitioner would encounter.
timeout: 300

grader:
  kind: letta_judge  # or model_judge or tool
  prompt: |
    Evaluation criteria for scoring the response.
    Score 1.0 if: <criteria for full credit>
    Score 0.5 if: <criteria for partial credit>
    Score 0.0 if: <criteria for no credit>
  extractor: last_assistant
```

If you need programmatic grading, also create a `grader.py` file with a function that returns a score 0-1.

## Guidelines

**Good test cases:**
- Test domain knowledge that practitioners genuinely need
- Have robust graders that accept all valid solutions
- Represent realistic tasks

**Bad test cases:**
- Test skill-specific trivia (exact script names, magic numbers)
- Have fragile graders that reject valid alternatives
- Could only be solved by memorizing the skill document

## Your Output

Output your test.yaml content directly in a YAML code block:

```yaml
name: descriptive-kebab-case-name
skills:
  - {skill_name}
prompt: |
  Your test prompt here...
timeout: 300
grader:
  kind: letta_judge
  prompt: |
    Scoring criteria...
  extractor: last_assistant
```
"""


def read_skill_files(skill_dir: Path) -> tuple[str | None, str | None, list[tuple[str, str]]]:
    """Read SKILL.md and any additional files from a skill directory.

    Returns:
        skill_name: Name from frontmatter or directory name
        skill_content: Content of SKILL.md
        additional_files: List of (filename, content) for other relevant files
    """
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None, None, []

    skill_content = skill_md.read_text(encoding="utf-8")

    # Extract name from frontmatter if present
    skill_name = skill_dir.name
    if skill_content.startswith("---"):
        # Parse YAML frontmatter
        end_idx = skill_content.find("---", 3)
        if end_idx > 0:
            frontmatter = skill_content[3:end_idx]
            for line in frontmatter.split("\n"):
                if line.startswith("name:"):
                    skill_name = line.split(":", 1)[1].strip()
                    break

    # Read additional markdown files (not SKILL.md)
    additional_files = []
    for f in skill_dir.glob("*.md"):
        if f.name != "SKILL.md":
            try:
                content = f.read_text(encoding="utf-8")
                # Truncate if too long
                if len(content) > 5000:
                    content = content[:5000] + "\n... (truncated)"
                additional_files.append((f.name, content))
            except (UnicodeDecodeError, OSError) as e:
                print(f"Warning: Could not read {f.name}: {e}")

    return skill_name, skill_content, additional_files


def format_additional_files(files: list[tuple[str, str]]) -> str:
    """Format additional files section for the prompt."""
    if not files:
        return ""

    sections = ["## Additional Skill Files\n"]
    for filename, content in files[:2]:  # Limit to 2 additional files
        sections.append(f"**{filename}:**\n```markdown\n{content}\n```\n")

    return "\n".join(sections)


def get_skill_file_tree(skill_dir: Path) -> str:
    """Get a tree representation of skill directory files."""
    files = []
    for f in sorted(skill_dir.rglob("*")):
        if f.is_file() and not f.name.startswith("."):
            rel_path = f.relative_to(skill_dir)
            files.append(str(rel_path))
    return "\n".join(files[:20])  # Limit to 20 files


def generate_dataset():
    """Generate the dataset.jsonl file."""
    rows = []

    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir() or skill_dir.name.startswith("."):
            continue

        skill_name, skill_content, additional_files = read_skill_files(skill_dir)
        if not skill_content:
            print(f"Skipping {skill_dir.name}: no SKILL.md found")
            continue

        # Format the prompt
        additional_section = format_additional_files(additional_files)
        prompt = PROMPT_TEMPLATE.format(
            skill_name=skill_name,
            skill_content=skill_content,
            additional_files_section=additional_section,
        )

        # Build rubric_vars for the judge
        rubric_vars = {
            "skill_name": skill_name,
            "skill_dir": skill_dir.name,
            "skill_file_tree": get_skill_file_tree(skill_dir),
            "additional_file_names": [f[0] for f in additional_files],
        }

        rows.append(
            {
                "sample_id": f"skill-test-{skill_name}",
                "input": prompt,
                "rubric_vars": rubric_vars,
            }
        )
        print(f"Added: {skill_name}")

    # Write JSONL
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nGenerated {len(rows)} samples to {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_dataset()
