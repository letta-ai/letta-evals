#!/usr/bin/env python3
"""Generate dataset_all.jsonl for skill-test-writing evaluation.

For each skill in reference_skills/, creates a row asking the agent to write
a test case for that skill. Uses extra_vars to pass skill metadata to the judge.
"""

import json
from pathlib import Path

SKILLS_DIR = Path(__file__).parent / "reference_skills"
OUTPUT_FILE = Path(__file__).parent / "data" / "dataset_all.jsonl"

# Prompt template for generating test cases
PROMPT_TEMPLATE = """You are an evaluation designer. You have a skill document that teaches an LLM domain-specific knowledge. Your job is to write ONE test case that distinguishes a knowledgeable practitioner from a novice in this domain.

## The Skill

**Skill Name:** {skill_name}

**SKILL.md Content:**
```markdown
{skill_content}
```

{additional_files_section}

## What Makes a Great Test Case

A great test case captures a **moment where domain expertise changes the outcome** -- a decision point, a subtle pitfall, or a design tradeoff that practitioners learn from experience. The test should feel like a real question someone would ask a senior colleague, not a quiz about documentation.

### The Transferability Principle

Ask yourself: "Would a domain expert who has NEVER read this specific skill document still get this right?" If yes, you are testing genuine domain knowledge. If no, you are testing document memorization.

**Transferable knowledge** (test this):
- Architectural patterns and tradeoffs (e.g., "formulas vs hardcoded values in spreadsheets")
- Common pitfalls and debugging patterns (e.g., "race condition when server isn't ready")
- When to use which tool/approach for a given problem (e.g., "library X for tables, library Y for merging")
- Destructive operations and their consequences (e.g., "opening with data_only=True then saving destroys formulas")

**Skill-document trivia** (do NOT test this):
- Exact script names, file paths, or CLI flags specific to this skill's tooling
- Magic numbers or specific configuration values mentioned in the skill
- The exact workflow steps described in the skill document
- Which fields are required in the skill's configuration format

### Prompt Style: Scenario-Based

Write the prompt as a realistic situation -- a practitioner encountering a problem, making a decision, or asking for guidance. Frame it as something someone would type into a chat with a knowledgeable colleague.

**Strong prompt patterns:**
- "I'm doing X and getting error Y. What's going wrong?" (debugging scenario)
- "I need to accomplish X. Should I use approach A or approach B?" (design decision)
- "I'm about to do X. Are there any risks I should know about?" (pitfall awareness)
- "My X works in development but fails in production. Why?" (environment difference)

**Weak prompt patterns to avoid:**
- "What is the correct workflow for X?" (asks to recite steps from docs)
- "What fields/parameters are required for X?" (asks to list facts)
- "How often should I do X?" (asks for a magic number)
- "What does tool X do?" (asks for a description, not application)

## Grader Design: Grade Concepts, Not Strings

The grader evaluates whether the response demonstrates **understanding of the underlying concept**, not whether it mentions specific terms.

### Grader Anti-Patterns (avoid these):
- Requiring a specific library name when alternatives exist (e.g., only accepting "openpyxl" when "xlsxwriter" also works)
- Requiring a specific numeric value (e.g., "must say 30 minutes") when the concept matters more
- Checking for exact phrases or keywords rather than conceptual understanding
- Requiring mention of skill-specific script names or tools
- Only accepting one valid approach when multiple exist

### Grader Best Practices:
- Grade on whether the response identifies the **core concept or pitfall**
- Accept multiple valid solutions to the same problem
- Use graduated scoring (0, 0.5, 1.0) based on depth of understanding
- Focus the 1.0 criteria on explaining the "why" not just the "what"

## Output Format

Output your test.yaml content directly in a YAML code block:

```yaml
name: descriptive-kebab-case-name
skills:
  - {skill_name}
prompt: |
  A scenario-based prompt that a practitioner would realistically encounter.
  Should test transferable domain knowledge, not skill-document trivia.
timeout: 120
grader:
  kind: letta_judge
  prompt: |
    Evaluate whether the response demonstrates understanding of [core concept].

    The key insight is: [what a knowledgeable practitioner would know]

    Score 1.0 if: Explains [concept] AND why it matters. Accepts any valid approach.
    Score 0.5 if: Identifies the issue but explanation is incomplete or misses the "why".
    Score 0.0 if: Does not address [concept] or gives incorrect guidance.
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
    """Generate the dataset_all.jsonl file."""
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
