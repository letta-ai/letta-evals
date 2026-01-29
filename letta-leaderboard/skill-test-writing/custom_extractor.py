"""Custom extractors for skill-test-writing evaluation."""

import json
import os
import re
from pathlib import Path

import yaml

from letta_evals.decorators import extractor
from letta_evals.extractors.utils import (
    flatten_content,
    get_assistant_messages,
)
from letta_evals.models import LettaMessageUnion


def extract_json_from_message(message: str) -> dict:
    """Extract the JSON from the message that contains 'sandbox_path' key."""
    candidates = []

    # Find all JSON within code blocks
    for match in re.finditer(r"```(?:json)?\s*\n?(.*?)```", message, re.DOTALL):
        candidates.append(match.group(1).strip())

    # Find all JSON objects in the message (look for { ... })
    for match in re.finditer(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", message, re.DOTALL):
        candidates.append(match.group(0).strip())

    # If no candidates found, try the whole message
    if not candidates:
        candidates.append(message.strip())

    jsons_with_sandbox_path = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "sandbox_path" in parsed:
                jsons_with_sandbox_path.append(parsed)
                # Prefer one with 'sandbox_path' as first key
                if parsed and next(iter(parsed.keys())) == "sandbox_path":
                    return parsed
        except json.JSONDecodeError:
            continue

    if jsons_with_sandbox_path:
        return jsons_with_sandbox_path[0]

    raise ValueError(f"No JSON with 'sandbox_path' key found in message: {message}")


def json_extractor(trajectory: list[list[LettaMessageUnion]]) -> dict:
    """Extract the sandbox path from the last assistant message content."""
    messages = get_assistant_messages(trajectory)
    if not messages:
        raise ValueError("No assistant messages found")
    last_message_text = flatten_content(messages[-1].content).strip()
    return extract_json_from_message(last_message_text)


def find_test_files(sandbox_dir: Path) -> list[tuple[Path, str]]:
    """Find all test.yaml and grader.py files in the sandbox directory."""
    test_files = []

    for root, _dirs, files in os.walk(sandbox_dir):
        root_path = Path(root)

        for file in files:
            if file in ("test.yaml", "grader.py"):
                file_path = root_path / file
                relative_path = file_path.relative_to(sandbox_dir)
                try:
                    content = file_path.read_text(encoding="utf-8")
                    test_files.append((relative_path, content))
                except Exception as e:
                    test_files.append((relative_path, f"[Error reading file: {e}]"))

    return test_files


def load_test_context(test_files: list[tuple[Path, str]]) -> dict:
    """Load context from test files for the judge prompt."""
    test_config = None
    grader_code = None

    for rel_path, content in test_files:
        if rel_path.name == "test.yaml":
            try:
                test_config = yaml.safe_load(content)
            except yaml.YAMLError:
                test_config = {"error": "Failed to parse test.yaml", "raw": content}
        elif rel_path.name == "grader.py":
            grader_code = content

    return {
        "test_config": test_config,
        "grader_code": grader_code,
    }


def format_for_judge(context: dict) -> dict:
    """Format the test context for the judge prompt template variables."""
    # Format prompt
    prompt = ""
    if context["test_config"]:
        prompt = context["test_config"].get("prompt", "[No prompt found in test.yaml]")

    # Format grader config
    grader_config = ""
    if context["test_config"] and "grader" in context["test_config"]:
        grader_config = yaml.dump(context["test_config"]["grader"])

    # Format grader code section
    grader_code_section = ""
    if context["grader_code"]:
        code = context["grader_code"]
        if len(code) > 3000:
            code = code[:3000] + "\n... (truncated)"
        grader_code_section = f"""
**Grader Code (grader.py):**
```python
{code}
```
"""

    return {
        "prompt": prompt,
        "grader_config": grader_config,
        "grader_code_section": grader_code_section,
    }


@extractor
def test_files_extractor(trajectory: list[list[LettaMessageUnion]], config: dict) -> str:
    """Extract test files from the sandbox and format for judge evaluation.

    Returns a formatted string with the test case content.
    """
    sandbox_path = json_extractor(trajectory)["sandbox_path"]

    if not sandbox_path:
        raise ValueError("No sandbox path found")

    if not os.path.exists(sandbox_path):
        return "[No sandbox directory found. No test files were created.]"

    sandbox_dir = Path(sandbox_path)
    if not sandbox_dir.is_dir():
        return f"[Sandbox path is not a directory: {sandbox_path}]"

    # Find all test files
    test_files = find_test_files(sandbox_dir)

    if not test_files:
        return "[No test.yaml or grader.py files found in sandbox.]"

    # Load and format context
    context = load_test_context(test_files)

    if not context["test_config"]:
        return "[No valid test.yaml found in sandbox.]"

    formatted = format_for_judge(context)

    # Build formatted output
    parts = []
    parts.append(f"**Prompt (what the model is asked):**\n```\n{formatted['prompt']}\n```")
    parts.append(f"**Grader Configuration:**\n```yaml\n{formatted['grader_config']}```")
    if formatted["grader_code_section"]:
        parts.append(formatted["grader_code_section"])

    return "\n\n".join(parts)
