#!/usr/bin/env python3
"""Parse results.jsonl files into a readable directory structure."""

import json
import re
import sys
from pathlib import Path


def slugify(text: str, max_len: int = 50) -> str:
    """Convert text to a filesystem-safe slug."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = text.strip("-")
    return text[:max_len]


def extract_skill_name(input_text: str) -> str | None:
    """Extract skill name from input prompt."""
    match = re.search(r"\*\*Skill Name:\*\*\s*([^\n*]+)", input_text)
    if match:
        return match.group(1).strip()
    return None


def extract_model_short_name(model_name: str) -> str:
    """Extract short model name from full model string."""
    # "anthropic/claude-haiku-4-5-20251001" -> "claude-haiku-4-5"
    name = model_name.split("/")[-1]  # drop provider prefix
    # Remove date suffix (8 digits at end)
    name = re.sub(r"-\d{8}$", "", name)
    return name


def process_result(result: dict, output_dir: Path, idx: int) -> Path:
    """Process a single result and write to output directory."""
    sample = result.get("sample", {})
    grade = result.get("grade", {})
    trajectory = result.get("trajectory", [])
    submission = result.get("submission", "")

    # Create directory: model_name/test_case
    model_name = result.get("model_name", "unknown")
    model_short = extract_model_short_name(model_name)
    input_text = sample.get("input", "")
    skill_name = extract_skill_name(input_text) or f"result-{idx:03d}"
    test_case_name = f"{idx:03d}-{slugify(skill_name)}"

    result_dir = output_dir / model_short / test_case_name
    result_dir.mkdir(parents=True, exist_ok=True)

    # 1. meta.json - summary info
    meta = {
        "skill_name": skill_name,
        "model": result.get("model_name", "unknown"),
        "score": grade.get("score"),
        "cost": result.get("cost", 0),
        "rationale": grade.get("rationale", ""),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
    }
    (result_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # 2. input.md - the task prompt
    (result_dir / "input.md").write_text(input_text)

    # 3. submission.txt - agent's final output
    (result_dir / "submission.txt").write_text(submission or "")

    # 4. trajectory.jsonl - one line per message
    with open(result_dir / "trajectory.jsonl", "w") as f:
        for turn_idx, turn in enumerate(trajectory):
            messages = turn if isinstance(turn, list) else [turn]
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                record = {
                    "turn": turn_idx,
                    "type": msg.get("message_type", "unknown"),
                    "content": msg.get("content", ""),
                }
                f.write(json.dumps(record) + "\n")

    return result_dir


def main():
    # Find all results.jsonl files
    base_dir = Path(__file__).parent
    results_dir = base_dir / "results"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    jsonl_files = list(results_dir.glob("**/results.jsonl"))
    if not jsonl_files:
        print(f"No results.jsonl files found in {results_dir}")
        sys.exit(1)

    output_base = base_dir / "parsed"
    # Clean existing output
    if output_base.exists():
        import shutil
        shutil.rmtree(output_base)
    output_base.mkdir(exist_ok=True)

    total_processed = 0

    for jsonl_file in jsonl_files:
        print(f"Processing: {jsonl_file}")

        # Create output dir mirroring the input structure
        rel_path = jsonl_file.parent.relative_to(results_dir)
        output_dir = output_base / rel_path
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(jsonl_file) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                result = data.get("result", data)

                result_dir = process_result(result, output_dir, idx)
                print(f"  -> {result_dir.name}")
                total_processed += 1

    print(f"\nProcessed {total_processed} results to {output_base}")


if __name__ == "__main__":
    main()
