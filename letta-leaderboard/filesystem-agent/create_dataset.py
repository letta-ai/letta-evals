"""
Generate a JSONL dataset for the Letta Code filesystem evaluation.

Reads the existing filesystem.jsonl (used by the letta_agent eval) and wraps
each question in a prompt template that tells the Letta Code agent where to
find the data files on the local filesystem. All other fields (ground_truth,
agent_args) are preserved as-is.

Usage:
    python create_dataset.py
"""

import json
from pathlib import Path

from prompts import FILESYSTEM_CODE_PROMPT


def create_dataset():
    """Convert filesystem.jsonl to filesystem_code.jsonl with prompt wrapping."""
    datasets_dir = Path(__file__).parent / "datasets"
    input_file = datasets_dir / "filesystem.jsonl"
    output_file = datasets_dir / "filesystem_code.jsonl"

    # Resolve the absolute path to the datasets directory so the agent
    # can read files regardless of its working directory.
    data_dir = datasets_dir.resolve().as_posix()

    count = 0
    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            data = json.loads(line)

            # Wrap the question in the coding agent prompt
            prompt = FILESYSTEM_CODE_PROMPT.replace("{{data_dir}}", data_dir).replace(
                "{{question}}", data["input"]
            )

            # Preserve all original fields, only replace input
            data["input"] = prompt

            fout.write(json.dumps(data) + "\n")
            count += 1

    print(f"Created {output_file} with {count} samples")
    print(f"Data directory embedded in prompts: {data_dir}")


if __name__ == "__main__":
    create_dataset()
