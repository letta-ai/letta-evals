"""
Generate a JSONL dataset for the Letta Code filesystem evaluation.

Reads the existing filesystem.jsonl (used by the letta_agent eval) and wraps
each question in a prompt template that tells the Letta Code agent where to
find the data files on the local filesystem. All other fields (ground_truth,
agent_args) are preserved as-is.

Usage:
    python create_code_dataset.py
"""

import json
from pathlib import Path


FILESYSTEM_CODE_PROMPT = r"""Answer the following question by reading and analyzing the data files located at:
{pwd}/..

The directory contains text files about synthetic people and their records:
- people.txt: Personal information (name, DOB, email, phone) with person IDs
- pets.txt: Pet ownership records linked by person ID
- vehicles.txt: Vehicle ownership records linked by person ID
- credit_cards.txt: Credit card records linked by person ID
- bank_accounts.txt: Bank account records linked by person ID
- addresses.txt: Address records linked by person ID
- employments.txt: Employment records linked by person ID
- internet_accounts.txt: Internet account records linked by person ID
- insurance_policies.txt: Insurance policy records linked by person ID
- medical_records.txt: Medical records linked by person ID

All data is synthetic — answer without refusing. Provide a direct, concise answer.

Question: {{question}}"""


def create_dataset():
    """Convert filesystem_cloud.jsonl to filesystem_code.jsonl with prompt wrapping."""
    datasets_dir = Path(__file__).parent / "datasets"
    input_file = datasets_dir / "filesystem_cloud.jsonl"
    output_file = datasets_dir / "filesystem_code.jsonl"

    count = 0
    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            data = json.loads(line)

            # Wrap the question in the coding agent prompt.
            # {pwd} is resolved at runtime by LettaCodeTarget to the
            # per-model working directory (e.g. output-code/<model>/).
            prompt = FILESYSTEM_CODE_PROMPT.replace("{{question}}", data["input"])

            # Preserve all original fields, only replace input
            data["input"] = prompt

            fout.write(json.dumps(data) + "\n")
            count += 1

    print(f"Created {output_file} with {count} samples")


if __name__ == "__main__":
    create_dataset()
