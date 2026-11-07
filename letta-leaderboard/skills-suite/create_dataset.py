import argparse
import csv
import json
from pathlib import Path

from prompts import SKILLS_TASK_PROMPT


def create_dataset(output_dir: str = "output", dataset_name: str = "dataset.csv", oracle: bool = True):
    """
    Convert all task JSON files from task_generator/output/ to a CSV dataset.

    Args:
        output_dir: Directory name for sandbox files (default: "output")
        dataset_name: Name of the output CSV file (default: "dataset.csv")
        oracle: Use oracle setting (default: True)

    CSV columns:
    - sample_id: The sample ID
    - skill: The name of the skill
    - input: The task description
    - rubric_vars: JSON string containing rubric_task_completion, rubric_skill_use, skill, and files
    """
    # Define paths
    input_dir = Path(__file__).parent / "task_generator" / "output"

    # Collect all task data
    tasks = []

    if oracle:
        dataset_name = dataset_name.replace(".csv", "_oracle.csv")
    else:
        dataset_name = dataset_name.replace(".csv", "_selection.csv")
    output_file = Path(__file__).parent / "data" / dataset_name

    # Process all JSON files in the output directory
    for idx, json_file in enumerate(sorted(input_dir.glob("*.json"))):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            skill_name = data.get("skill_name").replace("/", "-")
            sample_id = f"sample-{idx}"

            if oracle:
                task_name = f"{sample_id}\n\nSkill to use: {skill_name}"
            else:
                task_name = sample_id

            task_input = SKILLS_TASK_PROMPT.replace("{{task}}", data.get("task", "")).replace(
                "{{task_name}}", task_name
            )

            # Extract rubric variables
            rubric_vars = {
                "rubric_task_completion": data.get("rubric_task_completion", ""),
                "rubric_skill_use": data.get("rubric_skill_use", ""),
                "skill": skill_name,
                "files": data.get("files", []),
            }

            # Create task entry
            task_entry = {
                "sample_id": sample_id,
                "skill": skill_name,
                "input": task_input,
                "rubric_vars": json.dumps(rubric_vars),
            }

            tasks.append(task_entry)

        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue

    # Write to CSV
    if tasks:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "skill", "input", "rubric_vars"])
            writer.writeheader()
            writer.writerows(tasks)

        print(f"Dataset created successfully: {output_file}")
        print(f"Total tasks: {len(tasks)}")
    else:
        print("No tasks found to process")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert task JSON files to CSV dataset")
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory for sandbox files (default: output)"
    )
    parser.add_argument(
        "--dataset-name", type=str, default="dataset.csv", help="Name of the output CSV file (default: dataset.csv)"
    )
    parser.add_argument("--oracle", action="store_false", help="Use oracle setting")

    args = parser.parse_args()
    create_dataset(output_dir=args.output_dir, dataset_name=args.dataset_name, oracle=args.oracle)
