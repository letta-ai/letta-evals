import json
from pathlib import Path
from typing import Iterator, List, Optional, Union

import pandas as pd

from letta_evals.models import Sample


def load_jsonl(
    file_path: Path, max_samples: Optional[int] = None, sample_tags: Optional[List[str]] = None
) -> Iterator[Sample]:
    """Load samples from a JSONL file."""
    with open(file_path, "r") as f:
        line_index = 0
        yielded_count = 0
        for line in f:
            if max_samples and yielded_count >= max_samples:
                break

            data = json.loads(line.strip())

            # skip filtering by tags since metadata is removed
            if sample_tags:
                # tags filtering no longer supported without metadata
                pass

            sample = Sample(
                id=line_index,
                input=data["input"],
                ground_truth=data.get("ground_truth"),
                agent_args=data.get("agent_args"),
                rubric_vars=data.get("rubric_vars"),
                extra_vars=data.get("extra_vars"),
            )

            line_index += 1
            yielded_count += 1
            yield sample


def load_csv(
    file_path: Path, max_samples: Optional[int] = None, sample_tags: Optional[List[str]] = None
) -> Iterator[Sample]:
    """Load samples from a CSV file.

    Expected columns:
    - input (required): str or list of strings (as JSON array string)
    - ground_truth (optional): str
    - agent_args (optional): dict as JSON string
    - rubric_vars (optional): dict as JSON string
    - extra_vars (optional): dict as JSON string
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {file_path}: {e}")

    if df.empty:
        raise ValueError(f"CSV file {file_path} is empty")

    if "input" not in df.columns:
        raise ValueError(f"CSV file {file_path} missing required column 'input'. Found columns: {list(df.columns)}")

    yielded_count = 0
    for idx, row in df.iterrows():
        if max_samples and yielded_count >= max_samples:
            break

        # parse input field
        try:
            input_value = row["input"]
            if pd.isna(input_value):
                raise ValueError(f"Row {idx}: 'input' column cannot be null")

            # check if input is a JSON array (list of strings)
            input_str = str(input_value).strip()
            if input_str.startswith("[") and input_str.endswith("]"):
                try:
                    parsed_input = json.loads(input_str)
                    if not isinstance(parsed_input, list):
                        raise ValueError(f"Row {idx}: 'input' array must be a list")
                    input_value = parsed_input
                except json.JSONDecodeError as e:
                    raise ValueError(f"Row {idx}: 'input' appears to be JSON array but is invalid: {e}")
            else:
                input_value = str(input_value)
        except Exception as e:
            raise ValueError(f"Row {idx}: Failed to parse 'input' field: {e}")

        # parse ground_truth field
        ground_truth = None
        if "ground_truth" in df.columns and not pd.isna(row.get("ground_truth")):
            ground_truth = str(row["ground_truth"])

        # parse agent_args field (expects JSON string)
        agent_args = None
        if "agent_args" in df.columns and not pd.isna(row.get("agent_args")):
            try:
                agent_args_str = str(row["agent_args"]).strip()
                agent_args = json.loads(agent_args_str)
                if not isinstance(agent_args, dict):
                    raise ValueError(f"Row {idx}: 'agent_args' must be a JSON object/dict, got {type(agent_args)}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Row {idx}: 'agent_args' column contains invalid JSON: {e}")

        # parse rubric_vars field (expects JSON string)
        rubric_vars = None
        if "rubric_vars" in df.columns and not pd.isna(row.get("rubric_vars")):
            try:
                rubric_vars_str = str(row["rubric_vars"]).strip()
                rubric_vars = json.loads(rubric_vars_str)
                if not isinstance(rubric_vars, dict):
                    raise ValueError(f"Row {idx}: 'rubric_vars' must be a JSON object/dict, got {type(rubric_vars)}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Row {idx}: 'rubric_vars' column contains invalid JSON: {e}")

        # parse extra_vars field (expects JSON string)
        extra_vars = None
        if "extra_vars" in df.columns and not pd.isna(row.get("extra_vars")):
            try:
                extra_vars_str = str(row["extra_vars"]).strip()
                extra_vars = json.loads(extra_vars_str)
                if not isinstance(extra_vars, dict):
                    raise ValueError(f"Row {idx}: 'extra_vars' must be a JSON object/dict, got {type(extra_vars)}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Row {idx}: 'extra_vars' column contains invalid JSON: {e}")

        # create sample
        try:
            sample = Sample(
                id=int(idx),
                input=input_value,
                ground_truth=ground_truth,
                agent_args=agent_args,
                rubric_vars=rubric_vars,
                extra_vars=extra_vars,
            )
        except Exception as e:
            raise ValueError(f"Row {idx}: Failed to create Sample: {e}")

        yielded_count += 1
        yield sample


def load_dataset(
    file_path: Union[str, Path], max_samples: Optional[int] = None, sample_tags: Optional[List[str]] = None
) -> Iterator[Sample]:
    """Load samples from a dataset file (JSONL or CSV).

    Automatically detects format based on file extension:
    - .jsonl: Load as JSONL
    - .csv: Load as CSV

    Args:
        file_path: Path to dataset file (.jsonl or .csv)
        max_samples: Maximum number of samples to load
        sample_tags: Filter samples by tags (not currently supported)

    Returns:
        Iterator of Sample objects

    Raises:
        ValueError: If file format is unsupported or file is invalid
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise ValueError(f"Dataset file does not exist: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".jsonl":
        return load_jsonl(file_path, max_samples=max_samples, sample_tags=sample_tags)
    elif suffix == ".csv":
        return load_csv(file_path, max_samples=max_samples, sample_tags=sample_tags)
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}. Supported formats: .jsonl, .csv")
