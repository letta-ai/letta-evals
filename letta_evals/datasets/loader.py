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

            metadata = data.get("metadata") or {}
            sample = Sample(
                id=line_index,
                input=data.get("input") or data["prompt"],
                ground_truth=data.get("ground_truth"),
                agent_args=data.get("agent_args") or metadata.get("agent_args"),
                rubric_vars=data.get("rubric_vars") or metadata.get("rubric_vars"),
                extra_vars=data.get("extra_vars") or metadata.get("extra_vars"),
            )

            line_index += 1
            yielded_count += 1
            yield sample


def _parse_string_or_list(value: str, field_name: str, row_idx: int) -> Union[str, List[str]]:
    """Parse a CSV field that can be either a string or a JSON array of strings.

    Args:
        value: The string value from the CSV cell
        field_name: Name of the field (for error messages)
        row_idx: Row index (for error messages)

    Returns:
        Either the original string or a parsed list of strings
    """
    value_str = value.strip()
    if value_str.startswith("[") and value_str.endswith("]"):
        try:
            parsed = json.loads(value_str)
            if not isinstance(parsed, list):
                raise ValueError(f"Row {row_idx}: '{field_name}' array must be a list")
            return parsed
        except json.JSONDecodeError as e:
            raise ValueError(f"Row {row_idx}: '{field_name}' appears to be JSON array but is invalid: {e}")
    return value_str


def _parse_json_dict_field(df: pd.DataFrame, row, field_name: str, row_idx) -> Optional[dict]:
    """Parse an optional CSV column that expects a JSON object string."""
    if field_name not in df.columns or pd.isna(row.get(field_name)):
        return None
    try:
        value = json.loads(str(row[field_name]).strip())
        if not isinstance(value, dict):
            raise ValueError(f"Row {row_idx}: '{field_name}' must be a JSON object/dict, got {type(value)}")
        return value
    except json.JSONDecodeError as e:
        raise ValueError(f"Row {row_idx}: '{field_name}' column contains invalid JSON: {e}")


def load_csv(
    file_path: Path, max_samples: Optional[int] = None, sample_tags: Optional[List[str]] = None
) -> Iterator[Sample]:
    """Load samples from a CSV file.

    Expected columns:
    - input (required): str or list of strings (as JSON array string)
    - ground_truth (optional): str or list of strings (as JSON array string for per-turn evaluation)
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
        input_value = row["input"]
        if pd.isna(input_value):
            raise ValueError(f"Row {idx}: 'input' column cannot be null")
        input_value = _parse_string_or_list(str(input_value), "input", idx)

        # parse ground_truth field
        ground_truth = None
        if "ground_truth" in df.columns and not pd.isna(row.get("ground_truth")):
            ground_truth = _parse_string_or_list(str(row["ground_truth"]), "ground_truth", idx)

        agent_args = _parse_json_dict_field(df, row, "agent_args", idx)
        rubric_vars = _parse_json_dict_field(df, row, "rubric_vars", idx)
        extra_vars = _parse_json_dict_field(df, row, "extra_vars", idx)

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
