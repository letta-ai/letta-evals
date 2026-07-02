import json
from pathlib import Path
from typing import Any, Iterator, List, Optional, Union

from letta_evals.models import Sample, SampleId


def _normalize_sample_id(value) -> SampleId:
    """Return a JSON-serializable sample id while preserving explicit dataset identity."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _json_sample_id(data: dict, fallback: int) -> SampleId:
    for field_name in ("sample_id", "id"):
        if field_name in data and data[field_name] is not None:
            return _normalize_sample_id(data[field_name])
    return fallback


def _csv_sample_id(df: Any, row: Any, fallback: int) -> SampleId:
    import pandas as pd

    for field_name in ("sample_id", "id"):
        if field_name in df.columns and not pd.isna(row.get(field_name)):
            return _normalize_sample_id(row[field_name])
    return fallback


def _resolve_rubric_fields(
    rubric_inline: Optional[str],
    rubric_path: Optional[str],
    base_dir: Path,
    row_idx: int,
) -> Optional[str]:
    """Resolve per-sample rubric fields into a single rubric string.

    Returns the rubric text (loaded from disk if ``rubric_path`` is set,
    otherwise the inline ``rubric``), or ``None`` if neither is provided.
    Raises ``ValueError`` if both are set.

    Relative ``rubric_path`` values resolve against ``base_dir`` (the suite
    directory), consistent with every other path field in the suite —
    ``function``, ``extractor``, ``prompt_path``, ``setup_script``, etc. This
    is what lets a dataset live somewhere other than the suite dir (e.g. an
    HF-backed manifest in ``~/.cache/huggingface``) while its rubric files
    stay resolvable next to the suite.
    """
    if rubric_inline is not None and rubric_path is not None:
        raise ValueError(f"Row {row_idx}: cannot set both 'rubric' and 'rubric_path'. Use one or the other.")
    if rubric_path is not None:
        path = Path(rubric_path)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        if not path.exists():
            raise ValueError(f"Row {row_idx}: rubric_path '{rubric_path}' does not exist (resolved to {path}).")
        with open(path, "r") as f:
            return f.read()
    return rubric_inline


def load_jsonl(
    file_path: Path,
    max_samples: Optional[int] = None,
    sample_tags: Optional[List[str]] = None,
    base_dir: Optional[Path] = None,
) -> Iterator[Sample]:
    """Load samples from a JSONL file.

    ``base_dir`` is the suite directory, used to resolve relative
    ``rubric_path`` values. Falls back to the dataset file's own directory
    when unset (e.g. direct calls); for a colocated dataset the two coincide.
    """
    rubric_base = base_dir if base_dir is not None else file_path.parent
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
            rubric_text = _resolve_rubric_fields(
                data.get("rubric"),
                data.get("rubric_path"),
                rubric_base,
                line_index,
            )
            sample = Sample(
                id=_json_sample_id(data, line_index),
                input=data.get("input") or data["prompt"],
                ground_truth=data.get("ground_truth"),
                agent_args=data.get("agent_args") or metadata.get("agent_args"),
                rubric_vars=data.get("rubric_vars") or metadata.get("rubric_vars"),
                extra_vars=data.get("extra_vars") or metadata.get("extra_vars"),
                rubric=rubric_text,
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


def _parse_json_dict_field(df: Any, row: Any, field_name: str, row_idx) -> Optional[dict]:
    """Parse an optional CSV column that expects a JSON object string."""
    import pandas as pd

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
    file_path: Path,
    max_samples: Optional[int] = None,
    sample_tags: Optional[List[str]] = None,
    base_dir: Optional[Path] = None,
) -> Iterator[Sample]:
    """Load samples from a CSV file.

    Expected columns:
    - input (required): str or list of strings (as JSON array string)
    - ground_truth (optional): str or list of strings (as JSON array string for per-turn evaluation)
    - agent_args (optional): dict as JSON string
    - rubric_vars (optional): dict as JSON string
    - extra_vars (optional): dict as JSON string
    - rubric (optional): per-sample rubric text (multi-line strings in CSV
      cells are awkward; prefer ``rubric_path`` for non-trivial rubrics)
    - rubric_path (optional): per-sample rubric file path. Resolved relative
      to ``base_dir`` (the suite directory). Mutually exclusive with ``rubric``.
    """
    import pandas as pd

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {file_path}: {e}")

    if df.empty:
        raise ValueError(f"CSV file {file_path} is empty")

    if "input" not in df.columns:
        raise ValueError(f"CSV file {file_path} missing required column 'input'. Found columns: {list(df.columns)}")

    rubric_base = base_dir if base_dir is not None else file_path.parent
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

        rubric_inline = None
        if "rubric" in df.columns and not pd.isna(row.get("rubric")):
            rubric_inline = str(row["rubric"])
        rubric_path_value = None
        if "rubric_path" in df.columns and not pd.isna(row.get("rubric_path")):
            rubric_path_value = str(row["rubric_path"]).strip()

        rubric_text = _resolve_rubric_fields(rubric_inline, rubric_path_value, rubric_base, int(idx))

        # create sample
        try:
            sample = Sample(
                id=_csv_sample_id(df, row, int(idx)),
                input=input_value,
                ground_truth=ground_truth,
                agent_args=agent_args,
                rubric_vars=rubric_vars,
                extra_vars=extra_vars,
                rubric=rubric_text,
            )
        except Exception as e:
            raise ValueError(f"Row {idx}: Failed to create Sample: {e}")

        yielded_count += 1
        yield sample


def load_dataset(
    file_path: Union[str, Path],
    max_samples: Optional[int] = None,
    sample_tags: Optional[List[str]] = None,
    base_dir: Optional[Path] = None,
) -> Iterator[Sample]:
    """Load samples from a dataset file (JSONL or CSV).

    Automatically detects format based on file extension:
    - .jsonl: Load as JSONL
    - .csv: Load as CSV

    Args:
        file_path: Path to dataset file (.jsonl or .csv)
        max_samples: Maximum number of samples to load
        sample_tags: Filter samples by tags (not currently supported)
        base_dir: Suite directory, used to resolve relative ``rubric_path``
            values. Defaults to the dataset file's own directory when unset.

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
        return load_jsonl(file_path, max_samples=max_samples, sample_tags=sample_tags, base_dir=base_dir)
    elif suffix == ".csv":
        return load_csv(file_path, max_samples=max_samples, sample_tags=sample_tags, base_dir=base_dir)
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}. Supported formats: .jsonl, .csv")
