#!/usr/bin/env python3
"""Build per-category scenario data files for the sidecar viewer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KNOWN_KEYS = {
    "sample_id",
    "input",
    "ground_truth",
    "agent_args",
    "rubric_vars",
    "extra_vars",
}


def display_name_from_slug(slug: str) -> str:
    return slug.replace("-", " ").strip().title()


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def normalize_turn(turn: Any, default_role: str) -> dict[str, str]:
    if isinstance(turn, dict):
        role = (
            turn.get("role")
            or turn.get("speaker")
            or turn.get("type")
            or default_role
        )
        content = (
            turn.get("content")
            or turn.get("text")
            or turn.get("message")
            or turn.get("value")
        )
        if content is None:
            content = turn
        return {
            "role": stringify(role).strip() or default_role,
            "content": stringify(content),
        }
    return {"role": default_role, "content": stringify(turn)}


def normalize_turns(value: Any, default_role: str) -> list[dict[str, str]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_turn(item, default_role) for item in value]
    return [normalize_turn(value, default_role)]


def turns_to_text(turns: list[dict[str, str]]) -> str:
    chunks = []
    for turn in turns:
        role = turn.get("role", "").strip()
        content = turn.get("content", "").strip()
        if role:
            chunks.append(f"{role}: {content}")
        else:
            chunks.append(content)
    return "\n\n".join(chunks)


def build_scenario(
    record: dict[str, Any],
    dataset_name: str,
    source_path: str,
    line_no: int,
) -> dict[str, Any]:
    input_turns = normalize_turns(record.get("input"), "input")
    ground_turns = normalize_turns(record.get("ground_truth"), "expected")

    scenario_id = record.get("sample_id") or f"{dataset_name}:{line_no:04d}"

    meta: dict[str, Any] = {}
    for key in ("agent_args", "rubric_vars", "extra_vars"):
        if key in record:
            meta[key] = record[key]

    extra = {k: v for k, v in record.items() if k not in KNOWN_KEYS}
    if extra:
        meta["extra"] = extra

    scenario = {
        "id": scenario_id,
        "dataset": dataset_name,
        "source": source_path,
        "line": line_no,
        "input_turns": input_turns,
        "ground_truth_turns": ground_turns,
        "input_text": turns_to_text(input_turns),
        "ground_truth_text": turns_to_text(ground_turns),
        "meta": meta,
    }

    return scenario


def find_latest_run(results_dir: Path) -> Path | None:
    """Find the latest results run directory by modification time."""
    run_dirs = [
        d
        for d in results_dir.iterdir()
        if d.is_dir() and (d / "results.jsonl").exists()
    ]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda d: (d / "results.jsonl").stat().st_mtime)


def build_line_index(data_dir: Path) -> dict[int, str]:
    """Build a mapping from 0-based sample index to sample_id across all JSONL files.

    The eval framework assigns 0-based numeric IDs to samples in order.
    This must match the order samples appear in build_scenario calls.
    """
    index: dict[int, str] = {}
    idx = 0
    for jsonl_path in sorted(data_dir.glob("*.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    # Skip malformed lines - they won't produce scenarios either
                    continue
                sample_id = record.get("sample_id", f"sample:{idx:04d}")
                index[idx] = sample_id
                idx += 1
    return index


def ingest_results(
    category_dir: Path,
    scenarios: list[dict[str, Any]],
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Ingest results from the latest run and attach to scenarios.

    Returns (models, runs) lists.
    """
    results_dir = category_dir / "results"
    if not results_dir.exists():
        return [], []

    run_dir = find_latest_run(results_dir)
    if not run_dir:
        return [], []

    runs = [
        {
            "name": run_dir.name,
            "path": str(run_dir.relative_to(root)),
        }
    ]

    # Build line-index-to-sample_id mapping from data
    data_dir = category_dir / "data"
    line_index = build_line_index(data_dir) if data_dir.exists() else {}

    # Build scenario lookup by id
    scenario_map: dict[str, dict[str, Any]] = {}
    for s in scenarios:
        scenario_map[s["id"]] = s
        s.setdefault("results", [])

    # Read results.jsonl
    results_path = run_dir / "results.jsonl"
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            result = entry.get("result", {})
            sample = result.get("sample", {})
            grade = result.get("grade", {})
            model_name = result.get("model_name", "unknown")
            submission = result.get("submission", "")
            score = grade.get("score", 0)
            rationale = grade.get("rationale", "")

            # Map numeric sample id to scenario sample_id
            sample_id_num = sample.get("id")
            scenario_id = line_index.get(sample_id_num)
            if scenario_id and scenario_id in scenario_map:
                scenario_map[scenario_id]["results"].append(
                    {
                        "model_name": model_name,
                        "submission": submission,
                        "score": score,
                        "rationale": rationale,
                    }
                )

    # Sort each scenario's results by score descending
    for s in scenarios:
        if s.get("results"):
            s["results"].sort(key=lambda r: r["score"], reverse=True)

    # Read summary.json for model-level aggregates
    models: list[dict[str, Any]] = []
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            per_model = summary.get("metrics", {}).get("per_model", [])
            for m in per_model:
                models.append(
                    {
                        "model_name": m["model_name"],
                        "avg_score": round(m.get("avg_score_attempted", 0), 4),
                        "total": m.get("total", 0),
                        "attempted": m.get("total_attempted", 0),
                    }
                )
            models.sort(key=lambda m: m["avg_score"], reverse=True)
        except (json.JSONDecodeError, KeyError):
            pass

    return models, runs


def load_rubric(category_dir: Path, root: Path) -> dict[str, str] | None:
    """Load the rubric prompt text for a category, if present."""
    patterns = ("rubric_*.txt", "judge_prompt.md")
    for pattern in patterns:
        matches = sorted(category_dir.glob(pattern))
        if matches:
            path = matches[0]
            return {
                "path": str(path.relative_to(root)),
                "content": path.read_text(encoding="utf-8"),
            }
    return None


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    viewer_dir = root / "viewer"
    out_dir = viewer_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    categories = []

    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir():
            continue
        if category_dir.name == "viewer":
            continue
        if not (category_dir / "README.md").exists():
            continue

        data_dir = category_dir / "data"
        if not data_dir.exists():
            continue

        jsonl_files = sorted(data_dir.glob("*.jsonl"))
        if not jsonl_files:
            continue

        scenarios: list[dict[str, Any]] = []
        datasets: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for jsonl_path in jsonl_files:
            dataset_name = jsonl_path.stem
            count = 0
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        record = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        errors.append(
                            {
                                "file": str(jsonl_path.relative_to(root)),
                                "line": line_no,
                                "error": str(exc),
                            }
                        )
                        continue

                    count += 1
                    scenarios.append(
                        build_scenario(
                            record,
                            dataset_name,
                            str(jsonl_path.relative_to(root)),
                            line_no,
                        )
                    )

            datasets.append(
                {
                    "name": dataset_name,
                    "source": str(jsonl_path.relative_to(root)),
                    "count": count,
                }
            )

        # Ingest results from latest run
        models, runs = ingest_results(category_dir, scenarios, root)
        rubric = load_rubric(category_dir, root)

        category_payload = {
            "category": category_dir.name,
            "display_name": display_name_from_slug(category_dir.name),
            "generated_at": timestamp,
            "datasets": datasets,
            "scenarios": scenarios,
            "models": models,
            "runs": runs,
            "errors": errors,
            "rubric": rubric,
        }

        out_path = out_dir / f"{category_dir.name}.json"
        out_path.write_text(
            json.dumps(category_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        categories.append(
            {
                "name": category_dir.name,
                "display_name": display_name_from_slug(category_dir.name),
                "data_file": f"data/{category_dir.name}.json",
                "scenario_count": len(scenarios),
                "dataset_count": len(datasets),
            }
        )

    index_payload = {
        "generated_at": timestamp,
        "categories": categories,
    }

    (out_dir / "index.json").write_text(
        json.dumps(index_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(categories)} category files to {out_dir}")


if __name__ == "__main__":
    main()
