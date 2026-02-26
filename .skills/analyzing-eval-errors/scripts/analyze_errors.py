#!/usr/bin/env python3
"""
Analyze errors in a letta_evals results directory.

Parses results.jsonl, classifies errors, and optionally cross-references
with the Letta server to check actual agent state.

Usage:
    # Classify errors from JSONL only (no API calls)
    python analyze_errors.py --results-dir path/to/results

    # Full analysis with server-side checks
    python analyze_errors.py --results-dir path/to/results --check-server

    # Write output to a specific file
    python analyze_errors.py --results-dir path/to/results --check-server --output report.json
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


def classify_error(error: dict) -> str:
    """Classify an error into a category based on its message and type."""
    msg = error.get("message", "")
    exc = error.get("exception_type", "")
    cat = error.get("category", "")

    if "timed out" in msg:
        return "timeout"
    if "ExtractionError" in exc or cat == "extraction":
        return "extraction"
    if cat == "grading":
        return "grading"
    if "return code" in msg:
        return "cli_crash"
    return "other"


def parse_results(results_dir: Path) -> list[dict]:
    """Parse results.jsonl and extract errored samples."""
    results_file = results_dir / "results.jsonl"
    if not results_file.exists():
        print(f"Error: {results_file} not found", file=sys.stderr)
        sys.exit(1)

    errors = []
    total = 0

    with open(results_file) as f:
        for line in f:
            rec = json.loads(line)
            result = rec.get("result", rec)
            total += 1

            error = result.get("error")
            if not error:
                continue

            entry = {
                "sample_id": result["sample"]["id"],
                "agent_id": result.get("agent_id"),
                "model_name": result.get("model_name"),
                "error_category": error.get("category"),
                "exception_type": error.get("exception_type"),
                "error_message": error.get("message", "")[:500],
                "classification": classify_error(error),
            }
            errors.append(entry)

    return errors


def check_server(errors: list[dict]) -> list[dict]:
    """Cross-reference errors with Letta server state."""
    try:
        from letta_client import Letta
    except ImportError:
        print("Error: letta_client not installed. Run: pip install letta-client", file=sys.stderr)
        sys.exit(1)

    client = Letta(api_key=os.environ.get("LETTA_API_KEY"))
    enriched = []

    for entry in errors:
        aid = entry.get("agent_id")
        if not aid or entry["classification"] == "timeout":
            enriched.append(entry)
            continue

        try:
            # Check agent state
            agent = client.agents.retrieve(agent_id=aid)
            entry["server_last_stop_reason"] = agent.last_stop_reason
            entry["server_last_run_completion"] = str(agent.last_run_completion)

            # Check messages
            messages = client.agents.messages.list(agent_id=aid, limit=200, order="asc")
            items = messages.items
            entry["server_total_messages"] = len(items)
            entry["server_last_message_type"] = items[-1].message_type if items else None
            entry["server_agent_completed"] = (
                items[-1].message_type == "assistant_message" if items else False
            )

            # Collect run_ids from messages
            msg_run_ids = set()
            for m in items:
                rid = getattr(m, "run_id", None)
                if rid:
                    msg_run_ids.add(rid)

            # Find ghost runs
            runs = client.runs.list(agent_id=aid, limit=100)
            ghost_runs = [r for r in runs.items if r.id not in msg_run_ids]
            entry["ghost_run_ids"] = [r.id for r in ghost_runs]
            entry["ghost_run_count"] = len(ghost_runs)

            # Get ghost run error details
            if ghost_runs:
                ghost = ghost_runs[0]
                entry["ghost_run_status"] = ghost.status
                entry["ghost_run_stop_reason"] = ghost.stop_reason
                error_meta = ghost.metadata.get("error", {}) if ghost.metadata else {}
                entry["ghost_run_error"] = error_meta.get(
                    "detail", error_meta.get("message", "")
                )

            # Check for zero-token completions
            if entry["classification"] == "extraction" and len(items) <= 2:
                try:
                    all_runs = client.runs.list(agent_id=aid, limit=10)
                    if all_runs.items:
                        steps = client.runs.steps.list(
                            run_id=all_runs.items[0].id, limit=10
                        )
                        if steps.items:
                            step = steps.items[0]
                            entry["step_completion_tokens"] = step.completion_tokens
                            entry["step_provider"] = step.provider_name
                            entry["step_stop_reason"] = step.stop_reason
                except Exception:
                    pass

        except Exception as e:
            entry["server_check_error"] = str(e)

        enriched.append(entry)

    return enriched


def print_summary(errors: list[dict], check_server_flag: bool):
    """Print a human-readable summary."""
    if not errors:
        print("No errors found.")
        return

    # Group by model
    by_model = {}
    for e in errors:
        model = e.get("model_name", "unknown")
        by_model.setdefault(model, []).append(e)

    print(f"\n{'='*60}")
    print(f"Error Analysis: {len(errors)} total errors")
    print(f"{'='*60}")

    for model, model_errors in sorted(by_model.items()):
        classifications = Counter(e["classification"] for e in model_errors)
        print(f"\n  {model}: {len(model_errors)} errors")
        for cls, count in classifications.most_common():
            print(f"    {cls}: {count}")

        if check_server_flag:
            completed = sum(1 for e in model_errors if e.get("server_agent_completed"))
            ghost = sum(1 for e in model_errors if e.get("ghost_run_count", 0) > 0)
            if completed:
                print(f"    --- server check ---")
                print(f"    agent actually completed: {completed}/{len(model_errors)}")
            if ghost:
                print(f"    agents with ghost runs: {ghost}/{len(model_errors)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze letta_evals errors")
    parser.add_argument("--results-dir", required=True, type=Path, help="Path to results directory")
    parser.add_argument("--check-server", action="store_true", help="Cross-reference with Letta server")
    parser.add_argument("--output", type=Path, help="Output JSON path (default: error_analysis.json in results dir)")
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: {args.results_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Step 1: Parse and classify
    errors = parse_results(args.results_dir)
    print(f"Found {len(errors)} errors in {args.results_dir / 'results.jsonl'}")

    # Step 2: Server check (optional)
    if args.check_server:
        non_timeout = [e for e in errors if e["classification"] != "timeout"]
        print(f"Checking {len(non_timeout)} non-timeout errors against server...")
        errors = [e for e in errors if e["classification"] == "timeout"] + check_server(non_timeout)

    # Step 3: Print summary
    print_summary(errors, args.check_server)

    # Step 4: Write output
    output_path = args.output or (args.results_dir / "error_analysis.json")
    with open(output_path, "w") as f:
        json.dump(errors, f, indent=2, default=str)
    print(f"\nFull analysis written to: {output_path}")


if __name__ == "__main__":
    main()
