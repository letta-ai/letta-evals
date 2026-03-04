#!/usr/bin/env python3
"""Validate generated questions for data quality issues."""

import argparse
import importlib.util
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

try:
    from audit_dataset import AuditResult, audit_dataset_rows, load_dataset_rows, summarize_audit_results
except ModuleNotFoundError:
    audit_path = Path(__file__).with_name("audit_dataset.py")
    spec = importlib.util.spec_from_file_location("audit_dataset", audit_path)
    audit_dataset = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = audit_dataset
    spec.loader.exec_module(audit_dataset)
    AuditResult = audit_dataset.AuditResult
    audit_dataset_rows = audit_dataset.audit_dataset_rows
    load_dataset_rows = audit_dataset.load_dataset_rows
    summarize_audit_results = audit_dataset.summarize_audit_results


def load_questions(path: str | Path) -> list:
    """Load questions from JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def check_forbidden_terms(questions: list) -> list:
    """Check for SSN, neighbor, and other forbidden terms."""
    issues = []
    for i, q in enumerate(questions):
        question = q.get("question", "")
        flags = []
        if "SSN" in question:
            flags.append("SSN")
        if "neighbor" in question.lower():
            flags.append("neighbor")
        if flags:
            issues.append({"id": i, "type": "forbidden_term", "flags": flags})
    return issues


def check_answer_quality(questions: list) -> list:
    """Check for invalid answer formats."""
    issues = []
    question_phrases = ["the person with", "among", "residents of", "who has", "who owns"]
    negative_phrases = ["none", "no one", "nobody", "n/a", "does not", "doesn't"]

    for i, q in enumerate(questions):
        answer = q.get("answer", "")
        flags = []

        # Check for question text in answer
        if any(p in answer.lower() for p in question_phrases):
            flags.append("contains_question_text")

        # Check for negative answers
        if any(p in answer.lower() for p in negative_phrases):
            flags.append("negative_answer")

        # Check for overly long answers
        if len(answer) > 100:
            flags.append(f"too_long_{len(answer)}_chars")

        if flags:
            issues.append({"id": i, "type": "answer_quality", "flags": flags, "answer": answer[:50]})

    return issues


def check_verification_query(questions: list) -> list:
    """Check verification query quality."""
    issues = []
    for i, q in enumerate(questions):
        vq = q.get("verification_query", "")
        flags = []

        if not vq or not vq.strip():
            flags.append("missing")
        else:
            if "pers-" in vq:
                flags.append("hardcoded_person_id")
            if "CASE" in vq.upper() and "THEN '" in vq:
                flags.append("hardcoded_case_statement")
            if vq.count("SELECT") < 2:
                flags.append("not_end_to_end")
            if has_address_join_multiplicity_risk(vq):
                flags.append("address_join_multiplicity_risk")

        if flags:
            issues.append({"id": i, "type": "verification_query", "flags": flags})

    return issues


def has_address_join_multiplicity_risk(query: str) -> bool:
    """Flag common aggregation patterns that duplicate rows via raw addresses joins."""
    normalized = " ".join(query.lower().split())
    if "join addresses" not in normalized:
        return False
    if not any(func in normalized for func in ("count(", "sum(", "avg(")):
        return False

    risky_patterns = [
        r"(from|join)\s+(credit_cards|bank_accounts|vehicles|insurance_policies|internet_accounts|employments|pets|medical_records)\b.{0,160}join\s+addresses\b.{0,240}(count|sum|avg)\s*\(",
        r"join\s+addresses\b.{0,160}join\s+(credit_cards|bank_accounts|vehicles|insurance_policies|internet_accounts|employments|pets|medical_records)\b.{0,240}(count|sum|avg)\s*\(",
    ]
    return any(re.search(pattern, normalized) for pattern in risky_patterns)


def validate_gt_against_db(questions: list, db_path: str, sample_size: int | None = None) -> list:
    """Validate ground truths against database for all questions or a bounded sample."""
    issues = []
    db = sqlite3.connect(db_path)

    # Only validate questions with verification_query
    questions_with_vq = [(i, q) for i, q in enumerate(questions) if q.get("verification_query")]

    if not questions_with_vq:
        return [{"id": -1, "type": "no_verification_queries", "flags": ["none_found"]}]

    if sample_size is None or sample_size <= 0:
        sample = questions_with_vq
    else:
        import random

        sample = random.sample(questions_with_vq, min(sample_size, len(questions_with_vq)))

    for i, q in sample:
        vq = q.get("verification_query", "")
        gt = q.get("answer") or q.get("ground_truth", "")

        try:
            cursor = db.execute(vq)
            rows = cursor.fetchall()

            if len(rows) != 1:
                issues.append(
                    {"id": i, "type": "gt_validation", "flags": [f"query_returned_{len(rows)}_rows"], "gt": gt}
                )
                continue

            db_value = str(list(rows[0])[0]) if rows[0] else ""

            # Compare
            gt_normalized = gt.strip()

            # Try numeric comparison
            try:
                gt_num = float(gt_normalized.replace("$", "").replace(",", ""))
                db_num = float(db_value.replace("$", "").replace(",", ""))
                if abs(gt_num - db_num) > 0.01:
                    issues.append(
                        {"id": i, "type": "gt_validation", "flags": ["value_mismatch"], "gt": gt, "db_value": db_value}
                    )
            except (ValueError, TypeError):
                # String comparison
                if db_value.lower() != gt_normalized.lower():
                    issues.append(
                        {"id": i, "type": "gt_validation", "flags": ["value_mismatch"], "gt": gt, "db_value": db_value}
                    )

        except Exception as e:
            issues.append({"id": i, "type": "gt_validation", "flags": ["query_error"], "error": str(e)[:100]})

    db.close()
    return issues


def resolve_parsed_dataset_path(questions_path: Path) -> Path | None:
    """Find the parsed dataset artifact that corresponds to a raw generation file."""
    if questions_path.name == "agent_generated_questions.jsonl":
        candidate = questions_path.with_name("agent_generated_questions_parsed.jsonl")
        if candidate.exists():
            return candidate
    return None


def audit_parsed_dataset(parsed_path: Path, db_path: Path) -> list[dict]:
    """Audit the parsed dataset artifact and convert findings to validator issues."""
    rows = load_dataset_rows(parsed_path)
    results = audit_dataset_rows(rows, db_path)
    summary = summarize_audit_results(results)

    print(f"\n{'=' * 70}")
    print("Auditing Parsed Dataset...")
    print("=" * 70)
    print(f"Parsed file: {parsed_path}")
    for status in sorted(summary):
        print(f"  {status:20s} {summary[status]}")

    issues = []
    for result in results:
        if result.status == "correct":
            continue
        issue = {
            "id": result.index,
            "type": "parsed_dataset_audit",
            "flags": [result.status],
            "gt": result.ground_truth,
            "valid_answers": result.valid_answers,
            "question": result.question,
        }
        if result.note:
            issue["note"] = result.note
        issues.append(issue)

    if issues:
        for issue in issues:
            valid = ", ".join(issue["valid_answers"][:8]) if issue["valid_answers"] else "<none>"
            note = f" note='{issue['note']}'" if issue.get("note") else ""
            print(f"  Q{issue['id']}: {issue['flags']} - GT='{issue['gt']}' valid='{valid}'{note}")
    else:
        print("  ✓ Parsed dataset audit found no issues")

    return issues


def run_validation(
    questions_path: str | Path,
    sample_size: int = 0,
    skip_parsed_audit: bool = False,
    db_path: str | Path | None = None,
) -> list[dict]:
    """Run validation checks for a raw generation artifact and optional sibling parsed dataset audit."""
    questions_path = Path(questions_path)
    if db_path is None:
        db_path = Path(__file__).parent / "data" / "letta_file_bench.db"
    else:
        db_path = Path(db_path)

    print("=" * 70)
    print("FILESYSTEM EVAL - QUESTION VALIDATION")
    print("=" * 70)
    print(f"\nFile: {questions_path}")

    questions = load_questions(questions_path)
    print(f"Total questions: {len(questions)}")

    # Distribution
    print(f"\n{'=' * 70}")
    print("Question Type Distribution")
    print("=" * 70)
    types = Counter(q.get("question_type") for q in questions)
    for t, c in types.most_common():
        print(f"  {t:30s} {c:2d} ({c / len(questions) * 100:.0f}%)")

    # Run checks
    all_issues = []

    print(f"\n{'=' * 70}")
    print("Checking for forbidden terms...")
    print("=" * 70)
    issues = check_forbidden_terms(questions)
    all_issues.extend(issues)
    if issues:
        for issue in issues:
            print(f"  Q{issue['id']}: {issue['flags']}")
    else:
        print("  ✓ None found")

    print(f"\n{'=' * 70}")
    print("Checking answer quality...")
    print("=" * 70)
    issues = check_answer_quality(questions)
    all_issues.extend(issues)
    if issues:
        for issue in issues:
            print(f"  Q{issue['id']}: {issue['flags']} - '{issue.get('answer', '')}'")
    else:
        print("  ✓ All answers valid")

    print(f"\n{'=' * 70}")
    print("Checking verification queries...")
    print("=" * 70)
    issues = check_verification_query(questions)
    all_issues.extend(issues)
    if issues:
        for issue in issues:
            print(f"  Q{issue['id']}: {issue['flags']}")
    else:
        print("  ✓ All verification queries valid")

    print(f"\n{'=' * 70}")
    print("Validating GTs against database...")
    print("=" * 70)
    if db_path.exists():
        issues = validate_gt_against_db(questions, str(db_path), sample_size=sample_size)
        all_issues.extend(issues)
        if issues:
            for issue in issues:
                print(
                    f"  Q{issue['id']}: {issue['flags']} - GT='{issue.get('gt', '')}' DB='{issue.get('db_value', '')}'"
                )
        else:
            checked = "all" if sample_size <= 0 else f"{min(sample_size, len(questions))} sampled"
            print(f"  ✓ All {checked} GTs match database")
    else:
        print(f"  ⚠ Database not found at {db_path}")

    parsed_path = None if skip_parsed_audit else resolve_parsed_dataset_path(questions_path)
    if parsed_path and db_path.exists():
        issues = audit_parsed_dataset(parsed_path, db_path)
        all_issues.extend(issues)
    elif questions_path.name == "agent_generated_questions.jsonl" and not skip_parsed_audit:
        print(f"\n{'=' * 70}")
        print("Auditing Parsed Dataset...")
        print("=" * 70)
        print("  ⚠ Parsed dataset not found; skipping parsed audit")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"Total questions: {len(questions)}")
    print(f"Total issues: {len(all_issues)}")

    if all_issues:
        print("\n❌ ISSUES FOUND - review before testing")
        by_type = Counter(i["type"] for i in all_issues)
        for t, c in by_type.most_common():
            print(f"  {t}: {c}")
    else:
        print("\n✅ DATASET LOOKS CLEAN - ready for testing")

    return all_issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions_path", help="Path to agent_generated_questions.jsonl")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Validate this many verification queries instead of the full file. 0 means validate all.",
    )
    parser.add_argument(
        "--skip-parsed-audit",
        action="store_true",
        help="Skip auditing the sibling parsed dataset artifact when validating agent_generated_questions.jsonl.",
    )
    args = parser.parse_args()

    issues = run_validation(
        args.questions_path,
        sample_size=args.sample_size,
        skip_parsed_audit=args.skip_parsed_audit,
    )
    raise SystemExit(1 if issues else 0)


if __name__ == "__main__":
    main()
