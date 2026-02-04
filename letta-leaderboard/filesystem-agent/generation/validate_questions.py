#!/usr/bin/env python3
"""
Validate generated questions for data quality issues.

Usage:
    python validate_questions.py <path_to_agent_generated_questions.jsonl>

Example:
    python validate_questions.py data/generated_questions/run_20260203_211515/agent_generated_questions.jsonl
"""

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path


def load_questions(path: str) -> list:
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

        if flags:
            issues.append({"id": i, "type": "verification_query", "flags": flags})

    return issues


def validate_gt_against_db(questions: list, db_path: str, sample_size: int = 10) -> list:
    """Validate ground truths against database for a sample of questions."""
    issues = []
    db = sqlite3.connect(db_path)

    # Only validate questions with verification_query
    questions_with_vq = [(i, q) for i, q in enumerate(questions) if q.get("verification_query")]

    if not questions_with_vq:
        return [{"id": -1, "type": "no_verification_queries", "flags": ["none_found"]}]

    # Sample
    import random

    sample = random.sample(questions_with_vq, min(sample_size, len(questions_with_vq)))

    for i, q in sample:
        vq = q.get("verification_query", "")
        gt = q.get("answer", "")

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


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    questions_path = sys.argv[1]

    # Find DB path relative to questions
    script_dir = Path(__file__).parent
    db_path = script_dir / "data" / "letta_file_bench.db"

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
    print("Validating GTs against database (sample)...")
    print("=" * 70)
    if db_path.exists():
        issues = validate_gt_against_db(questions, str(db_path))
        all_issues.extend(issues)
        if issues:
            for issue in issues:
                print(
                    f"  Q{issue['id']}: {issue['flags']} - GT='{issue.get('gt', '')}' DB='{issue.get('db_value', '')}'"
                )
        else:
            print("  ✓ All sampled GTs match database")
    else:
        print(f"  ⚠ Database not found at {db_path}")

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
        sys.exit(1)
    else:
        print("\n✅ DATASET LOOKS CLEAN - ready for testing")
        sys.exit(0)


if __name__ == "__main__":
    main()
