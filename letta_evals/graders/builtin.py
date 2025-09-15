import re

from letta_evals.decorators import grader
from letta_evals.models import GradeResult, Sample


@grader
def exact_match(sample: Sample, submission: str) -> GradeResult:
    """Check if submission exactly matches ground_truth."""
    if not sample.ground_truth:
        return GradeResult(score=0.0, rationale="No ground_truth answer provided")

    matches = submission.strip() == sample.ground_truth.strip()
    score = 1.0 if matches else 0.0
    return GradeResult(score=score, rationale=f"Exact match: {matches}")


@grader
def contains(sample: Sample, submission: str) -> GradeResult:
    """Check if submission contains ground_truth answer."""
    if not sample.ground_truth:
        return GradeResult(score=0.0, rationale="No ground_truth answer provided")

    found = sample.ground_truth.lower() in submission.lower()
    score = 1.0 if found else 0.0
    return GradeResult(score=score, rationale=f"Contains ground_truth: {found}")


@grader
def regex_match(sample: Sample, submission: str) -> GradeResult:
    """Check if submission matches ground_truth regex pattern."""
    if not sample.ground_truth:
        return GradeResult(score=0.0, rationale="No ground_truth regex pattern provided")

    try:
        pattern = re.compile(sample.ground_truth)
        matches = bool(pattern.search(submission))
        score = 1.0 if matches else 0.0
        return GradeResult(score=score, rationale=f"Regex match: {matches}")
    except re.error as e:
        return GradeResult(score=0.0, rationale=f"Invalid regex pattern: {e}")
