import json
import re
from typing import List

from letta_evals.decorators import extractor, grader
from letta_evals.extractors.builtin import last_assistant
from letta_evals.models import GradeResult, LettaMessageUnion, Sample


@extractor
def json_object_extractor(trajectory: List[List[LettaMessageUnion]], config: dict) -> str:
    """Extract the first JSON object from the last assistant response."""
    text = last_assistant(trajectory, config).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else text


@grader
def ticket_classification_grader(sample: Sample, submission: str) -> GradeResult:
    """Grade a JSON ticket classification response."""
    try:
        observed = json.loads(submission)
    except json.JSONDecodeError:
        return GradeResult(score=0.0, rationale="Submission was not valid JSON")

    try:
        expected = json.loads(sample.ground_truth or "{}")
    except json.JSONDecodeError:
        return GradeResult(score=0.0, rationale="Ground truth was not valid JSON")

    if not isinstance(expected, dict):
        return GradeResult(score=0.0, rationale="Ground truth must be a JSON object")

    category_ok = str(observed.get("category", "")).lower() == str(expected.get("category", "")).lower()
    priority_ok = str(observed.get("priority", "")).lower() == str(expected.get("priority", "")).lower()
    score = (float(category_ok) + float(priority_ok)) / 2

    return GradeResult(
        score=score,
        rationale=f"category_ok={category_ok}, priority_ok={priority_ok}",
        metadata={"observed": observed, "expected": expected},
    )
