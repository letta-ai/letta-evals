import json
from typing import List

from letta_client import LettaMessageUnion, ToolCallMessage

from letta_evals.graders.extractors.base import SubmissionExtractor
from letta_evals.models import GradeResult, Sample


class MemoryInsertExtractor(SubmissionExtractor):
    """Extract memory_insert tool calls from trajectory."""

    def extract(self, trajectory: List[List[LettaMessageUnion]]) -> str:
        for turn in trajectory:
            for message in turn:
                if isinstance(message, ToolCallMessage) and message.tool_call.name == "memory_insert":
                    return message.tool_call.arguments

        return "{}"


def grade_fruit_preference(sample: Sample, submission: str) -> GradeResult:
    """Grade if the fruit preference was correctly stored in memory."""
    try:
        args = json.loads(submission)
    except json.JSONDecodeError:
        return GradeResult(score=0.0, rationale="No valid memory_insert tool call found")

    # check label is user_fruit_preferences
    label = args.get("label", "")
    if label != "user_fruit_preferences":
        return GradeResult(
            score=0.0, rationale=f"Wrong memory block label: expected 'user_fruit_preferences', got '{label}'"
        )

    # check fruit name is in new_str
    fruit = sample.ground_truth.lower()
    new_str = args.get("new_str", "").lower()

    if fruit not in new_str:
        return GradeResult(score=0.0, rationale=f"Fruit '{fruit}' not found in new_str")

    # check inserted at line 0
    insert_line = args.get("insert_line", -1)
    if insert_line != 0:
        return GradeResult(score=0.7, rationale=f"Fruit preference stored but not at line 0 (line: {insert_line})")

    return GradeResult(score=1.0, rationale="Fruit preference correctly stored")
