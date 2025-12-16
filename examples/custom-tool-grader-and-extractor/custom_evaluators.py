import json
from typing import List

from letta_client.types.agents import ToolCallMessage

from letta_evals.decorators import extractor, grader
from letta_evals.models import GradeResult, LettaMessageUnion, Sample


@extractor
def memory_insert_extractor(trajectory: List[List[LettaMessageUnion]], config: dict) -> str:
    """Extract memory_insert tool calls from trajectory."""
    for turn in trajectory:
        for message in turn:
            if isinstance(message, ToolCallMessage):
                # SDK v1.0 uses tool_calls (array), fall back to tool_call (singular) for compatibility
                tool_calls = (
                    message.tool_calls if message.tool_calls else ([message.tool_call] if message.tool_call else [])
                )
                for tool_call in tool_calls:
                    if tool_call.name == "memory_insert":
                        return tool_call.arguments

    return "{}"


@grader
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

    return GradeResult(score=1.0, rationale="Fruit preference correctly stored")
