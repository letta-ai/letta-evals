import json
from typing import List

from letta_client import LettaMessageUnion, AssistantMessage
from letta_evals.decorators import extractor, grader
from letta_evals.models import GradeResult, Sample


@extractor
def CoreMemoryResponseExtractor(trajectory: List[List[LettaMessageUnion]], config: dict) -> str:
    """Extract the final assistant message from the conversation trajectory."""
    
    # Go through trajectory in reverse to find the last assistant message
    for turn in reversed(trajectory):
        for message in reversed(turn):
            if isinstance(message, AssistantMessage) and message.content:
                # Return just the message content, which should be the answer
                return message.content.strip()
    
    # If no assistant message found, return empty string
    return ""


@grader
def grade_core_memory_read(sample: Sample, submission: str) -> GradeResult:
    """
    Grade whether the agent correctly read information from core memory.
    
    This evaluator checks if the agent's response contains the expected answer
    from the ground truth, allowing for some flexibility in phrasing.
    """
    
    # Get the expected answer from ground truth
    expected_answer = sample.ground_truth.strip().lower()
    agent_response = submission.strip().lower()
    
    if not agent_response:
        return GradeResult(
            score=0.0, 
            rationale="No response from agent"
        )
    
    if not expected_answer:
        return GradeResult(
            score=0.0, 
            rationale="No expected answer provided"
        )
    
    # Check if the expected answer appears in the agent's response
    if expected_answer in agent_response:
        return GradeResult(
            score=1.0,
            rationale=f"Correct answer '{expected_answer}' found in response: '{submission}'"
        )
    
    # Check for exact match (case insensitive)
    if expected_answer == agent_response:
        return GradeResult(
            score=1.0,
            rationale=f"Exact match for answer: '{expected_answer}'"
        )
    
    # For some tolerance, check if key words from expected answer are present
    expected_words = set(expected_answer.split())
    response_words = set(agent_response.split())
    
    # If more than 50% of expected words are in response, give partial credit
    if expected_words and len(expected_words.intersection(response_words)) >= len(expected_words) * 0.5:
        overlap_ratio = len(expected_words.intersection(response_words)) / len(expected_words)
        return GradeResult(
            score=overlap_ratio,
            rationale=f"Partial match: {overlap_ratio:.2f} word overlap between expected '{expected_answer}' and response '{submission}'"
        )
    
    return GradeResult(
        score=0.0,
        rationale=f"Expected answer '{expected_answer}' not found in response '{submission}'"
    )


@grader
def grade_core_memory_read_strict(sample: Sample, submission: str) -> GradeResult:
    """
    Strict grader that requires exact or very close matching for core memory read tasks.
    """
    
    expected_answer = sample.ground_truth.strip().lower()
    agent_response = submission.strip().lower()
    
    if not agent_response:
        return GradeResult(score=0.0, rationale="No response from agent")
    
    if not expected_answer:
        return GradeResult(score=0.0, rationale="No expected answer provided")
    
    # Check for exact match
    if expected_answer == agent_response:
        return GradeResult(score=1.0, rationale=f"Exact match: '{expected_answer}'")
    
    # Check if expected answer is contained in response (for longer responses)
    if expected_answer in agent_response:
        return GradeResult(score=1.0, rationale=f"Expected answer '{expected_answer}' found in response")
    
    return GradeResult(
        score=0.0,
        rationale=f"Expected '{expected_answer}' but got '{submission}'"
    )