from typing import List, Optional

from letta_evals.models import Sample

JUDGE_SYSTEM_PROMPT = """You are an evaluation judge. You will be given:
1. A rubric describing evaluation criteria
2. An input/question
3. A submission to evaluate

Evaluate the submission according to the rubric and return a JSON response with:
{
    "score": (REQUIRED: a decimal number between 0.0 and 1.0 inclusive),
    "rationale": "explanation of your grading decision"
}

IMPORTANT:
- The score MUST be a number between 0.0 and 1.0 (inclusive)
- 0.0 means complete failure, 1.0 means perfect
- Use decimal values for partial credit (e.g., 0.25, 0.5, 0.75)
- Be objective and follow the rubric strictly"""


def build_judge_prompt(
    prompt: str,
    sample: Sample,
    submission: str,
    rubric_vars: Optional[List[str]] = None,
    judge_tool_name: Optional[str] = None,
) -> str:
    """Build a multipart judge prompt with rubric, input, ground truth, and submission.

    Args:
        prompt: The rubric text
        sample: The evaluation sample
        submission: The extracted submission to evaluate
        rubric_vars: Optional list of variable names to substitute from sample.rubric_vars
        judge_tool_name: Optional name of the tool for agent judges to call (if None, expects JSON output)

    Returns:
        Formatted prompt string with all sections
    """
    # substitute custom rubric variables
    if rubric_vars and sample.rubric_vars:
        for var_name in rubric_vars:
            var_value = str(sample.rubric_vars[var_name])
            prompt = prompt.replace(f"{{{var_name}}}", var_value)

    parts = [
        "You are an evaluation judge. An agent was given a task and returned a submission for that task.",
        "Your job is to evaluate the quality of that submission according to the rubric below.",
        "",
        "IMPORTANT: Do NOT answer the original question yourself. You are ONLY evaluating someone else's answer.",
        "",
        "---",
        "",
        "## Evaluation Rubric",
        prompt,
        "",
        "---",
        "",
        "## Original Question Given to the Agent",
        str(sample.input),
    ]

    if sample.ground_truth:
        parts.extend(
            [
                "",
                "## Expected Answer (Ground Truth)",
                sample.ground_truth,
            ]
        )

    parts.extend(
        [
            "",
            "---",
            "",
            "## The Agent's Submission (What You Are Evaluating)",
            submission,
            "",
            "---",
            "",
            "## Your Instructions",
            "",
            "1. Review the evaluation rubric carefully",
            "2. Compare the agent's submission against the original question and rubric criteria",
            "3. Determine a score between 0.0 (complete failure) and 1.0 (perfect)",
        ]
    )

    if judge_tool_name:
        # letta agent judge - must call tool
        parts.extend(
            [
                f"4. Use the {judge_tool_name} tool to submit your evaluation with:",
                "   - score: a float between 0.0 and 1.0",
                "   - rationale: your reasoning for the score",
                "",
                f"Do NOT respond with text - you MUST call the {judge_tool_name} tool to complete the evaluation.",
            ]
        )
    else:
        # llm judge - return json
        parts.extend(
            [
                "4. Return your evaluation as JSON with:",
                "   - score: a float between 0.0 and 1.0",
                "   - rationale: your reasoning for the score",
            ]
        )

    return "\n".join(parts)
