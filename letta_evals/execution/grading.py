"""Runner-adjacent grading helpers.

These helpers are extracted from ``Runner`` so the runner can focus on target
execution, progress sequencing, result assembly, and orchestration.
"""

import string
import time
from typing import Dict, List, Optional

from letta_evals.graders.base import Grader
from letta_evals.models import (
    AgentState,
    Error,
    GradeResult,
    LettaMessageUnion,
    ModelJudgeGraderSpec,
    Sample,
    SuiteSpec,
)
from letta_evals.types import ErrorCategory


async def grade_sample(
    sample: Sample,
    trajectory: List[List[LettaMessageUnion]],
    agent_state: Optional[AgentState],
    graders: Dict[str, Grader],
) -> tuple[Dict[str, GradeResult], Dict[str, str], Dict[str, float]]:
    """Grade a sample across all graders. Returns (grades, submissions, per_grader_time)."""
    grades_dict: Dict[str, GradeResult] = {}
    submissions_dict: Dict[str, str] = {}
    per_grader_time: Dict[str, float] = {}

    for key, grader in graders.items():
        t_grader_start = time.perf_counter()
        grade, submission = await grader.grade(sample, trajectory, agent_state=agent_state)
        per_grader_time[key] = time.perf_counter() - t_grader_start
        grades_dict[key] = grade
        submissions_dict[key] = submission

    return grades_dict, submissions_dict, per_grader_time


def detect_errors(
    grades_dict: Dict[str, GradeResult],
    trajectory: list,
    submissions: Dict[str, str],
) -> Optional[Error]:
    """Detect extraction or grading errors from results."""
    if grades_dict:
        first_key = next(iter(grades_dict.keys()))
        first_grade = grades_dict[first_key]
        first_submission = submissions.get(first_key, "")
        is_extraction_error = first_grade.score == 0.0 and (
            not trajectory
            or not first_submission
            or (
                first_grade.rationale
                and ("Empty trajectory" in first_grade.rationale or "Empty submission" in first_grade.rationale)
            )
        )
        if is_extraction_error:
            return Error(
                category=ErrorCategory.EXTRACTION,
                exception_type="ExtractionError",
                message=first_grade.rationale or "Empty trajectory or submission",
            )

    grading_errors = {k: gr.metadata["error"] for k, gr in grades_dict.items() if gr.metadata.get("error")}
    if grading_errors:
        details = "; ".join(f"{k}: {v}" for k, v in grading_errors.items())
        return Error(
            category=ErrorCategory.GRADING,
            exception_type="GradingError",
            message=f"Grading failed for: {details}",
        )

    return None


def validate_rubric_vars(suite: SuiteSpec, samples: List[Sample]) -> None:
    """Validate model-judge rubric placeholders against per-sample rubric_vars."""
    if not suite.graders:
        return

    for grader_key, grader_spec in suite.graders.items():
        if not isinstance(grader_spec, ModelJudgeGraderSpec):
            continue
        rubric_text = grader_spec.prompt
        if rubric_text is None:
            continue

        referenced: set = set()
        for _, field_name, _, _ in string.Formatter().parse(rubric_text):
            if field_name:
                referenced.add(field_name.split(".")[0].split("[")[0])

        reserved = {"input", "ground_truth", "submission"}
        extras_needed = referenced - reserved
        if not extras_needed:
            continue

        for sample in samples:
            if sample.rubric is not None:
                continue
            provided = set((sample.rubric_vars or {}).keys())
            missing = extras_needed - provided
            if missing:
                raise ValueError(
                    f"Sample {sample.id} is missing rubric variables required by "
                    f"grader '{grader_key}': {sorted(missing)}. "
                    f"Add them to sample.rubric_vars, or override the rubric "
                    f"via sample.rubric / sample.rubric_path."
                )
