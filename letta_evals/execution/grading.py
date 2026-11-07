"""Runner-adjacent grading helpers.

These helpers are extracted from ``Runner`` so the runner can focus on target
execution, progress sequencing, result assembly, and orchestration.
"""

import time
from typing import Any, Dict, List, Optional

from jinja2 import TemplateSyntaxError

from letta_evals.graders.base import Grader
from letta_evals.graders.prompt_utils import RESERVED_VARS, check_rubric
from letta_evals.graders.scorers import sample_criteria
from letta_evals.models import (
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
    context: Optional[Dict[str, Any]],
    graders: Dict[str, Grader],
) -> tuple[Dict[str, GradeResult], Dict[str, str], Dict[str, float]]:
    """Grade a sample across all graders. Returns (grades, submissions, per_grader_time)."""
    grades_dict: Dict[str, GradeResult] = {}
    submissions_dict: Dict[str, str] = {}
    per_grader_time: Dict[str, float] = {}

    for key, grader in graders.items():
        t_grader_start = time.perf_counter()
        grade, submission = await grader.grade(sample, trajectory, context=context)
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


def validate_model_judges(suite: SuiteSpec, samples: List[Sample]) -> None:
    """Check every model judge's inputs before any agent runs.

    For each sample, the rubric it will be judged with (its own, or the
    grader's) must be a valid template, must not use the old single-brace
    placeholders, and must only reference variables the sample provides. With
    the checklist scorer, the sample's criteria must be well formed.
    """
    judges = {key: spec for key, spec in (suite.graders or {}).items() if isinstance(spec, ModelJudgeGraderSpec)}
    for grader_key, grader_spec in judges.items():
        for sample in samples:
            rubric = sample.rubric if sample.rubric is not None else grader_spec.prompt
            provided = set(RESERVED_VARS) | set(sample.rubric_vars or {})
            try:
                check_rubric(rubric, provided)
                if grader_spec.scorer == "checklist":
                    sample_criteria(sample)
            except (ValueError, TemplateSyntaxError) as e:
                raise ValueError(f"Sample {sample.id}, grader '{grader_key}': {e}") from e
