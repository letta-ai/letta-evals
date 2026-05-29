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
    PerTurnGrade,
    Sample,
    SampleId,
    SuiteSpec,
)
from letta_evals.types import ErrorCategory
from letta_evals.utils import build_turn_summary, is_per_turn_evaluation
from letta_evals.visualization.base import ProgressCallback


async def grade_per_turn(
    sample: Sample,
    trajectory: List[List[LettaMessageUnion]],
    agent_state: Optional[AgentState],
    grader: Grader,
    grader_key: str,
    sample_id: SampleId,
    agent_id: str,
    model_handle: str,
    progress_callback: Optional[ProgressCallback],
) -> tuple[GradeResult, str]:
    """Grade each turn independently and return averaged GradeResult + combined submission."""
    ground_truths = sample.ground_truth  # type: List[str]
    num_turns = len(ground_truths)
    per_turn_grades: List[PerTurnGrade] = []
    grader_extraction_time = 0.0

    for turn_idx in range(num_turns):
        single_turn_trajectory = [trajectory[turn_idx]] if turn_idx < len(trajectory) else []

        turn_sample = Sample(
            id=sample.id,
            input=sample.input[turn_idx] if isinstance(sample.input, list) else sample.input,
            ground_truth=ground_truths[turn_idx],
            agent_args=sample.agent_args,
            rubric_vars=sample.rubric_vars,
            extra_vars=sample.extra_vars,
            rubric=sample.rubric,
        )

        turn_grade, turn_submission = await grader.grade(turn_sample, single_turn_trajectory, agent_state=agent_state)
        grader_extraction_time += turn_grade.metadata.get("extraction_time", 0.0)

        per_turn_grades.append(
            PerTurnGrade(
                turn=turn_idx,
                score=turn_grade.score,
                rationale=turn_grade.rationale,
                submission=turn_submission,
            )
        )

        if progress_callback:
            await progress_callback.turn_graded(
                sample_id=sample_id,
                turn_num=turn_idx,
                total_turns=num_turns,
                turn_score=turn_grade.score,
                grader_key=grader_key,
                agent_id=agent_id,
                model_handle=model_handle,
            )

    turn_scores = [g.score for g in per_turn_grades]
    final_score = sum(turn_scores) / num_turns if num_turns > 0 else 0.0
    turns_passed = sum(1 for sc in turn_scores if sc >= 1.0)

    summary_rationale = build_turn_summary(turn_scores)
    combined_submission = " | ".join(f"[Turn {g.turn}] {g.submission}" for g in per_turn_grades)

    grade = GradeResult(
        score=final_score,
        rationale=summary_rationale,
        per_turn_grades=per_turn_grades,
        metadata={
            "turns_passed": turns_passed,
            "turns_total": num_turns,
            "extraction_time": grader_extraction_time,
        },
    )
    return grade, combined_submission


async def grade_sample(
    sample: Sample,
    trajectory: List[List[LettaMessageUnion]],
    agent_state: Optional[AgentState],
    graders: Dict[str, Grader],
    sample_id: SampleId,
    agent_id: str,
    model_handle: str,
    progress_callback: Optional[ProgressCallback],
) -> tuple[Dict[str, GradeResult], Dict[str, str], Dict[str, float]]:
    """Grade a sample across all graders. Returns (grades, submissions, per_grader_time)."""
    grades_dict: Dict[str, GradeResult] = {}
    submissions_dict: Dict[str, str] = {}
    per_grader_time: Dict[str, float] = {}

    is_per_turn = is_per_turn_evaluation(sample)

    for key, grader in graders.items():
        t_grader_start = time.perf_counter()

        if is_per_turn:
            grade, submission = await grade_per_turn(
                sample,
                trajectory,
                agent_state,
                grader,
                key,
                sample_id,
                agent_id,
                model_handle,
                progress_callback,
            )
        else:
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
