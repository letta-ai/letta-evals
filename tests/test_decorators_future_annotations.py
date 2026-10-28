"""Decorated functions in a module using ``from __future__ import annotations``
(annotations stored as strings) load and run."""

from datetime import datetime, timezone

import pytest
from letta_client.types.agents import AssistantMessage

from letta_evals.graders.tool import ToolGrader
from letta_evals.models import Sample
from letta_evals.utils import load_object

SUITE_MODULE = """
from __future__ import annotations

from typing import TYPE_CHECKING

from letta_evals import extractor, grader, suite_setup
from letta_evals.models import GradeResult

if TYPE_CHECKING:  # only visible to type checkers; annotations must never be evaluated
    from letta_evals.models import Sample


@extractor
def future_upper(trajectory: list, config: dict) -> str:
    return trajectory[-1][-1].content.upper()


@grader
def future_equals(sample: Sample, submission: str) -> GradeResult:
    return GradeResult(score=float(submission == sample.ground_truth))


@suite_setup
def future_setup(client: object) -> None:
    pass
"""


@pytest.mark.asyncio
async def test_string_annotations_are_accepted(tmp_path):
    (tmp_path / "suite_funcs.py").write_text(SUITE_MODULE)

    assert load_object("suite_funcs.py:future_setup", tmp_path)._is_suite_setup

    grader = ToolGrader(
        function="suite_funcs.py:future_equals", extractor="suite_funcs.py:future_upper", base_dir=tmp_path
    )
    message = AssistantMessage(
        id="m", message_type="assistant_message", date=datetime(2026, 1, 1, tzinfo=timezone.utc), content="hi"
    )
    result, submission = await grader.grade(Sample(id=0, input="q", ground_truth="HI"), [[message]])

    assert submission == "HI"
    assert result.score == 1.0
