"""Tests for the redesigned rubric grader.

Covers:
- ``build_judge_prompt`` Jinja rendering, and ``check_rubric``'s template checks.
- ``Sample.rubric`` / ``Sample.rubric_path`` mutual-exclusion validation.
- The JSONL/CSV loader resolving per-sample rubric overrides.
- ``RubricGrader.grade`` precedence (per-sample rubric > grader prompt),
  the system prompt file, reasoning effort and output-token settings sent
  to each provider, and the absence of legacy wrapper injection in the
  prompt sent to the judge.
- The pre-run validator (``validate_model_judges``) checking every
  sample's rubric before any agent runs.
- Response formats and scorers: the built-in ``score`` and ``checklist``,
  custom ones loaded from the suite, and re-asking the judge on rejected replies.
- ``ModelJudgeGraderSpec`` validation of the judge settings.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from jinja2 import TemplateSyntaxError
from jinja2.exceptions import SecurityError
from letta_client.types.agents import AssistantMessage
from pydantic import ValidationError

from letta_evals.datasets.loader import load_csv, load_jsonl
from letta_evals.execution.grading import validate_model_judges
from letta_evals.graders.prompt_utils import RESERVED_VARS, build_judge_prompt, check_rubric
from letta_evals.graders.rubric import MAX_ATTEMPTS, RubricGrader
from letta_evals.graders.scorers import (
    ChecklistResponse,
    CriterionVerdict,
    ScoreResponse,
    checklist,
    sample_criteria,
)
from letta_evals.models import (
    LettaCodeTargetSpec,
    MetricRewardSpec,
    ModelJudgeGraderSpec,
    Sample,
    SuiteSpec,
)
from letta_evals.types import LLMProvider

_FAKE_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _trajectory(text: str) -> list[list]:
    return [[AssistantMessage(id="msg-0", message_type="assistant_message", date=_FAKE_DATE, content=text)]]


# ── build_judge_prompt ────────────────────────────────────────────────────────


class TestBuildJudgePrompt:
    def test_basic_substitution(self):
        s = Sample(id=0, input="Q?", ground_truth="A", rubric_vars={"k": "v"})
        out = build_judge_prompt("in={{ input }} gt={{ ground_truth }} sub={{ submission }} k={{ k }}", s, "SUB")
        assert out == "in=Q? gt=A sub=SUB k=v"

    def test_ground_truth_none_becomes_empty_string(self):
        s = Sample(id=0, input="Q?", ground_truth=None)
        out = build_judge_prompt("[{{ ground_truth }}]", s, "x")
        assert out == "[]"

    def test_input_list_renders_as_list(self):
        s = Sample(id=0, input=["turn1", "turn2"], ground_truth="A")
        assert build_judge_prompt("{{ input }}", s, "x") == "['turn1', 'turn2']"
        assert build_judge_prompt("{% for t in input %}<{{ t }}>{% endfor %}", s, "x") == "<turn1><turn2>"

    def test_missing_var_raises_key_error_with_available_list(self):
        s = Sample(id=0, input="Q?", rubric_vars={"foo": "bar"})
        with pytest.raises(KeyError) as exc:
            build_judge_prompt("hello {{ missing }}", s, "x")
        msg = str(exc.value)
        assert "missing" in msg
        assert "foo" in msg
        for r in RESERVED_VARS:
            assert r in msg

    def test_reserved_collision_raises_value_error(self):
        s = Sample(id=0, input="x", rubric_vars={"input": "shadow"})
        with pytest.raises(ValueError, match="reserved"):
            build_judge_prompt("hi {{ input }}", s, "sub")

    def test_single_braces_are_literal(self):
        s = Sample(id=0, input="Q?")
        out = build_judge_prompt('Reply like {"score": 1} for {{ input }}', s, "x")
        assert out == 'Reply like {"score": 1} for Q?'

    def test_structured_rubric_vars_can_be_looped_over(self):
        criteria = [{"id": "c1", "text": "Names Paris", "weight": 2}, {"id": "c2", "text": "Is short"}]
        s = Sample(id=0, input="Q?", rubric_vars={"criteria": criteria})
        out = build_judge_prompt("{% for c in criteria %}- [{{ c.id }}] {{ c.text }}\n{% endfor %}", s, "x")
        assert out == "- [c1] Names Paris\n- [c2] Is short\n"

    def test_trailing_newline_kept(self):
        s = Sample(id=0, input="Q?")
        assert build_judge_prompt("{{ input }}\n", s, "x") == "Q?\n"

    def test_sandbox_blocks_python_internals(self):
        s = Sample(id=0, input="Q?")
        with pytest.raises(SecurityError):
            build_judge_prompt("{{ input.__class__.__mro__ }}", s, "x")

    def test_rubric_vars_None_treated_as_empty(self):
        s = Sample(id=0, input="Q?", rubric_vars=None)
        out = build_judge_prompt("{{ input }}/{{ submission }}", s, "SUB")
        assert out == "Q?/SUB"


class TestCheckRubric:
    PROVIDED = {"input", "ground_truth", "submission", "category"}

    def test_valid_rubric_passes(self):
        check_rubric("{{ input }} {{ category }} {% for x in [1] %}{{ x }}{% endfor %}", self.PROVIDED)

    def test_single_brace_placeholder_rejected(self):
        with pytest.raises(ValueError, match=r"\{input\} -> \{\{ input \}\}"):
            check_rubric("Grade {input} against {{ ground_truth }}", self.PROVIDED)

    def test_single_brace_rubric_var_rejected(self):
        with pytest.raises(ValueError, match="category"):
            check_rubric("Category: { category }", self.PROVIDED)

    def test_unrelated_single_braces_allowed(self):
        check_rubric('Reply like {"score": 1} or {score}; set {a, b}', self.PROVIDED)

    def test_missing_variable_rejected(self):
        with pytest.raises(ValueError, match="doesn't provide"):
            check_rubric("{{ input }} {{ topic }}", self.PROVIDED)

    def test_syntax_error_raised(self):
        with pytest.raises(TemplateSyntaxError):
            check_rubric("{% for x in %}", self.PROVIDED)


# ── Sample rubric / rubric_path validation ─────────────────────────────────


class TestSampleRubricFields:
    def test_rubric_only(self):
        s = Sample(id=0, input="x", rubric="some rubric")
        assert s.rubric == "some rubric"
        assert s.rubric_path is None

    def test_rubric_path_only(self):
        s = Sample(id=0, input="x", rubric_path=Path("/tmp/x.txt"))
        assert s.rubric is None
        assert s.rubric_path == Path("/tmp/x.txt")

    def test_both_set_raises(self):
        with pytest.raises(ValueError, match="cannot have both"):
            Sample(id=0, input="x", rubric="r", rubric_path="/tmp/x.txt")

    def test_neither_set_defaults_none(self):
        s = Sample(id=0, input="x")
        assert s.rubric is None
        assert s.rubric_path is None


# ── Loader: per-sample rubric/rubric_path ──────────────────────────────────


class TestLoaderPerSampleRubric:
    def test_jsonl_inline_rubric(self, tmp_path: Path):
        data = tmp_path / "data.jsonl"
        data.write_text(json.dumps({"input": "Q?", "rubric": "inline rubric"}) + "\n")
        samples = list(load_jsonl(data))
        assert samples[0].rubric == "inline rubric"

    def test_jsonl_rubric_path(self, tmp_path: Path):
        rubric_file = tmp_path / "rubric.txt"
        rubric_file.write_text("file rubric")
        data = tmp_path / "data.jsonl"
        data.write_text(json.dumps({"input": "Q?", "rubric_path": "rubric.txt"}) + "\n")
        samples = list(load_jsonl(data))
        # The loader resolves the file and stores its content in sample.rubric.
        assert samples[0].rubric == "file rubric"

    def test_jsonl_rubric_path_resolves_against_base_dir(self, tmp_path: Path):
        # Dataset in a subdir; rubric file lives next to the suite (base_dir),
        # not next to the dataset. Relative rubric_path must resolve against
        # base_dir — the same anchor every other suite path field uses — so an
        # off-suite dataset (e.g. an HF cache dir) keeps rubric files findable.
        (tmp_path / "rubric.txt").write_text("suite-level rubric")
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir()
        data = datasets_dir / "data.jsonl"
        data.write_text(json.dumps({"input": "Q?", "rubric_path": "rubric.txt"}) + "\n")
        samples = list(load_jsonl(data, base_dir=tmp_path))
        assert samples[0].rubric == "suite-level rubric"

    def test_jsonl_rubric_path_absolute(self, tmp_path: Path):
        rubric_file = tmp_path / "abs.txt"
        rubric_file.write_text("absolute rubric")
        data = tmp_path / "data.jsonl"
        data.write_text(json.dumps({"input": "Q?", "rubric_path": str(rubric_file)}) + "\n")
        samples = list(load_jsonl(data))
        assert samples[0].rubric == "absolute rubric"

    def test_jsonl_both_rubric_and_rubric_path_raises(self, tmp_path: Path):
        data = tmp_path / "data.jsonl"
        data.write_text(json.dumps({"input": "Q?", "rubric": "r", "rubric_path": "x.txt"}) + "\n")
        with pytest.raises(ValueError, match="cannot set both"):
            list(load_jsonl(data))

    def test_jsonl_missing_rubric_path_raises(self, tmp_path: Path):
        data = tmp_path / "data.jsonl"
        data.write_text(json.dumps({"input": "Q?", "rubric_path": "missing.txt"}) + "\n")
        with pytest.raises(ValueError, match="does not exist"):
            list(load_jsonl(data))

    def test_csv_inline_rubric(self, tmp_path: Path):
        data = tmp_path / "data.csv"
        data.write_text('input,rubric\n"Q?","inline csv rubric"\n')
        samples = list(load_csv(data))
        assert samples[0].rubric == "inline csv rubric"

    def test_csv_rubric_path(self, tmp_path: Path):
        rubric_file = tmp_path / "csv_rubric.txt"
        rubric_file.write_text("from csv file")
        data = tmp_path / "data.csv"
        data.write_text("input,rubric_path\n" + '"Q?","csv_rubric.txt"\n')
        samples = list(load_csv(data))
        assert samples[0].rubric == "from csv file"


# ── RubricGrader.grade integration with a mocked OpenAI client ──────────────


def _grader(**spec_kwargs) -> RubricGrader:
    return RubricGrader(ModelJudgeGraderSpec(**spec_kwargs))


def _validation_error() -> ValidationError:
    try:
        ScoreResponse.model_validate({"rationale": "x", "score": 2.5})
    except ValidationError as e:
        return e
    raise AssertionError("expected a ValidationError")


def _openai_reply(parsed) -> MagicMock:
    response = MagicMock()
    response.output_parsed = parsed
    response.status = "completed" if parsed is not None else "incomplete"
    response.usage = MagicMock(model_dump=lambda: {"input_tokens": 1})
    return response


def _patch_openai(grader: RubricGrader, *replies) -> MagicMock:
    """Replace grader.client with a mock whose responses.parse returns each reply in turn.

    A reply is a parsed model, None (no parsable output), or an exception to raise.
    """
    replies = replies or (ScoreResponse(rationale="test", score=0.75),)
    effects = [r if isinstance(r, Exception) else _openai_reply(r) for r in replies]
    mock_client = MagicMock()
    mock_client.responses.parse = AsyncMock(side_effect=effects)
    grader.client = mock_client
    return mock_client


@pytest.mark.asyncio
class TestRubricGraderGrade:
    @pytest.fixture(autouse=True)
    def _openai_key(self, monkeypatch):
        # The constructor needs a key; _patch_openai replaces the client before any call.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    async def test_uses_grader_prompt_when_sample_has_no_rubric(self):
        grader = _grader(prompt="grader-level: {{ input }}/{{ submission }}")
        mock_client = _patch_openai(grader)

        sample = Sample(id=0, input="Q?")
        result, submission = await grader.grade(sample, _trajectory("SUB"))

        assert result.score == 0.75
        call_kwargs = mock_client.responses.parse.call_args.kwargs
        assert call_kwargs["input"] == "grader-level: Q?/SUB"
        assert call_kwargs["text_format"] is ScoreResponse
        # No system prompt by default.
        assert "instructions" not in call_kwargs
        assert call_kwargs["store"] is False

    async def test_per_sample_rubric_overrides_grader_prompt(self):
        grader = _grader(prompt="grader-level: {{ input }}")
        mock_client = _patch_openai(grader)

        sample = Sample(id=0, input="Q?", rubric="per-sample: {{ input }}/{{ submission }}")
        await grader.grade(sample, _trajectory("SUB"))

        sent = mock_client.responses.parse.call_args.kwargs["input"]
        assert sent == "per-sample: Q?/SUB"
        assert "grader-level" not in sent

    async def test_system_prompt_read_from_file(self, tmp_path: Path):
        system_file = tmp_path / "system.txt"
        system_file.write_text("You are a strict judge.")
        grader = _grader(prompt="hi {{ input }}", system_prompt_path=system_file)
        mock_client = _patch_openai(grader)

        await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))

        assert mock_client.responses.parse.call_args.kwargs["instructions"] == "You are a strict judge."

    async def test_call_settings_default_to_provider_defaults(self):
        grader = _grader(prompt="x")
        mock_client = _patch_openai(grader)

        await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))

        call_kwargs = mock_client.responses.parse.call_args.kwargs
        assert "reasoning" not in call_kwargs
        assert call_kwargs["max_output_tokens"] == 16000

    async def test_reasoning_effort_and_max_output_tokens_sent(self):
        grader = _grader(prompt="x", reasoning_effort="low", max_output_tokens=2000)
        mock_client = _patch_openai(grader)

        await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))

        call_kwargs = mock_client.responses.parse.call_args.kwargs
        assert call_kwargs["reasoning"] == {"effort": "low"}
        assert call_kwargs["max_output_tokens"] == 2000

    async def test_no_legacy_wrapper_text_injected(self):
        grader = _grader(prompt="only-this")
        mock_client = _patch_openai(grader)

        await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))

        sent = mock_client.responses.parse.call_args.kwargs["input"]
        # Old wrapper headings should be entirely gone.
        for legacy in (
            "You are an evaluation judge",
            "Original Question",
            "Expected Answer",
            "Agent's Submission",
            "Your Instructions",
            "Do NOT answer the original question yourself",
        ):
            assert legacy not in sent
        assert sent == "only-this"

    async def test_reply_saved_in_metadata(self):
        grader = _grader(prompt="x")
        _patch_openai(grader, ScoreResponse(rationale="looks right", score=1.0))

        result, _ = await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))

        assert result.rationale == "looks right"
        assert result.metadata["response"] == {"rationale": "looks right", "score": 1.0}
        assert result.metadata["rejections"] == []
        assert "error" not in result.metadata

    async def test_invalid_reply_is_asked_again(self):
        grader = _grader(prompt="x")
        mock_client = _patch_openai(grader, _validation_error(), ScoreResponse(rationale="ok", score=0.5))

        result, _ = await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))

        assert result.score == 0.5
        assert len(result.metadata["rejections"]) == 1
        assert mock_client.responses.parse.await_count == 2

    async def test_missing_reply_is_asked_again(self):
        grader = _grader(prompt="x")
        _patch_openai(grader, None, ScoreResponse(rationale="ok", score=0.5))

        result, _ = await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))

        assert result.score == 0.5
        assert len(result.metadata["rejections"]) == 1

    async def test_rejected_on_every_attempt_is_a_grading_error(self):
        grader = _grader(prompt="x")
        mock_client = _patch_openai(grader, *[None] * MAX_ATTEMPTS)

        result, _ = await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))

        assert result.score == 0.0
        assert f"rejected {MAX_ATTEMPTS} times" in result.metadata["error"]
        assert "status=incomplete" in result.metadata["error"]
        assert mock_client.responses.parse.await_count == MAX_ATTEMPTS

    async def test_api_error_is_not_asked_again(self):
        grader = _grader(prompt="x")
        mock_client = _patch_openai(grader, RuntimeError("boom"))

        result, _ = await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))

        assert result.metadata["error"] == "boom"
        assert mock_client.responses.parse.await_count == 1


# ── RubricGrader.grade with a mocked Anthropic client ───────────────────────


def _anthropic_reply(parsed, stop_reason: str = "end_turn") -> MagicMock:
    response = MagicMock()
    response.parsed_output = parsed
    response.stop_reason = stop_reason
    response.usage = MagicMock(
        input_tokens=1, output_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=0
    )
    return response


@pytest.mark.asyncio
class TestRubricGraderAnthropic:
    @pytest.fixture(autouse=True)
    def _anthropic_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    @staticmethod
    def _grader(*replies, **spec_kwargs) -> RubricGrader:
        grader = _grader(model="claude-haiku-5-5", provider=LLMProvider.ANTHROPIC, **{"prompt": "x", **spec_kwargs})
        grader.client = MagicMock()
        grader.client.messages.parse = AsyncMock(side_effect=list(replies))
        return grader

    async def test_call_settings_default_to_provider_defaults(self):
        grader = self._grader(_anthropic_reply(ScoreResponse(rationale="ok", score=1.0)))

        result, _ = await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))

        assert result.score == 1.0
        call_kwargs = grader.client.messages.parse.call_args.kwargs
        assert call_kwargs["output_format"] is ScoreResponse
        assert "output_config" not in call_kwargs
        assert call_kwargs["max_tokens"] == 16000
        assert "system" not in call_kwargs

    async def test_effort_max_tokens_and_system_prompt_sent(self, tmp_path: Path):
        system_file = tmp_path / "system.txt"
        system_file.write_text("You are a strict judge.")
        grader = self._grader(
            _anthropic_reply(ScoreResponse(rationale="ok", score=1.0)),
            reasoning_effort="medium",
            max_output_tokens=3000,
            system_prompt_path=system_file,
        )

        await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))

        call_kwargs = grader.client.messages.parse.call_args.kwargs
        assert call_kwargs["output_config"] == {"effort": "medium"}
        assert call_kwargs["max_tokens"] == 3000
        assert call_kwargs["system"][0]["text"] == "You are a strict judge."

    async def test_refusal_is_asked_again_then_a_grading_error(self):
        grader = self._grader(*[_anthropic_reply(None, stop_reason="refusal") for _ in range(MAX_ATTEMPTS)])

        result, _ = await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))

        assert result.score == 0.0
        assert "stop_reason=refusal" in result.metadata["error"]
        assert grader.client.messages.parse.await_count == MAX_ATTEMPTS


# ── Checklist scorer ────────────────────────────────────────────────────────


def _checklist_sample(*criteria: dict) -> Sample:
    criteria = criteria or (
        {"id": "c1", "text": "Names Paris", "weight": 2},
        {"id": "c2", "text": "Is short"},
        {"id": "leak", "text": "Leaks a secret", "weight": -1},
    )
    return Sample(id=0, input="Q?", rubric_vars={"criteria": list(criteria)})


def _verdicts(**verdicts: str) -> ChecklistResponse:
    return ChecklistResponse(
        criteria=[CriterionVerdict(id=i, rationale=f"why {i}", verdict=v) for i, v in verdicts.items()]
    )


class TestChecklistScorer:
    def test_all_positive_met(self):
        assert checklist(_checklist_sample(), _verdicts(c1="MET", c2="MET", leak="UNMET")) == 1.0

    def test_weighted_share_of_positive_criteria(self):
        assert checklist(_checklist_sample(), _verdicts(c1="MET", c2="UNMET", leak="UNMET")) == pytest.approx(2 / 3)

    def test_negative_criterion_met_subtracts_its_weight(self):
        assert checklist(_checklist_sample(), _verdicts(c1="MET", c2="MET", leak="MET")) == pytest.approx(2 / 3)

    def test_clipped_at_zero(self):
        assert checklist(_checklist_sample(), _verdicts(c1="UNMET", c2="UNMET", leak="MET")) == 0.0

    @pytest.mark.parametrize(
        "verdicts, problem",
        [
            ({"c1": "MET", "c2": "MET"}, "missing ['leak']"),
            ({"c1": "MET", "c2": "MET", "leak": "UNMET", "c9": "MET"}, "unknown ['c9']"),
        ],
    )
    def test_missing_or_unknown_ids_rejected(self, verdicts, problem):
        with pytest.raises(ValueError, match=re.escape(problem)):
            checklist(_checklist_sample(), _verdicts(**verdicts))

    def test_duplicate_ids_rejected(self):
        response = _verdicts(c1="MET", c2="MET", leak="UNMET")
        response.criteria.append(CriterionVerdict(id="c1", rationale="again", verdict="UNMET"))
        with pytest.raises(ValueError, match=re.escape("duplicated ['c1']")):
            checklist(_checklist_sample(), response)

    @pytest.mark.parametrize(
        "rubric_vars, problem",
        [
            (None, "needs rubric_vars"),
            ({"criteria": []}, "needs rubric_vars"),
            ({"criteria": [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}]}, "duplicated"),
            ({"criteria": [{"id": "a", "text": "x", "weight": -1}]}, "positive weight"),
            ({"criteria": [{"id": "a", "text": "x", "wieght": 2}]}, "wieght"),
        ],
    )
    def test_malformed_criteria_rejected(self, rubric_vars, problem):
        with pytest.raises(ValueError, match=problem):
            sample_criteria(Sample(id=0, input="Q?", rubric_vars=rubric_vars))


@pytest.mark.asyncio
class TestChecklistGrader:
    @pytest.fixture(autouse=True)
    def _openai_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    async def test_scores_verdicts_and_saves_them(self):
        grader = _grader(prompt="{% for c in criteria %}[{{ c.id }}] {{ c.text }}\n{% endfor %}", scorer="checklist")
        mock_client = _patch_openai(grader, _verdicts(c1="MET", c2="UNMET", leak="UNMET"))

        result, _ = await grader.grade(_checklist_sample(), _trajectory("SUB"))

        assert result.score == pytest.approx(2 / 3)
        call_kwargs = mock_client.responses.parse.call_args.kwargs
        assert call_kwargs["text_format"] is ChecklistResponse
        # Weights are never shown to the judge.
        assert call_kwargs["input"] == "[c1] Names Paris\n[c2] Is short\n[leak] Leaks a secret\n"
        assert [c["verdict"] for c in result.metadata["response"]["criteria"]] == ["MET", "UNMET", "UNMET"]

    async def test_missing_criterion_is_asked_again(self):
        grader = _grader(prompt="x", scorer="checklist")
        _patch_openai(grader, _verdicts(c1="MET", c2="MET"), _verdicts(c1="MET", c2="MET", leak="UNMET"))

        result, _ = await grader.grade(_checklist_sample(), _trajectory("SUB"))

        assert result.score == 1.0
        assert result.metadata["rejections"] == ["each criterion must be judged exactly once: missing ['leak']"]


@pytest.mark.asyncio
class TestChecklistCriteriaVar:
    @pytest.fixture(autouse=True)
    def _openai_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    async def test_criteria_var_names_the_list(self):
        grader = _grader(prompt="x", scorer="checklist", criteria_var="task_criteria")
        _patch_openai(grader, _verdicts(a="MET", b="UNMET"))
        criteria = [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]

        result, _ = await grader.grade(
            Sample(id=0, input="Q?", rubric_vars={"task_criteria": criteria}), _trajectory("SUB")
        )

        assert result.score == 0.5


# ── Custom response formats and scorers ─────────────────────────────────────

_SCORING_PY = """
from typing import Literal
from pydantic import BaseModel


class Level(BaseModel):
    rationale: str
    level: Literal["CORRECT", "PARTIAL", "WRONG"]


def score_level(sample, response):
    if response.level == "PARTIAL" and sample.ground_truth is None:
        raise ValueError("PARTIAL needs a ground truth")
    return {"CORRECT": 1.0, "PARTIAL": 0.5, "WRONG": 0.0}[response.level]


class Bounded(BaseModel):
    rationale: str
    score: Literal[0, 0.5, 1]


class NoScore(BaseModel):
    rationale: str


class Scores(BaseModel):
    scores: dict[str, float]


class Node(BaseModel):
    name: str
    children: list["Node"]
"""


@pytest.mark.asyncio
class TestCustomResponseFormat:
    @pytest.fixture(autouse=True)
    def _openai_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    @pytest.fixture
    def suite_dir(self, tmp_path: Path) -> Path:
        (tmp_path / "scoring.py").write_text(_SCORING_PY)
        return tmp_path

    async def test_custom_scorer_and_format(self, suite_dir: Path):
        grader = _grader(
            prompt="x", scorer="scoring.py:score_level", response_format="scoring.py:Level", base_dir=suite_dir
        )
        reply = grader.response_format(rationale="half right", level="PARTIAL")
        mock_client = _patch_openai(grader, reply)

        result, _ = await grader.grade(Sample(id=0, input="Q?", ground_truth="A"), _trajectory("SUB"))

        assert result.score == 0.5
        assert result.rationale == "half right"
        assert mock_client.responses.parse.call_args.kwargs["text_format"].__name__ == "Level"

    async def test_scorer_value_error_rejects_the_reply(self, suite_dir: Path):
        grader = _grader(
            prompt="x", scorer="scoring.py:score_level", response_format="scoring.py:Level", base_dir=suite_dir
        )
        partial = grader.response_format(rationale="?", level="PARTIAL")
        wrong = grader.response_format(rationale="no", level="WRONG")
        _patch_openai(grader, partial, wrong)

        result, _ = await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))

        assert result.score == 0.0
        assert len(result.metadata["rejections"]) == 1

    async def test_default_scorer_with_custom_format(self, suite_dir: Path):
        grader = _grader(prompt="x", response_format="scoring.py:Bounded", base_dir=suite_dir)
        _patch_openai(grader, grader.response_format(rationale="ok", score=0.5))

        result, _ = await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))

        assert result.score == 0.5

    @pytest.mark.parametrize(
        "response_format, problem",
        [
            ("scoring.py:NoScore", "needs a score field"),
            ("scoring.py:Scores", "free-form object"),
            ("scoring.py:Node", "recursive"),
        ],
    )
    async def test_unusable_formats_rejected_at_init(self, suite_dir: Path, response_format, problem):
        with pytest.raises(ValueError, match=problem):
            _grader(prompt="x", response_format=response_format, base_dir=suite_dir)


# ── Suite validator: missing rubric_vars detected pre-run ──────────────────


def _minimal_suite(prompt: str, tmp_path: Path) -> SuiteSpec:
    dataset = tmp_path / "d.jsonl"
    dataset.write_text(json.dumps({"input": "Q?"}) + "\n")
    return SuiteSpec(
        name="t",
        description="t",
        dataset=str(dataset),
        target=LettaCodeTargetSpec(model_handles=["openai/gpt-4.1-mini"]),
        graders={
            "g": ModelJudgeGraderSpec(prompt=prompt, model="gpt-4o-mini"),
        },
        reward=MetricRewardSpec(kind="metric", metric_key="g"),
    )


class TestValidateModelJudgesPreRun:
    def test_missing_rubric_var_raises_with_clear_message(self, tmp_path: Path):
        suite = _minimal_suite("rubric needs {{ category }}", tmp_path)
        samples = [Sample(id=0, input="Q?")]
        with pytest.raises(ValueError, match=r"Sample 0, grader 'g'.*category"):
            validate_model_judges(suite, samples)

    def test_sample_provides_rubric_var(self, tmp_path: Path):
        suite = _minimal_suite("rubric needs {{ category }}", tmp_path)
        samples = [Sample(id=0, input="Q?", rubric_vars={"category": "x"})]
        validate_model_judges(suite, samples)

    def test_per_sample_rubric_replaces_grader_rubric(self, tmp_path: Path):
        suite = _minimal_suite("rubric needs {{ category }}", tmp_path)
        samples = [Sample(id=0, input="Q?", rubric="no vars used")]
        validate_model_judges(suite, samples)

    def test_per_sample_rubric_is_checked(self, tmp_path: Path):
        suite = _minimal_suite("{{ input }}", tmp_path)
        samples = [Sample(id="cat", input="Q?", rubric="needs {{ topic }}")]
        with pytest.raises(ValueError, match=r"Sample cat, grader 'g'.*topic"):
            validate_model_judges(suite, samples)

    def test_single_brace_placeholder_stops_the_run(self, tmp_path: Path):
        suite = _minimal_suite("Grade {submission}", tmp_path)
        with pytest.raises(ValueError, match="single-brace"):
            validate_model_judges(suite, [Sample(id=0, input="Q?")])

    def test_syntax_error_stops_the_run(self, tmp_path: Path):
        suite = _minimal_suite("{% if %}", tmp_path)
        with pytest.raises(ValueError, match="Sample 0"):
            validate_model_judges(suite, [Sample(id=0, input="Q?")])

    def test_malformed_checklist_criteria_stop_the_run(self, tmp_path: Path):
        suite = _minimal_suite("{{ input }}", tmp_path)
        suite.graders["g"].scorer = "checklist"
        samples = [Sample(id=7, input="Q?", rubric_vars={"criteria": [{"id": "a"}]})]
        with pytest.raises(ValueError, match=r"(?s)Sample 7, grader 'g'.*text"):
            validate_model_judges(suite, samples)

    def test_wellformed_checklist_criteria_pass(self, tmp_path: Path):
        suite = _minimal_suite("{% for c in criteria %}{{ c.text }}{% endfor %}", tmp_path)
        suite.graders["g"].scorer = "checklist"
        validate_model_judges(suite, [_checklist_sample()])


# ── ModelJudgeGraderSpec: judge call settings ───────────────────────────────


class TestModelJudgeGraderSpecSettings:
    def test_unknown_reasoning_effort_rejected(self):
        with pytest.raises(ValueError, match="reasoning_effort"):
            ModelJudgeGraderSpec(prompt="x", reasoning_effort="hgih")

    def test_max_output_tokens_must_be_positive(self):
        with pytest.raises(ValueError, match="max_output_tokens"):
            ModelJudgeGraderSpec(prompt="x", max_output_tokens=0)

    def test_inline_system_prompt_rejected(self):
        with pytest.raises(ValueError, match="system_prompt"):
            ModelJudgeGraderSpec(prompt="x", system_prompt="You are a judge.")

    def test_missing_system_prompt_file_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="does not exist"):
            ModelJudgeGraderSpec(prompt="x", system_prompt_path=tmp_path / "missing.txt")

    def test_relative_system_prompt_path_resolves_against_suite_dir(self, tmp_path: Path):
        (tmp_path / "system.txt").write_text("framing")
        (tmp_path / "d.jsonl").write_text(json.dumps({"input": "Q?"}) + "\n")
        suite = SuiteSpec.from_yaml(
            {
                "name": "t",
                "dataset": "d.jsonl",
                "target": {"kind": "letta_code", "model_handles": ["openai/gpt-4.1-mini"]},
                "graders": {"g": {"kind": "model_judge", "prompt": "x", "system_prompt_path": "system.txt"}},
                "reward": {"kind": "metric", "metric_key": "g"},
            },
            base_dir=tmp_path,
        )
        assert suite.graders["g"].system_prompt_path == (tmp_path / "system.txt").resolve()

    def test_unknown_builtin_scorer_rejected(self):
        with pytest.raises(ValueError, match="scorer must be"):
            ModelJudgeGraderSpec(prompt="x", scorer="levels")

    def test_custom_scorer_needs_response_format(self):
        with pytest.raises(ValueError, match="needs a response_format"):
            ModelJudgeGraderSpec(prompt="x", scorer="scoring.py:fn")

    def test_checklist_takes_no_response_format(self):
        with pytest.raises(ValueError, match="remove response_format"):
            ModelJudgeGraderSpec(prompt="x", scorer="checklist", response_format="scoring.py:Model")
