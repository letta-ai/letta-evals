"""Tests for the redesigned rubric grader.

Covers:
- ``build_judge_prompt`` template substitution semantics.
- ``Sample.rubric`` / ``Sample.rubric_path`` mutual-exclusion validation.
- The JSONL/CSV loader resolving per-sample rubric overrides.
- ``RubricGrader.grade`` precedence (per-sample rubric > grader prompt),
  the system prompt file, reasoning effort and output-token settings sent
  to each provider, and the absence of legacy wrapper injection in the
  prompt sent to the judge.
- The pre-run validator (``validate_rubric_vars``) catching
  missing ``sample.rubric_vars`` keys before any judge call.
- ``ModelJudgeGraderSpec`` validation of the judge call settings.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from letta_client.types.agents import AssistantMessage

from letta_evals.datasets.loader import load_csv, load_jsonl
from letta_evals.graders.prompt_utils import RESERVED_VARS, build_judge_prompt
from letta_evals.graders.rubric import RubricGrader
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
        out = build_judge_prompt("in={input} gt={ground_truth} sub={submission} k={k}", s, "SUB")
        assert out == "in=Q? gt=A sub=SUB k=v"

    def test_ground_truth_none_becomes_empty_string(self):
        s = Sample(id=0, input="Q?", ground_truth=None)
        out = build_judge_prompt("[{ground_truth}]", s, "x")
        assert out == "[]"

    def test_input_list_is_stringified(self):
        s = Sample(id=0, input=["turn1", "turn2"], ground_truth="A")
        out = build_judge_prompt("{input}", s, "x")
        # str() of a list — verbatim, no special joining
        assert out == "['turn1', 'turn2']"

    def test_missing_var_raises_key_error_with_available_list(self):
        s = Sample(id=0, input="Q?", rubric_vars={"foo": "bar"})
        with pytest.raises(KeyError) as exc:
            build_judge_prompt("hello {missing}", s, "x")
        msg = str(exc.value)
        assert "missing" in msg
        assert "foo" in msg
        for r in RESERVED_VARS:
            assert r in msg

    def test_reserved_collision_raises_value_error(self):
        s = Sample(id=0, input="x", rubric_vars={"input": "shadow"})
        with pytest.raises(ValueError, match="reserved"):
            build_judge_prompt("hi {input}", s, "sub")

    def test_escaped_literal_braces_preserved(self):
        s = Sample(id=0, input="Q?")
        out = build_judge_prompt("answer in {{brackets}} like {input}", s, "x")
        assert out == "answer in {brackets} like Q?"

    def test_no_rubric_vars_only_reserved_available(self):
        s = Sample(id=0, input="Q?")
        out = build_judge_prompt("{input}/{submission}", s, "SUB")
        assert out == "Q?/SUB"

    def test_rubric_vars_None_treated_as_empty(self):
        s = Sample(id=0, input="Q?", rubric_vars=None)
        out = build_judge_prompt("{input}/{submission}", s, "SUB")
        assert out == "Q?/SUB"


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


def _patch_openai(grader: RubricGrader, output: dict | None = None) -> MagicMock:
    """Replace grader.client with a mock Responses API client."""

    mock_response = MagicMock()
    mock_response.output_text = json.dumps(output or {"score": 0.75, "rationale": "test"})
    mock_response.usage = MagicMock(model_dump=lambda: {"input_tokens": 1})
    mock_client = MagicMock()
    mock_client.responses.create = AsyncMock(return_value=mock_response)
    grader.client = mock_client
    return mock_client


@pytest.mark.asyncio
class TestRubricGraderGrade:
    @pytest.fixture(autouse=True)
    def _openai_key(self, monkeypatch):
        # The constructor needs a key; _patch_openai replaces the client before any call.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    async def test_uses_grader_prompt_when_sample_has_no_rubric(self):
        grader = _grader(prompt="grader-level: {input}/{submission}")
        mock_client = _patch_openai(grader)

        sample = Sample(id=0, input="Q?")
        result, submission = await grader.grade(sample, _trajectory("SUB"))

        assert result.score == 0.75
        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert call_kwargs["input"] == "grader-level: Q?/SUB"
        # No system prompt by default.
        assert "instructions" not in call_kwargs
        assert call_kwargs["store"] is False

    async def test_per_sample_rubric_overrides_grader_prompt(self):
        grader = _grader(prompt="grader-level: {input}")
        mock_client = _patch_openai(grader)

        sample = Sample(id=0, input="Q?", rubric="per-sample: {input}/{submission}")
        await grader.grade(sample, _trajectory("SUB"))

        sent = mock_client.responses.create.call_args.kwargs["input"]
        assert sent == "per-sample: Q?/SUB"
        assert "grader-level" not in sent

    async def test_system_prompt_read_from_file(self, tmp_path: Path):
        system_file = tmp_path / "system.txt"
        system_file.write_text("You are a strict judge.")
        grader = _grader(prompt="hi {input}", system_prompt_path=system_file)
        mock_client = _patch_openai(grader)

        await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))

        assert mock_client.responses.create.call_args.kwargs["instructions"] == "You are a strict judge."

    async def test_call_settings_default_to_provider_defaults(self):
        grader = _grader(prompt="x")
        mock_client = _patch_openai(grader)

        await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert "reasoning" not in call_kwargs
        assert call_kwargs["max_output_tokens"] == 16000

    async def test_reasoning_effort_and_max_output_tokens_sent(self):
        grader = _grader(prompt="x", reasoning_effort="low", max_output_tokens=2000)
        mock_client = _patch_openai(grader)

        await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert call_kwargs["reasoning"] == {"effort": "low"}
        assert call_kwargs["max_output_tokens"] == 2000

    async def test_no_legacy_wrapper_text_injected(self):
        grader = _grader(prompt="only-this")
        mock_client = _patch_openai(grader)

        await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))

        sent = mock_client.responses.create.call_args.kwargs["input"]
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

    async def test_response_format_is_json_schema_non_strict(self):
        grader = _grader(prompt="x")
        mock_client = _patch_openai(grader)

        await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))

        fmt = mock_client.responses.create.call_args.kwargs["text"]["format"]
        assert fmt["type"] == "json_schema"
        # We intentionally do NOT pass strict=true: it forbids `minimum` /
        # `maximum` keywords and would require a hand-built schema that
        # duplicates _JudgeResponse. The pydantic-derived schema is sent as
        # guidance, and the score is clamped to [0, 1] Python-side.
        assert fmt["strict"] is False
        schema = fmt["schema"]
        assert schema["type"] == "object"
        assert set(schema["required"]) == {"score", "rationale"}
        # Order matters: the judge reasons before it scores.
        assert list(schema["properties"]) == ["rationale", "score"]

    async def test_score_is_clamped_to_unit_interval(self):
        """Belt-and-suspenders: even though the OpenAI strict schema lets
        unbounded numbers through, the grader clamps the parsed score into
        [0, 1] before returning it."""
        grader = _grader(prompt="x")

        _patch_openai(grader, {"score": 2.5, "rationale": "ok"})
        result, _ = await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))
        assert result.score == 1.0

        _patch_openai(grader, {"score": -0.4, "rationale": "ok"})
        result, _ = await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))
        assert result.score == 0.0


# ── RubricGrader.grade with a mocked Anthropic client ───────────────────────


def _anthropic_response(content: list, stop_reason: str = "end_turn") -> MagicMock:
    response = MagicMock()
    response.content = content
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

    @pytest.fixture
    def grader(self):
        grader = _grader(prompt="x", model="claude-haiku-5-5", provider=LLMProvider.ANTHROPIC)
        grader.client = MagicMock()
        return grader

    @staticmethod
    def _ok_response() -> MagicMock:
        text = MagicMock(type="text", text=json.dumps({"score": 1.0, "rationale": "ok"}))
        return _anthropic_response([text])

    async def test_call_settings_default_to_provider_defaults(self, grader):
        grader.client.messages.create = AsyncMock(return_value=self._ok_response())

        await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))

        call_kwargs = grader.client.messages.create.call_args.kwargs
        assert "effort" not in call_kwargs["output_config"]
        assert call_kwargs["output_config"]["format"]["type"] == "json_schema"
        assert call_kwargs["max_tokens"] == 16000
        assert "system" not in call_kwargs

    async def test_effort_max_tokens_and_system_prompt_sent(self, tmp_path: Path):
        system_file = tmp_path / "system.txt"
        system_file.write_text("You are a strict judge.")
        grader = _grader(
            prompt="x",
            model="claude-haiku-5-5",
            provider=LLMProvider.ANTHROPIC,
            reasoning_effort="medium",
            max_output_tokens=3000,
            system_prompt_path=system_file,
        )
        grader.client = MagicMock()
        grader.client.messages.create = AsyncMock(return_value=self._ok_response())

        await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))

        call_kwargs = grader.client.messages.create.call_args.kwargs
        assert call_kwargs["output_config"]["effort"] == "medium"
        assert call_kwargs["output_config"]["format"]["type"] == "json_schema"
        assert call_kwargs["max_tokens"] == 3000
        assert call_kwargs["system"][0]["text"] == "You are a strict judge."

    async def test_reads_text_after_thinking_block(self, grader):
        thinking = MagicMock(type="thinking", spec=["type", "thinking"])
        text = MagicMock(type="text", text=json.dumps({"score": 0.5, "rationale": "ok"}))
        grader.client.messages.create = AsyncMock(return_value=_anthropic_response([thinking, text]))

        result, _ = await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))

        assert result.score == 0.5
        assert result.rationale == "ok"

    async def test_no_text_block_is_a_grading_error(self, grader):
        grader.client.messages.create = AsyncMock(return_value=_anthropic_response([], stop_reason="refusal"))

        result, _ = await grader.grade(Sample(id=0, input="Q?"), _trajectory("SUB"))

        assert result.score == 0.0
        assert "stop_reason=refusal" in result.metadata["error"]


# ── Suite validator: missing rubric_vars detected pre-run ──────────────────


def _minimal_suite(prompt: str, dataset: Path) -> SuiteSpec:
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


class TestValidateRubricVarsPreRun:
    def test_missing_rubric_var_raises_with_clear_message(self, tmp_path: Path):
        from letta_evals.execution.grading import validate_rubric_vars

        dataset = tmp_path / "d.jsonl"
        dataset.write_text(json.dumps({"input": "Q?"}) + "\n")

        suite = _minimal_suite("rubric needs {category}", dataset)
        samples = [Sample(id=0, input="Q?")]
        with pytest.raises(ValueError, match="missing rubric variables"):
            validate_rubric_vars(suite, samples)

    def test_sample_provides_rubric_var(self, tmp_path: Path):
        from letta_evals.execution.grading import validate_rubric_vars

        dataset = tmp_path / "d.jsonl"
        dataset.write_text(json.dumps({"input": "Q?"}) + "\n")

        suite = _minimal_suite("rubric needs {category}", dataset)
        samples = [Sample(id=0, input="Q?", rubric_vars={"category": "x"})]
        # Should not raise.
        validate_rubric_vars(suite, samples)

    def test_per_sample_rubric_skips_static_validation(self, tmp_path: Path):
        from letta_evals.execution.grading import validate_rubric_vars

        dataset = tmp_path / "d.jsonl"
        dataset.write_text(json.dumps({"input": "Q?"}) + "\n")

        suite = _minimal_suite("rubric needs {category}", dataset)
        # Sample overrides the rubric; pre-run check skips it.
        samples = [Sample(id=0, input="Q?", rubric="no vars used")]
        validate_rubric_vars(suite, samples)


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
