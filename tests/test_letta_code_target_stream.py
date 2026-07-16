"""Stream-json handling: the drain-only LettaCodeTarget stdout reader and
final-line usage extraction (issue #319)."""

import logging
import os

import pytest

from letta_evals.execution.trace import extract_usage_stats
from letta_evals.models import Sample
from letta_evals.targets.errors import TargetError
from letta_evals.targets.letta_code_target import LettaCodeTarget

# --- extract_usage_stats -----------------------------------------------------

RESULT_LINE = '{"type": "result", "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}'


def test_extract_usage_from_result_line():
    stats = extract_usage_stats(RESULT_LINE)
    assert stats == [
        {
            "message_type": "usage_statistics",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cached_input_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
        }
    ]


def test_extract_usage_empty_line_is_none():
    assert extract_usage_stats("") is None


def test_extract_usage_non_result_event_is_none():
    assert extract_usage_stats('{"type": "message", "content": "hi"}') is None
    assert extract_usage_stats("42") is None


def test_extract_usage_malformed_line_warns_and_is_none(caplog):
    # e.g. a result event glued onto a torn tool-return line
    with caplog.at_level(logging.WARNING):
        assert extract_usage_stats('{"type": "message"}' + RESULT_LINE) is None
    assert "Unparseable final stream line" in caplog.text
    assert "Extra data" in caplog.text


# --- run() end-to-end against a fake letta CLI -------------------------------

INIT_LINE_B = b'{"type": "system", "subtype": "init", "agent_id": "agent-stream-test"}'
RESULT_LINE_B = RESULT_LINE.encode()
TOOL_RETURN_LINE_B = b'{"type": "message", "message_type": "tool_return_message", "content": "' + b"x" * 30000 + b'"}'


def _install_fake_letta(tmp_path, monkeypatch, stdout_payload: bytes, returncode: int = 0) -> None:
    """Install a fake `letta` CLI on PATH that consumes stdin, emits a fixed
    stdout payload byte-for-byte, and exits with the given code."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    payload = bin_dir / "stdout_payload"
    payload.write_bytes(stdout_payload)
    script = bin_dir / "letta"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "sys.stdin.read()\n"
        f"sys.stdout.buffer.write(pathlib.Path({str(payload)!r}).read_bytes())\n"
        f"sys.exit({returncode})\n"
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")


def _make_target(tmp_path) -> LettaCodeTarget:
    # client is only used by the agent_script factory path, never here.
    return LettaCodeTarget(client=None, model_handle="test-model", base_dir=tmp_path, timeout=30)


@pytest.mark.asyncio
async def test_run_ignores_malformed_mid_stream_lines(tmp_path, monkeypatch):
    # A torn tool_return line (two records on one line) between init and
    # result is drained without parsing: no effect on the run.
    payload = b"\n".join([INIT_LINE_B, TOOL_RETURN_LINE_B + TOOL_RETURN_LINE_B, RESULT_LINE_B, b""])
    _install_fake_letta(tmp_path, monkeypatch, payload)

    result = await _make_target(tmp_path).run(Sample(id=0, input="hello"))

    assert result.agent_id == "agent-stream-test"
    assert result.agent_usage[0]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_run_glued_result_line_warns_and_yields_no_usage(tmp_path, monkeypatch, caplog):
    # If the writer glues the result event onto the preceding tool_return
    # line, usage is lost (accepted tradeoff) but the run still succeeds.
    payload = b"\n".join([INIT_LINE_B, TOOL_RETURN_LINE_B + RESULT_LINE_B, b""])
    _install_fake_letta(tmp_path, monkeypatch, payload)

    with caplog.at_level(logging.WARNING):
        result = await _make_target(tmp_path).run(Sample(id=0, input="hello"))

    assert result.agent_id == "agent-stream-test"
    assert result.agent_usage is None
    assert "Unparseable final stream line" in caplog.text


@pytest.mark.asyncio
async def test_run_survives_invalid_utf8_in_stream(tmp_path, monkeypatch):
    payload = b"\n".join([INIT_LINE_B, b'{"type": "message", "content": "\xff\xfe garbage"}', RESULT_LINE_B, b""])
    _install_fake_letta(tmp_path, monkeypatch, payload)

    result = await _make_target(tmp_path).run(Sample(id=0, input="hello"))

    assert result.agent_id == "agent-stream-test"
    assert result.agent_usage[0]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_run_missing_result_event_yields_no_usage(tmp_path, monkeypatch):
    # A stream cut short of the result event reports usage as None.
    payload = b"\n".join([INIT_LINE_B, TOOL_RETURN_LINE_B, b""])
    _install_fake_letta(tmp_path, monkeypatch, payload)

    result = await _make_target(tmp_path).run(Sample(id=0, input="hello"))

    assert result.agent_id == "agent-stream-test"
    assert result.agent_usage is None


@pytest.mark.asyncio
async def test_run_nonzero_exit_surfaces_last_stdout_line(tmp_path, monkeypatch):
    payload = b"\n".join([INIT_LINE_B, b'{"type": "error", "message": "boom"}', b""])
    _install_fake_letta(tmp_path, monkeypatch, payload, returncode=1)

    with pytest.raises(TargetError) as exc_info:
        await _make_target(tmp_path).run(Sample(id=0, input="hello"))

    assert "return code 1" in str(exc_info.value)
    assert "boom" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_without_init_event_fails_with_no_agent_id(tmp_path, monkeypatch):
    payload = b"\n".join([TOOL_RETURN_LINE_B, RESULT_LINE_B, b""])
    _install_fake_letta(tmp_path, monkeypatch, payload)

    with pytest.raises(TargetError, match="No agent_id found"):
        await _make_target(tmp_path).run(Sample(id=0, input="hello"))
