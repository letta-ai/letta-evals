"""Stream-json handling: the parse_stream_line salvage helper and the
drain-only LettaCodeTarget stdout reader (issue #319)."""

import logging
import os

import pytest

from letta_evals.execution.trace import parse_stream_line
from letta_evals.models import Sample
from letta_evals.targets.errors import TargetError
from letta_evals.targets.letta_code_target import LettaCodeTarget

# --- parse_stream_line -------------------------------------------------------


def test_parse_single_object():
    assert parse_stream_line('{"type": "result"}') == [{"type": "result"}]


def test_parse_empty_line():
    assert parse_stream_line("") == []


def test_parse_concatenated_records_salvages_all(caplog):
    # Torn writer output: two records glued onto one line ("Extra data").
    line = '{"a": 1}{"b": 2}'
    with caplog.at_level(logging.WARNING):
        events = parse_stream_line(line)
    assert events == [{"a": 1}, {"b": 2}]
    assert "Extra data" in caplog.text
    assert "salvaged 2 event(s)" in caplog.text


def test_parse_concatenated_records_with_whitespace_between():
    assert parse_stream_line('{"a": 1} \t {"b": 2}') == [{"a": 1}, {"b": 2}]


def test_parse_truncated_line_returns_empty(caplog):
    line = '{"type": "message", "content": "abc'
    with caplog.at_level(logging.WARNING):
        events = parse_stream_line(line)
    assert events == []
    assert "Unterminated string" in caplog.text


def test_parse_object_followed_by_truncated_record():
    # First record intact, second torn off mid-payload: salvage the first.
    assert parse_stream_line('{"a": 1}{"b": "tru') == [{"a": 1}]


def test_parse_control_character_payload(caplog):
    line = '{"content": "a\x01b"}'
    with caplog.at_level(logging.WARNING):
        assert parse_stream_line(line) == []
    assert "Invalid control character" in caplog.text


def test_parse_non_object_json_ignored():
    assert parse_stream_line("42") == []
    assert parse_stream_line("[1, 2]") == []


# --- run() end-to-end against a fake letta CLI -------------------------------

INIT_LINE = b'{"type": "system", "subtype": "init", "agent_id": "agent-stream-test"}'
RESULT_LINE = b'{"type": "result", "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}'
TOOL_RETURN_LINE = b'{"type": "message", "message_type": "tool_return_message", "content": "' + b"x" * 30000 + b'"}'


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
async def test_run_ignores_malformed_mid_stream_lines(tmp_path, monkeypatch, caplog):
    # A torn tool_return line (two records on one line) between init and
    # result is drained without parsing: no warning, no effect on the run.
    payload = b"\n".join([INIT_LINE, TOOL_RETURN_LINE + TOOL_RETURN_LINE, RESULT_LINE, b""])
    _install_fake_letta(tmp_path, monkeypatch, payload)

    with caplog.at_level(logging.WARNING):
        result = await _make_target(tmp_path).run(Sample(id=0, input="hello"))

    assert result.agent_id == "agent-stream-test"
    assert result.agent_usage == [
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
    assert "Malformed stream-json line" not in caplog.text


@pytest.mark.asyncio
async def test_run_salvages_result_event_glued_to_tool_return(tmp_path, monkeypatch, caplog):
    # If the writer glues the final result event onto the preceding
    # tool_return line, raw_decode salvage still recovers the usage stats.
    payload = b"\n".join([INIT_LINE, TOOL_RETURN_LINE + RESULT_LINE, b""])
    _install_fake_letta(tmp_path, monkeypatch, payload)

    with caplog.at_level(logging.WARNING):
        result = await _make_target(tmp_path).run(Sample(id=0, input="hello"))

    assert result.agent_id == "agent-stream-test"
    assert result.agent_usage is not None
    assert result.agent_usage[0]["total_tokens"] == 15
    assert "salvaged 2 event(s)" in caplog.text


@pytest.mark.asyncio
async def test_run_survives_invalid_utf8_in_stream(tmp_path, monkeypatch):
    payload = b"\n".join([INIT_LINE, b'{"type": "message", "content": "\xff\xfe garbage"}', RESULT_LINE, b""])
    _install_fake_letta(tmp_path, monkeypatch, payload)

    result = await _make_target(tmp_path).run(Sample(id=0, input="hello"))

    assert result.agent_id == "agent-stream-test"
    assert result.agent_usage[0]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_run_missing_result_event_yields_no_usage(tmp_path, monkeypatch):
    # A stream cut short of the result event reports usage as None.
    payload = b"\n".join([INIT_LINE, TOOL_RETURN_LINE, b""])
    _install_fake_letta(tmp_path, monkeypatch, payload)

    result = await _make_target(tmp_path).run(Sample(id=0, input="hello"))

    assert result.agent_id == "agent-stream-test"
    assert result.agent_usage is None


@pytest.mark.asyncio
async def test_run_nonzero_exit_surfaces_last_stdout_line(tmp_path, monkeypatch):
    payload = b"\n".join([INIT_LINE, b'{"type": "error", "message": "boom"}', b""])
    _install_fake_letta(tmp_path, monkeypatch, payload, returncode=1)

    with pytest.raises(TargetError) as exc_info:
        await _make_target(tmp_path).run(Sample(id=0, input="hello"))

    assert "return code 1" in str(exc_info.value)
    assert "boom" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_without_init_event_fails_with_no_agent_id(tmp_path, monkeypatch):
    payload = b"\n".join([TOOL_RETURN_LINE, RESULT_LINE, b""])
    _install_fake_letta(tmp_path, monkeypatch, payload)

    with pytest.raises(TargetError, match="No agent_id found"):
        await _make_target(tmp_path).run(Sample(id=0, input="hello"))
