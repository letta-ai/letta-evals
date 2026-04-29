"""Unit tests for LettaCodeTarget._build_subprocess_env per-sample env overrides."""

import os
from pathlib import Path

import pytest

from letta_evals.models import Sample
from letta_evals.targets.base import TargetError
from letta_evals.targets.letta_code_target import LettaCodeTarget


def _make_target(tmp_path, **overrides) -> LettaCodeTarget:
    """Build a LettaCodeTarget with a sandboxed working_dir and a stub client."""
    kwargs = dict(
        client=None,  # _build_subprocess_env never touches the client
        model_handle="anthropic/claude-sonnet-4-5-20250929",
        working_dir=tmp_path,
        sandbox=False,
    )
    kwargs.update(overrides)
    return LettaCodeTarget(**kwargs)


def _make_sample(extra_vars=None, sample_id: int = 0) -> Sample:
    return Sample(id=sample_id, input="hello", extra_vars=extra_vars)


def test_build_env_inherits_os_environ_when_no_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("EXISTING_VAR", "from-os")
    # Make sure target-managed vars are not present in os.environ for this test
    monkeypatch.delenv("LETTA_BASE_URL", raising=False)
    monkeypatch.delenv("MEMORY_DIR", raising=False)

    target = _make_target(tmp_path)
    sample = _make_sample()

    env = target._build_subprocess_env(sample, agent_id=None)

    assert env["EXISTING_VAR"] == "from-os"
    assert "LETTA_BASE_URL" not in env  # no base_url configured
    assert "MEMORY_DIR" not in env  # no memory permission mode


def test_build_env_sets_base_url_when_configured(tmp_path):
    target = _make_target(tmp_path, base_url="https://api.example.com")
    sample = _make_sample()

    env = target._build_subprocess_env(sample, agent_id=None)

    assert env["LETTA_BASE_URL"] == "https://api.example.com"


def test_build_env_sets_memory_dir_in_memory_permission_mode(tmp_path, monkeypatch):
    # Redirect HOME so the test doesn't pollute the real ~/.letta
    monkeypatch.setenv("HOME", str(tmp_path))

    target = _make_target(tmp_path, permission_mode="memory")
    sample = _make_sample()

    env = target._build_subprocess_env(sample, agent_id="agent-abc")

    expected = str(Path(tmp_path) / ".letta" / "agents" / "agent-abc" / "memory")
    assert env["MEMORY_DIR"] == expected
    # mkdir(parents=True, exist_ok=True) should have created it
    assert Path(expected).is_dir()


def test_build_env_skips_memory_dir_without_agent_id(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMORY_DIR", raising=False)

    target = _make_target(tmp_path, permission_mode="memory")
    sample = _make_sample()

    env = target._build_subprocess_env(sample, agent_id=None)

    assert "MEMORY_DIR" not in env


def test_build_env_applies_per_sample_overrides(tmp_path):
    target = _make_target(tmp_path)
    sample = _make_sample(extra_vars={"env": {"LETTA_PARENT_AGENT_ID": "agent-parent-123", "FOO": "bar"}})

    env = target._build_subprocess_env(sample, agent_id="agent-child")

    assert env["LETTA_PARENT_AGENT_ID"] == "agent-parent-123"
    assert env["FOO"] == "bar"


def test_build_env_per_sample_overrides_win_over_target_managed(tmp_path):
    target = _make_target(tmp_path, base_url="https://api.example.com")
    sample = _make_sample(extra_vars={"env": {"LETTA_BASE_URL": "https://override.example.com"}})

    env = target._build_subprocess_env(sample, agent_id=None)

    assert env["LETTA_BASE_URL"] == "https://override.example.com"


def test_build_env_per_sample_overrides_win_over_os_environ(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_VAR", "from-os")

    target = _make_target(tmp_path)
    sample = _make_sample(extra_vars={"env": {"MY_VAR": "from-sample"}})

    env = target._build_subprocess_env(sample, agent_id=None)

    assert env["MY_VAR"] == "from-sample"


def test_build_env_coerces_non_string_values_to_str(tmp_path):
    target = _make_target(tmp_path)
    sample = _make_sample(extra_vars={"env": {"INT_VAR": 42, "FLOAT_VAR": 3.14, "BOOL_VAR": True}})

    env = target._build_subprocess_env(sample, agent_id=None)

    assert env["INT_VAR"] == "42"
    assert env["FLOAT_VAR"] == "3.14"
    assert env["BOOL_VAR"] == "True"


def test_build_env_none_value_becomes_empty_string(tmp_path):
    target = _make_target(tmp_path)
    sample = _make_sample(extra_vars={"env": {"NULL_VAR": None}})

    env = target._build_subprocess_env(sample, agent_id=None)

    assert env["NULL_VAR"] == ""


def test_build_env_empty_extra_vars_is_noop(tmp_path):
    target = _make_target(tmp_path)
    sample = _make_sample(extra_vars={})

    env = target._build_subprocess_env(sample, agent_id=None)

    # No crash; env still inherits os.environ
    assert isinstance(env, dict)


def test_build_env_empty_env_dict_is_noop(tmp_path):
    target = _make_target(tmp_path)
    sample = _make_sample(extra_vars={"env": {}})

    env = target._build_subprocess_env(sample, agent_id=None)

    # No crash; nothing added
    assert isinstance(env, dict)


def test_build_env_other_extra_vars_keys_ignored(tmp_path):
    """Keys other than 'env' in extra_vars must not affect the subprocess env."""
    target = _make_target(tmp_path)
    sample = _make_sample(extra_vars={"rubric_var_x": "irrelevant", "env": {"REAL_VAR": "real"}})

    env = target._build_subprocess_env(sample, agent_id=None)

    assert env["REAL_VAR"] == "real"
    assert "rubric_var_x" not in env


def test_build_env_rejects_non_dict_env(tmp_path):
    target = _make_target(tmp_path)
    sample = _make_sample(extra_vars={"env": "FOO=bar"})  # str, not dict

    with pytest.raises(TargetError) as exc_info:
        target._build_subprocess_env(sample, agent_id="agent-x")

    assert "must be a dict" in str(exc_info.value)
    assert exc_info.value.agent_id == "agent-x"


def test_build_env_rejects_non_string_keys(tmp_path):
    target = _make_target(tmp_path)
    sample = _make_sample(extra_vars={"env": {42: "value"}})  # type: ignore[dict-item]

    with pytest.raises(TargetError) as exc_info:
        target._build_subprocess_env(sample, agent_id=None)

    assert "keys must be strings" in str(exc_info.value)


def test_build_env_does_not_mutate_os_environ(tmp_path):
    target = _make_target(tmp_path)
    sample = _make_sample(extra_vars={"env": {"NEW_TEST_VAR": "x"}})

    target._build_subprocess_env(sample, agent_id=None)

    # The injected key must NOT have leaked into the process's actual environment.
    assert "NEW_TEST_VAR" not in os.environ
