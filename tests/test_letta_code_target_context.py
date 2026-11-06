"""Agent factory context: return forms, layering over sample.extra_vars, the
target's memory_dir and cwd write-back, and the working directory. Runs
LettaCodeTarget.run() against a fake letta CLI that records the env, cwd and
prompt it was launched with."""

import json
import os
from pathlib import Path

import pytest

from letta_evals.models import Sample
from letta_evals.targets.errors import TargetError
from letta_evals.targets.letta_code_target import LettaCodeTarget

FACTORY_AGENT_ID = "agent-factory-1"


def _install_fake_letta(tmp_path, monkeypatch) -> Path:
    """Install a fake `letta` CLI that records MEMORY_DIR/FOO/BAR, cwd and the prompt."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "cli_record.json"
    script = bin_dir / "letta"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "prompt = sys.stdin.read()\n"
        f"with open({str(record)!r}, 'w') as f:\n"
        "    json.dump({k: os.environ.get(k) for k in ('MEMORY_DIR', 'FOO', 'BAR')}"
        " | {'cwd': os.getcwd(), 'prompt': prompt}, f)\n"
        "print(json.dumps({'type': 'system', 'subtype': 'init', 'agent_id': 'agent-from-stream'}))\n"
        "print(json.dumps({'type': 'result', 'usage': {}}))\n"
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return record


def _write_factory(tmp_path, returns: str, ws: str = "") -> str:
    """Write a factory whose body is ``return <returns>``; ``WS`` is in scope."""
    (tmp_path / "factory.py").write_text(
        f"WS = {ws!r}\nasync def factory(client, sample):\n    agent_id = {FACTORY_AGENT_ID!r}\n    return {returns}\n"
    )
    return "factory.py:factory"


def _default_cwd(tmp_path) -> Path:
    return tmp_path / "default-cwd"


def _make_target(tmp_path, **overrides) -> LettaCodeTarget:
    _default_cwd(tmp_path).mkdir(exist_ok=True)
    kwargs = dict(
        client=None, model_handle="test-model", default_cwd=_default_cwd(tmp_path), base_dir=tmp_path, timeout=30
    )
    kwargs.update(overrides)
    return LettaCodeTarget(**kwargs)


@pytest.mark.asyncio
async def test_bare_agent_id_return_context_holds_only_cwd(tmp_path, monkeypatch):
    _install_fake_letta(tmp_path, monkeypatch)
    target = _make_target(tmp_path, agent_script=_write_factory(tmp_path, "agent_id"))

    result = await target.run(Sample(id=0, input="hi"))

    assert result.agent_id == FACTORY_AGENT_ID
    assert result.context == {"cwd": str(_default_cwd(tmp_path).resolve())}


@pytest.mark.asyncio
async def test_tuple_return_context_lands_on_result(tmp_path, monkeypatch):
    _install_fake_letta(tmp_path, monkeypatch)
    target = _make_target(tmp_path, agent_script=_write_factory(tmp_path, "agent_id, {'seed': 3}"))

    result = await target.run(Sample(id=0, input="hi"))

    assert result.agent_id == FACTORY_AGENT_ID
    assert result.context == {"seed": 3, "cwd": str(_default_cwd(tmp_path).resolve())}


@pytest.mark.asyncio
async def test_invalid_factory_return_raises_target_error(tmp_path, monkeypatch):
    _install_fake_letta(tmp_path, monkeypatch)
    target = _make_target(tmp_path, agent_script=_write_factory(tmp_path, "[agent_id]"))

    with pytest.raises(TargetError, match=r"agent_id or \(agent_id, context dict\), got list"):
        await target.run(Sample(id=0, input="hi"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("factory_returns", "target_kwargs", "expected"),
    [
        # factory hands back a seeded workspace
        ("agent_id, {'memory_dir': WS}", {}, "{ws}"),
        # factory injects MEMORY_DIR via env; write-back normalizes to memory_dir
        ("agent_id, {'env': {'MEMORY_DIR': WS}}", {}, "{ws}"),
        # factory returns nothing; the agent's own memory root is used
        ("agent_id", {}, "{home}/.letta/agents/agent-factory-1/memory"),
        # explicit suite-level memory_dir
        ("agent_id", {"memory_dir": "SUITE"}, "{suite}"),
    ],
)
async def test_memory_dir_write_back(tmp_path, monkeypatch, factory_returns, target_kwargs, expected):
    record = _install_fake_letta(tmp_path, monkeypatch)
    ws = str(tmp_path / "seeded")
    suite_dir = tmp_path / "suite-ws"
    if target_kwargs.get("memory_dir") == "SUITE":
        target_kwargs = {**target_kwargs, "memory_dir": suite_dir}
    target = _make_target(
        tmp_path, agent_script=_write_factory(tmp_path, factory_returns, ws), memory_workspace=True, **target_kwargs
    )

    result = await target.run(Sample(id=0, input="hi"))

    expected = expected.format(ws=ws, home=tmp_path / "home", suite=suite_dir)
    assert result.context["memory_dir"] == expected
    # The written-back path is exactly what the subprocess saw.
    cli = json.loads(record.read_text())
    assert cli["MEMORY_DIR"] == expected
    # The memory workspace doesn't move the agent's working directory.
    assert os.path.realpath(cli["cwd"]) == os.path.realpath(_default_cwd(tmp_path))


@pytest.mark.asyncio
async def test_no_factory_explicit_memory_dir_is_written_back(tmp_path, monkeypatch):
    _install_fake_letta(tmp_path, monkeypatch)
    explicit = tmp_path / "explicit-ws"
    target = _make_target(tmp_path, memory_workspace=True, memory_dir=explicit)

    result = await target.run(Sample(id=0, input="hi"))

    assert result.agent_id == "agent-from-stream"
    assert result.context == {"memory_dir": str(explicit), "cwd": str(_default_cwd(tmp_path).resolve())}


@pytest.mark.asyncio
async def test_memory_workspace_off_leaves_context_memory_dir_alone(tmp_path, monkeypatch):
    record = _install_fake_letta(tmp_path, monkeypatch)
    target = _make_target(tmp_path, agent_script=_write_factory(tmp_path, "agent_id, {'memory_dir': '/not/used'}"))

    result = await target.run(Sample(id=0, input="hi"))

    assert result.context == {"memory_dir": "/not/used", "cwd": str(_default_cwd(tmp_path).resolve())}
    cli = json.loads(record.read_text())
    assert cli["MEMORY_DIR"] is None


@pytest.mark.asyncio
async def test_context_layers_over_extra_vars_and_merges_env(tmp_path, monkeypatch):
    record = _install_fake_letta(tmp_path, monkeypatch)
    target = _make_target(
        tmp_path,
        agent_script=_write_factory(
            tmp_path, "agent_id, {'env': {'BAR': 'from-factory'}, 'memory_dir': WS}", str(tmp_path / "from-factory")
        ),
        memory_workspace=True,
    )
    sample = Sample(
        id=0,
        input="hi",
        extra_vars={
            "env": {"FOO": "from-dataset", "BAR": "from-dataset"},
            "memory_dir": str(tmp_path / "from-dataset"),
        },
    )

    result = await target.run(sample)

    cli = json.loads(record.read_text())
    assert cli["FOO"] == "from-dataset"  # dataset env keys survive
    assert cli["BAR"] == "from-factory"  # factory wins per key
    assert result.context["memory_dir"] == str(tmp_path / "from-factory")
    # The shared sample is not mutated by layering.
    assert sample.extra_vars["env"] == {"FOO": "from-dataset", "BAR": "from-dataset"}


@pytest.mark.asyncio
async def test_agent_starts_in_the_default_cwd_and_pwd_names_it(tmp_path, monkeypatch):
    record = _install_fake_letta(tmp_path, monkeypatch)
    target = _make_target(tmp_path)

    result = await target.run(Sample(id=0, input="files are in {pwd}"))

    default = _default_cwd(tmp_path).resolve()
    cli = json.loads(record.read_text())
    assert os.path.realpath(cli["cwd"]) == str(default)
    assert cli["prompt"] == f"files are in {default.as_posix()}"
    assert result.context == {"cwd": str(default)}


@pytest.mark.asyncio
async def test_relative_cwd_from_extra_vars_resolves_against_the_suite_dir(tmp_path, monkeypatch):
    record = _install_fake_letta(tmp_path, monkeypatch)
    (tmp_path / "data").mkdir()
    target = _make_target(tmp_path)

    result = await target.run(Sample(id=0, input="files are in {pwd}", extra_vars={"cwd": "data"}))

    data = (tmp_path / "data").resolve()
    cli = json.loads(record.read_text())
    assert os.path.realpath(cli["cwd"]) == str(data)
    assert cli["prompt"] == f"files are in {data.as_posix()}"
    assert result.context == {"cwd": str(data)}


@pytest.mark.asyncio
async def test_factory_cwd_wins_over_extra_vars(tmp_path, monkeypatch):
    record = _install_fake_letta(tmp_path, monkeypatch)
    staged = tmp_path / "staged-task"
    staged.mkdir()
    (tmp_path / "data").mkdir()
    target = _make_target(tmp_path, agent_script=_write_factory(tmp_path, "agent_id, {'cwd': WS}", str(staged)))

    result = await target.run(Sample(id=0, input="hi", extra_vars={"cwd": "data"}))

    cli = json.loads(record.read_text())
    assert os.path.realpath(cli["cwd"]) == str(staged.resolve())
    assert result.context == {"cwd": str(staged.resolve())}


@pytest.mark.asyncio
async def test_missing_cwd_fails_the_rollout(tmp_path, monkeypatch):
    record = _install_fake_letta(tmp_path, monkeypatch)
    target = _make_target(tmp_path)

    with pytest.raises(TargetError, match=r"cwd .*missing for sample 0 is not a directory"):
        await target.run(Sample(id=0, input="hi", extra_vars={"cwd": "missing"}))

    assert not record.exists()
