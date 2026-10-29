"""backend: local — LettaCodeTarget against a fake `letta` CLI that implements
`agents create --body -` and a headless run, recording what it was given."""

import json
import os
from pathlib import Path

import pytest
from letta_client.types import ToolReturnMessage
from letta_client.types.agents import AssistantMessage, SummaryMessage, ToolCallMessage, UserMessage

from letta_evals.models import Sample, TargetTrace
from letta_evals.sandbox.dispatch import SandboxMount, build_sandbox_command
from letta_evals.targets.errors import TargetError
from letta_evals.targets.letta_code_target import LettaCodeTarget

AGENT_ID = "agent-local-test"
RECORDED_ENV = (
    "LETTA_LOCAL_BACKEND_DIR",
    "LETTA_BASE_URL",
    "LETTA_DISABLE_SESSION_PERSIST",
    "MEMORY_DIR",
    "LETTA_MEMORY_DIR_EXPLICIT",
    "FOO",
)

STREAM = [
    {"type": "system", "subtype": "init", "agent_id": AGENT_ID},
    {"type": "message", "message_type": "usage_statistics", "prompt_tokens": 1, "run_id": "local-run-1"},
    {
        "type": "message",
        "message_type": "tool_call_message",
        "id": "tool-call-1",
        "date": "2026-01-01T00:00:00Z",
        "tool_call": {"name": "Bash", "tool_call_id": "call_1", "arguments": "{}"},
        "session_id": AGENT_ID,
        "uuid": "u1",
    },
    {
        "type": "message",
        "message_type": "tool_return_message",
        "id": "tool-return-1",
        "date": "2026-01-01T00:00:01Z",
        "status": "success",
        "tool_call_id": "call_1",
        "tool_return": "ok",
    },
    {
        "type": "message",
        "message_type": "summary_message",
        "id": "summary-1",
        "date": "2026-01-01T00:00:02Z",
        "summary": "earlier work",
        "run_id": "local-run-2",
    },
    {
        "type": "message",
        "message_type": "assistant_message",
        "id": "assistant-1",
        "date": "2026-01-01T00:00:03Z",
        "content": [{"type": "text", "text": "done"}],
        "run_id": "local-run-2",
    },
    {"type": "message", "message_type": "stop_reason", "stop_reason": "end_turn"},
    {"type": "result", "subtype": "success", "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
]

FAKE_LETTA = """#!/usr/bin/env python3
import json, os, sys
record_dir, fail = {record_dir!r}, {fail!r}
argv = sys.argv[1:]
stdin = sys.stdin.read()
env = {{k: os.environ.get(k) for k in {env_keys!r}}}
if argv[:4] == ["--backend", "local", "agents", "create"]:
    json.dump({{"argv": argv, "body": json.loads(stdin), "env": env}}, open(os.path.join(record_dir, "create.json"), "w"))
    print(json.dumps({{"id": {agent_id!r}, "name": "n", "tags": []}}))
    sys.exit(0)
json.dump({{"argv": argv, "stdin": stdin, "env": env, "cwd": os.getcwd()}}, open(os.path.join(record_dir, "run.json"), "w"))
for event in {stream!r}[: 4 if fail else None]:
    print(json.dumps(event))
sys.exit(1 if fail else 0)
"""


def _install_fake_letta(tmp_path: Path, monkeypatch, *, fail: bool = False) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record_dir = tmp_path / "record"
    record_dir.mkdir()
    script = bin_dir / "letta"
    script.write_text(
        FAKE_LETTA.format(
            record_dir=str(record_dir), fail=fail, env_keys=RECORDED_ENV, agent_id=AGENT_ID, stream=STREAM
        )
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return record_dir


def _write_factory(tmp_path: Path, body: str) -> str:
    (tmp_path / "factory.py").write_text(f"async def factory(client, sample):\n{body}\n")
    return "factory.py:factory"


CREATING_FACTORY = """    agent = await client.agents.create(name="reflector", system="sys", tags=["parent:p"], model="server/model",
                                       model_settings={"max_tokens": 64}, context_window_limit=1024)
    return agent.id, {"env": {"FOO": "bar"}}"""


def _make_target(tmp_path: Path, **overrides) -> LettaCodeTarget:
    store = tmp_path / "store"
    store.mkdir(exist_ok=True)
    kwargs = dict(
        client=None,
        model_handle="llama.cpp/local-model",
        base_dir=tmp_path,
        timeout=30,
        local_backend_dir=store,
        base_url="http://server.invalid",
    )
    kwargs.update(overrides)
    return LettaCodeTarget(**kwargs)


def _read(record_dir: Path, name: str) -> dict:
    return json.loads((record_dir / name).read_text())


@pytest.mark.asyncio
async def test_factory_agent_is_created_in_the_sample_store_and_resumed(tmp_path, monkeypatch):
    record_dir = _install_fake_letta(tmp_path, monkeypatch)
    target = _make_target(tmp_path, agent_script=_write_factory(tmp_path, CREATING_FACTORY))

    result = await target.run(Sample(id=0, input="reflect"))

    create = _read(record_dir, "create.json")
    run = _read(record_dir, "run.json")
    # The suite's model handle replaces the factory's server model.
    assert create["body"] == {
        "name": "reflector",
        "system": "sys",
        "tags": ["parent:p"],
        "model": "llama.cpp/local-model",
        "model_settings": {"parallel_tool_calls": True, "max_tokens": 64},
        "context_window_limit": 1024,
    }
    assert run["argv"][:2] == ["--backend", "local"]
    assert run["argv"][run["argv"].index("--agent") + 1] == AGENT_ID
    assert "--model" not in run["argv"]
    assert run["stdin"] == "reflect"
    store = str(target.local_backend_dir)
    assert create["env"]["LETTA_LOCAL_BACKEND_DIR"] == run["env"]["LETTA_LOCAL_BACKEND_DIR"] == store
    assert run["env"]["LETTA_BASE_URL"] is None
    assert run["env"]["LETTA_DISABLE_SESSION_PERSIST"] == "1"
    assert run["env"]["FOO"] == "bar"
    assert isinstance(result, TargetTrace)
    assert result.agent_id == AGENT_ID
    assert result.context == {"env": {"FOO": "bar"}}
    assert result.agent_usage[0]["prompt_tokens"] == 7


@pytest.mark.asyncio
async def test_trajectory_comes_from_the_stream(tmp_path, monkeypatch):
    _install_fake_letta(tmp_path, monkeypatch)

    result = await _make_target(tmp_path).run(Sample(id=0, input="hi"))

    [turn] = result.trajectory
    assert [type(m) for m in turn] == [
        UserMessage,
        ToolCallMessage,
        ToolReturnMessage,
        SummaryMessage,
        AssistantMessage,
    ]
    assert turn[0].content == "hi"
    assert turn[3].summary == "earlier work"
    # Tool events carry no run_id; they group under the latest one.
    assert turn[1].step_id == turn[2].step_id == "step-local-run-1"
    assert turn[4].step_id == "step-local-run-2"


@pytest.mark.asyncio
async def test_without_a_factory_letta_code_creates_the_agent(tmp_path, monkeypatch):
    record_dir = _install_fake_letta(tmp_path, monkeypatch)

    result = await _make_target(tmp_path).run(Sample(id=0, input="hi"))

    run = _read(record_dir, "run.json")
    assert "--new-agent" in run["argv"]
    assert run["argv"][run["argv"].index("--model") + 1] == "llama.cpp/local-model"
    assert result.agent_id == AGENT_ID
    assert not (record_dir / "create.json").exists()


@pytest.mark.asyncio
async def test_memory_workspace_defaults_to_the_local_agent_memory(tmp_path, monkeypatch):
    record_dir = _install_fake_letta(tmp_path, monkeypatch)
    target = _make_target(tmp_path, agent_script=_write_factory(tmp_path, CREATING_FACTORY), memory_workspace=True)

    result = await target.run(Sample(id=0, input="hi"))

    expected = target.local_backend_dir / "memfs" / AGENT_ID / "memory"
    run = _read(record_dir, "run.json")
    assert result.context["memory_dir"] == str(expected)
    assert run["env"]["MEMORY_DIR"] == str(expected)
    assert run["env"]["LETTA_MEMORY_DIR_EXPLICIT"] == "1"
    assert Path(run["cwd"]).resolve() == expected.resolve()


@pytest.mark.asyncio
async def test_factory_server_calls_are_rejected(tmp_path, monkeypatch):
    _install_fake_letta(tmp_path, monkeypatch)
    factory = _write_factory(tmp_path, "    await client.blocks.create(label='x')\n    return 'a'")

    with pytest.raises(TargetError, match=r"only call client.agents.create \(got client.blocks\)"):
        await _make_target(tmp_path, agent_script=factory).run(Sample(id=0, input="hi"))


@pytest.mark.asyncio
async def test_failed_run_keeps_the_partial_trajectory(tmp_path, monkeypatch):
    _install_fake_letta(tmp_path, monkeypatch, fail=True)

    with pytest.raises(TargetError) as excinfo:
        await _make_target(tmp_path).run(Sample(id=0, input="hi"))

    [turn] = excinfo.value.trajectory
    assert [m.message_type for m in turn] == ["user_message", "tool_call_message", "tool_return_message"]
    assert excinfo.value.agent_id == AGENT_ID


def test_sandbox_command_carries_the_backend():
    mount = SandboxMount(
        local_root=Path("/x"), remote_root="/mnt/suite", pythonpath=None, suite_yaml_remote="/mnt/suite/s.yaml"
    )

    assert "--backend local" in build_sandbox_command(mount, "m", "local")
    assert "--backend" not in build_sandbox_command(mount, "m", "api")
