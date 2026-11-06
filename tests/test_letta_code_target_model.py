"""use_agent_model: a factory-created agent runs on its own model settings, so
the target skips --model. Runs LettaCodeTarget.run() against a fake letta CLI
that records its argv."""

import json
import os

import pytest

from letta_evals.models import LettaCodeTargetSpec, Sample
from letta_evals.targets.letta_code_target import LettaCodeTarget

HANDLE = "opus-5.5-max"


def _install_fake_letta(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "argv.json"
    script = bin_dir / "letta"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "sys.stdin.read()\n"
        f"json.dump(sys.argv[1:], open({str(record)!r}, 'w'))\n"
        "print(json.dumps({'type': 'system', 'subtype': 'init', 'agent_id': 'agent-from-stream'}))\n"
        "print(json.dumps({'type': 'result', 'usage': {}}))\n"
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    return record


def _make_target(tmp_path, *, factory: bool, use_agent_model: bool) -> LettaCodeTarget:
    agent_script = None
    if factory:
        (tmp_path / "factory.py").write_text("async def factory(client, sample):\n    return 'agent-factory-1'\n")
        agent_script = "factory.py:factory"
    return LettaCodeTarget(
        client=None,
        model_handle=HANDLE,
        default_cwd=tmp_path,
        base_dir=tmp_path,
        timeout=30,
        agent_script=agent_script,
        use_agent_model=use_agent_model,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("factory", "use_agent_model", "passes_model"),
    [
        (True, True, False),  # the factory owns the model and its settings
        (True, False, True),  # default: the suite's handle is applied
        (False, True, True),  # Letta Code creates the agent from the handle
    ],
)
async def test_model_flag(tmp_path, monkeypatch, factory, use_agent_model, passes_model):
    record = _install_fake_letta(tmp_path, monkeypatch)
    target = _make_target(tmp_path, factory=factory, use_agent_model=use_agent_model)

    result = await target.run(Sample(id=0, input="hi"))

    argv = json.loads(record.read_text())
    if passes_model:
        assert argv[argv.index("--model") + 1] == HANDLE
    else:
        assert "--model" not in argv
    # Results are still labeled with the suite's handle.
    assert result.model_handle == HANDLE


def test_spec_requires_a_factory():
    with pytest.raises(ValueError, match="use_agent_model requires agent_script"):
        LettaCodeTargetSpec(model_handles=[HANDLE], use_agent_model=True)


def test_spec_allows_one_model_handle():
    with pytest.raises(ValueError, match="use_agent_model allows one model handle"):
        LettaCodeTargetSpec(agent_script="f.py:f", model_handles=[HANDLE, "sonnet-5.5"], use_agent_model=True)

    spec = LettaCodeTargetSpec(agent_script="f.py:f", model_handles=[HANDLE], use_agent_model=True)
    assert spec.use_agent_model is True
