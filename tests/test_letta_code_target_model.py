"""use_agent_model: a factory-created agent runs on its own model settings, so
the target skips --model. Runs LettaCodeTarget.run() against the fake letta CLI
from the factory-context tests."""

import json

import pytest

from letta_evals.models import LettaCodeTargetSpec, Sample
from tests.test_letta_code_target_context import _install_fake_letta, _make_target, _write_factory

HANDLE = "opus-5.5-max"


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
    agent_script = _write_factory(tmp_path, "agent_id") if factory else None
    target = _make_target(tmp_path, model_handle=HANDLE, agent_script=agent_script, use_agent_model=use_agent_model)

    result = await target.run(Sample(id=0, input="hi"))

    argv = json.loads(record.read_text())["argv"]
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
