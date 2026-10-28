"""Runner._run_setup dispatch across the supported @suite_setup signatures."""

import json
from unittest.mock import MagicMock

import pytest

from letta_evals.runner import Runner

SIGNATURES = {
    "sync_no_args": "def setup():",
    "sync_client": "def setup(client):",
    "sync_client_model": "def setup(client, model_handle):",
    "async_no_args": "async def setup():",
    "async_client": "async def setup(client):",
    "async_client_model": "async def setup(client, model_handle):",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", sorted(SIGNATURES))
async def test_run_setup_passes_declared_args(tmp_path, variant):
    record = tmp_path / "record.json"
    (tmp_path / "setup.py").write_text(
        "import json\n"
        "from letta_evals import suite_setup\n"
        "@suite_setup\n"
        f"{SIGNATURES[variant]}\n"
        f"    open({str(record)!r}, 'w').write(json.dumps(sorted(k for k in locals() if k != 'json')))\n"
    )
    runner = Runner.__new__(Runner)
    runner.suite = MagicMock(setup_script="setup.py:setup", base_dir=tmp_path)
    runner.client = MagicMock()

    await runner._run_setup(model_handle="openai/gpt-x")

    expected = {"no_args": [], "client": ["client"], "client_model": ["client", "model_handle"]}
    assert json.loads(record.read_text()) == expected[variant.split("_", 1)[1]]
