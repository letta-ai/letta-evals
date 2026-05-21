"""Unit tests for ModalSandboxSpec and the Modal sandbox driver.

The live driver test is skipped unless MODAL_TOKEN_ID and MODAL_TOKEN_SECRET
are set, and a test image is reachable. The spec-parsing tests do not touch
Modal at all.
"""

from __future__ import annotations

import os

import pytest

from letta_evals.models import ModalSandboxSpec, SuiteSpec
from letta_evals.sandbox.base import ExecResult


def _minimal_suite_yaml(**sandbox_overrides):
    sandbox = {
        "kind": "modal",
        "image": "ghcr.io/letta/letta-evals-runtime:test",
        **sandbox_overrides,
    }
    return {
        "name": "unit-suite",
        "dataset": "samples.jsonl",
        "target": {"kind": "letta_agent", "agent_id": "agent-test"},
        "graders": {
            "g": {
                "kind": "tool",
                "function": "exact_match",
            }
        },
        "gate": {"kind": "simple", "metric_key": "g", "op": "gte", "value": 1.0},
        "sandbox": sandbox,
    }


class TestModalSandboxSpec:
    def test_defaults(self):
        spec = ModalSandboxSpec(image="img:latest")
        assert spec.kind == "modal"
        assert spec.image == "img:latest"
        assert spec.cpu == 2
        assert spec.memory_mb == 2048
        assert spec.timeout_sec == 1800
        assert spec.block_network is False
        assert spec.app_name == "letta-evals"
        assert spec.secrets == []
        assert spec.volumes == {}
        assert spec.letta_evals_version is None

    def test_overrides(self):
        spec = ModalSandboxSpec(
            image="img:1",
            letta_evals_version="0.17.0",
            secrets=["k1", "k2"],
            volumes={"/mnt/cache": "cache-vol"},
            cpu=4,
            memory_mb=8192,
            timeout_sec=600,
            block_network=True,
            app_name="my-app",
        )
        assert spec.secrets == ["k1", "k2"]
        assert spec.volumes == {"/mnt/cache": "cache-vol"}
        assert spec.cpu == 4
        assert spec.memory_mb == 8192
        assert spec.timeout_sec == 600
        assert spec.block_network is True
        assert spec.app_name == "my-app"
        assert spec.letta_evals_version == "0.17.0"


class TestSuiteSpecWithSandbox:
    def test_sandbox_absent_keeps_field_none(self):
        yaml_data = _minimal_suite_yaml()
        del yaml_data["sandbox"]
        suite = SuiteSpec.from_yaml(yaml_data)
        assert suite.sandbox is None

    def test_sandbox_parsed_from_yaml(self):
        yaml_data = _minimal_suite_yaml(secrets=["letta-api-key"], cpu=4)
        suite = SuiteSpec.from_yaml(yaml_data)
        assert suite.sandbox is not None
        assert suite.sandbox.kind == "modal"
        assert suite.sandbox.image == "ghcr.io/letta/letta-evals-runtime:test"
        assert suite.sandbox.secrets == ["letta-api-key"]
        assert suite.sandbox.cpu == 4


class TestExecResult:
    def test_construct(self):
        r = ExecResult(stdout="ok", stderr="", return_code=0)
        assert r.stdout == "ok"
        assert r.return_code == 0


class TestModalDriverLazyImport:
    def test_import_does_not_require_modal(self, monkeypatch):
        """Importing the module must not pull in the modal SDK eagerly."""
        # Force re-import.
        import importlib
        import sys

        for mod in list(sys.modules):
            if mod == "letta_evals.sandbox.modal":
                del sys.modules[mod]

        # Pretend modal is unavailable; importing our module should still work.
        monkeypatch.setitem(sys.modules, "modal", None)  # sentinel; not used since import is lazy
        importlib.import_module("letta_evals.sandbox.modal")


@pytest.mark.skipif(
    not (os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET")) and not os.path.exists(
        os.path.expanduser("~/.modal.toml")
    ),
    reason="Modal credentials not configured",
)
@pytest.mark.skipif(
    not os.getenv("LETTA_EVALS_MODAL_TEST_IMAGE"),
    reason="Set LETTA_EVALS_MODAL_TEST_IMAGE to a reachable image to run live driver tests",
)
class TestModalDriverLive:
    @pytest.mark.asyncio
    async def test_echo_round_trip(self):
        from letta_evals.sandbox.modal import ModalSandbox

        spec = ModalSandboxSpec(
            image=os.environ["LETTA_EVALS_MODAL_TEST_IMAGE"],
            timeout_sec=120,
            cpu=1,
            memory_mb=512,
        )
        sandbox = ModalSandbox(spec=spec, session_id="unit-echo")
        await sandbox.start()
        try:
            res = await sandbox.exec("echo hello")
            assert res.return_code == 0
            assert "hello" in res.stdout
        finally:
            await sandbox.stop()
