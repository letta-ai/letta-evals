"""Unit tests for ModalSandboxSpec and the Modal sandbox driver.

The live driver test is skipped unless Modal credentials are configured
(MODAL_TOKEN_ID / MODAL_TOKEN_SECRET or ~/.modal.toml). It builds the
bundled Dockerfile (the default image) — no per-test image config required.
The spec-parsing tests do not touch Modal at all.
"""

from __future__ import annotations

import os
from pathlib import Path

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
        "target": {"kind": "letta_code", "model_handles": ["openai/gpt-4.1-mini"]},
        "graders": {
            "g": {
                "kind": "tool",
                "function": "exact_match",
            }
        },
        "reward": {"kind": "metric", "metric_key": "g"},
        "sandbox": sandbox,
    }


class TestModalSandboxSpec:
    def test_defaults(self):
        """Image defaults to None (build from bundled Dockerfile); other
        defaults let a minimal `sandbox: { kind: modal }` block work."""
        spec = ModalSandboxSpec()
        assert spec.kind == "modal"
        assert spec.image is None
        assert spec.cpu == 2
        assert spec.memory_mb == 2048
        assert spec.timeout_sec == 1800
        assert spec.block_network is False
        assert spec.app_name == "letta-evals"
        assert spec.secrets == []
        assert spec.forward_env == []
        assert spec.volumes == {}
        assert spec.letta_evals_version is None
        assert spec.letta_code_version is None

    def test_image_override(self):
        spec = ModalSandboxSpec(image="ghcr.io/custom/runtime:1.0")
        assert spec.image == "ghcr.io/custom/runtime:1.0"

    def test_yaml_without_image_leaves_image_unset(self):
        """A suite YAML can declare `sandbox: { kind: modal }` with no image
        — the driver builds from the bundled Dockerfile."""
        yaml_data = {
            "name": "u",
            "dataset": "s.jsonl",
            "target": {"kind": "letta_code", "model_handles": ["openai/gpt-4.1-mini"]},
            "graders": {"g": {"kind": "tool", "function": "exact_match"}},
            "reward": {"kind": "metric", "metric_key": "g"},
            "sandbox": {"kind": "modal"},
        }
        suite = SuiteSpec.from_yaml(yaml_data)
        assert suite.sandbox is not None
        assert suite.sandbox.image is None

    def test_bundled_dockerfile_exists(self):
        """The Dockerfile must ship with the package so Image.from_dockerfile
        can resolve it when image is unset."""
        from pathlib import Path

        import letta_evals.sandbox as sandbox_pkg

        dockerfile = Path(sandbox_pkg.__file__).parent / "Dockerfile"
        assert dockerfile.is_file(), f"Bundled Dockerfile missing: {dockerfile}"
        contents = dockerfile.read_text()
        # Sanity-check the recipe carries both runtimes.
        assert "letta-evals" in contents
        assert "@letta-ai/letta-code" in contents
        # The letta-code install must be pinnable via the LETTA_CODE_VERSION
        # build arg the Modal driver passes from sandbox.letta_code_version.
        assert "ARG LETTA_CODE_VERSION" in contents
        assert "@letta-ai/letta-code@${LETTA_CODE_VERSION}" in contents

    def test_overrides(self):
        spec = ModalSandboxSpec(
            image="img:1",
            letta_evals_version="0.17.0",
            letta_code_version="0.27.17",
            secrets=["k1", "k2"],
            forward_env=["MY_CUSTOM_KEY"],
            volumes={"/mnt/cache": "cache-vol"},
            cpu=4,
            memory_mb=8192,
            timeout_sec=600,
            block_network=True,
            app_name="my-app",
        )
        assert spec.secrets == ["k1", "k2"]
        assert spec.forward_env == ["MY_CUSTOM_KEY"]
        assert spec.volumes == {"/mnt/cache": "cache-vol"}
        assert spec.cpu == 4
        assert spec.memory_mb == 8192
        assert spec.timeout_sec == 600
        assert spec.block_network is True
        assert spec.app_name == "my-app"
        assert spec.letta_evals_version == "0.17.0"
        assert spec.letta_code_version == "0.27.17"


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

    def test_target_memory_workspace_fields_parse_and_resolve(self, tmp_path):
        yaml_data = _minimal_suite_yaml()
        yaml_data["target"] = {
            "kind": "letta_code",
            "model_handles": ["openai/gpt-4.1-mini"],
            "permission_mode": "unrestricted",
            "memory_workspace": True,
            "memory_dir": "seeded-memory",
        }

        suite = SuiteSpec.from_yaml(yaml_data, base_dir=tmp_path)

        assert suite.target.permission_mode == "unrestricted"
        assert suite.target.memory_workspace is True
        assert suite.target.memory_dir == Path(tmp_path / "seeded-memory").resolve()

    def test_target_rejects_removed_memory_permission_mode(self, tmp_path):
        yaml_data = _minimal_suite_yaml()
        yaml_data["target"] = {
            "kind": "letta_code",
            "model_handles": ["openai/gpt-4.1-mini"],
            "permission_mode": "memory",
        }

        with pytest.raises(ValueError, match="permission_mode: memory was removed"):
            SuiteSpec.from_yaml(yaml_data, base_dir=tmp_path)


class TestExecResult:
    def test_construct(self):
        r = ExecResult(stdout="ok", stderr="", return_code=0)
        assert r.stdout == "ok"
        assert r.return_code == 0


def _install_fake_modal(monkeypatch):
    """Inject a fake ``modal`` SDK and return it so tests can assert on calls.

    ``ModalSandbox.start`` imports modal lazily and calls App.lookup,
    Image.from_dockerfile / from_registry, and Sandbox.create. We stub all of
    them so start() runs end-to-end without touching Modal, letting us inspect
    how the image was built.
    """
    import sys
    from unittest.mock import AsyncMock, MagicMock

    import letta_evals.sandbox.modal as modal_driver

    fake_modal = MagicMock(name="modal")
    fake_modal.App.lookup.aio = AsyncMock(return_value=MagicMock(name="app"))
    fake_image = MagicMock(name="image")
    fake_modal.Image.from_dockerfile = MagicMock(return_value=fake_image)
    fake_modal.Image.from_registry = MagicMock(return_value=fake_image)
    fake_sandbox = MagicMock(name="sandbox")
    fake_sandbox.object_id = "sb-xyz"
    fake_modal.Sandbox.create.aio = AsyncMock(return_value=fake_sandbox)

    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setattr(modal_driver, "_check_modal_auth", lambda: None)
    return fake_modal


class TestModalDriverImageBuild:
    """The driver turns sandbox.letta_code_version into a Dockerfile build arg."""

    @pytest.mark.asyncio
    async def test_letta_code_version_becomes_build_arg(self, monkeypatch):
        from letta_evals.sandbox.modal import ModalSandbox

        fake_modal = _install_fake_modal(monkeypatch)
        spec = ModalSandboxSpec(letta_code_version="0.27.17", timeout_sec=60, cpu=1, memory_mb=512)
        sandbox = ModalSandbox(spec=spec, session_id="unit-build-args")

        await sandbox.start()

        fake_modal.Image.from_dockerfile.assert_called_once()
        _, kwargs = fake_modal.Image.from_dockerfile.call_args
        assert kwargs["build_args"] == {"LETTA_CODE_VERSION": "0.27.17"}
        fake_modal.Image.from_registry.assert_not_called()

    @pytest.mark.asyncio
    async def test_unset_version_passes_empty_build_args(self, monkeypatch):
        from letta_evals.sandbox.modal import ModalSandbox

        fake_modal = _install_fake_modal(monkeypatch)
        spec = ModalSandboxSpec(timeout_sec=60)
        sandbox = ModalSandbox(spec=spec, session_id="unit-no-version")

        await sandbox.start()

        _, kwargs = fake_modal.Image.from_dockerfile.call_args
        assert kwargs["build_args"] == {}

    @pytest.mark.asyncio
    async def test_registry_image_ignores_version(self, monkeypatch):
        from letta_evals.sandbox.modal import ModalSandbox

        fake_modal = _install_fake_modal(monkeypatch)
        spec = ModalSandboxSpec(image="ghcr.io/custom/runtime:1.0", letta_code_version="0.27.17", timeout_sec=60)
        sandbox = ModalSandbox(spec=spec, session_id="unit-registry")

        await sandbox.start()

        fake_modal.Image.from_registry.assert_called_once_with("ghcr.io/custom/runtime:1.0")
        fake_modal.Image.from_dockerfile.assert_not_called()


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
    not os.getenv("LETTA_EVALS_LIVE_MODAL_TESTS"),
    reason=(
        "Live Modal driver tests are opt-in (they pull a real image and "
        "create a real sandbox). Set LETTA_EVALS_LIVE_MODAL_TESTS=1 to run."
    ),
)
@pytest.mark.skipif(
    not (os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"))
    and not os.path.exists(os.path.expanduser("~/.modal.toml")),
    reason="Modal credentials not configured",
)
class TestModalDriverLive:
    """Live driver test against the default base image.

    Builds the bundled Dockerfile (letta_evals/sandbox/Dockerfile), the
    default when `image` is unset, so suite authors don't have to wire any
    project-specific image to exercise this path.
    """

    @pytest.mark.asyncio
    async def test_echo_round_trip(self):
        from letta_evals.sandbox.modal import ModalSandbox

        # No image override: defaults to building the bundled Dockerfile.
        spec = ModalSandboxSpec(timeout_sec=120, cpu=1, memory_mb=512)
        sandbox = ModalSandbox(spec=spec, session_id="unit-echo")
        await sandbox.start()
        try:
            res = await sandbox.exec("echo hello")
            assert res.return_code == 0
            assert "hello" in res.stdout
        finally:
            await sandbox.stop()
