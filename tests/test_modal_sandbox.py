"""Unit tests for ModalSandboxSpec and the Modal sandbox driver.

The live driver test is skipped unless Modal credentials are configured
(MODAL_TOKEN_ID / MODAL_TOKEN_SECRET or ~/.modal.toml). It builds the
bundled system base and exact application layers. The spec-parsing tests do
not touch Modal at all.
"""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

import anyio
import pytest

from letta_evals.models import ModalSandboxSpec, SuiteSpec
from letta_evals.sandbox.base import ExecResult
from letta_evals.sandbox.dispatch import build_upload_filter
from letta_evals.sandbox.modal import ModalSandbox


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
    def test_registry_image_keeps_runtime_pins_optional(self):
        spec = ModalSandboxSpec(image="ghcr.io/custom/runtime:1.0")
        assert spec.kind == "modal"
        assert spec.image == "ghcr.io/custom/runtime:1.0"
        assert spec.letta_evals_version is None
        assert spec.letta_code_version is None
        assert spec.cpu == 2
        assert spec.memory_mb == 2048
        assert spec.timeout_sec == 1800
        assert spec.block_network is False
        assert spec.app_name == "letta-evals"
        assert spec.secrets == []
        assert spec.forward_env == []
        assert spec.volumes == {}
        assert spec.project_root is None
        assert spec.respect_gitignore is True

    @pytest.mark.parametrize(
        ("kwargs", "missing"),
        [
            ({}, "letta_evals_version, letta_code_version"),
            ({"letta_evals_version": "0.25.0"}, "letta_code_version"),
            ({"letta_code_version": "0.30.5"}, "letta_evals_version"),
            ({"letta_evals_version": "", "letta_code_version": "0.30.5"}, "letta_evals_version"),
            ({"letta_evals_version": "0.25.0", "letta_code_version": ""}, "letta_code_version"),
        ],
    )
    def test_bundled_image_requires_both_pins(self, kwargs, missing):
        with pytest.raises(ValueError, match=missing):
            ModalSandboxSpec(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"letta_evals_version": "latest", "letta_code_version": "0.30.5"},
            {"letta_evals_version": "0.25.0", "letta_code_version": "latest"},
            {"letta_evals_version": "0.25.0", "letta_code_version": "^0.30.5"},
            {
                "letta_evals_version": "https://example.com/letta-evals-current.whl",
                "letta_code_version": "0.30.5",
            },
            {
                "letta_evals_version": "letta-evals @ git+https://github.com/letta-ai/letta-evals.git@main",
                "letta_code_version": "0.30.5",
            },
        ],
    )
    def test_bundled_image_rejects_mutable_or_ranged_pins(self, kwargs):
        with pytest.raises(ValueError):
            ModalSandboxSpec(**kwargs)

    def test_custom_image_rejects_unverifiable_eval_pin(self):
        with pytest.raises(ValueError, match="full 40-character commit SHA"):
            ModalSandboxSpec(
                image="img:test",
                letta_evals_version="git+https://github.com/letta-ai/letta-evals.git@main",
            )

    def test_rejects_noncanonical_python_version(self):
        with pytest.raises(ValueError, match="canonical exact version"):
            ModalSandboxSpec(
                letta_evals_version="1.0-rc1",
                letta_code_version="0.30.5",
            )

    def test_yaml_without_image_accepts_exact_pins(self):
        commit = "a" * 40
        yaml_data = {
            "name": "u",
            "dataset": "s.jsonl",
            "target": {"kind": "letta_code", "model_handles": ["openai/gpt-4.1-mini"]},
            "graders": {"g": {"kind": "tool", "function": "exact_match"}},
            "reward": {"kind": "metric", "metric_key": "g"},
            "sandbox": {
                "kind": "modal",
                "letta_evals_version": (f"letta-evals @ git+https://github.com/letta-ai/letta-evals.git@{commit}"),
                "letta_code_version": "0.30.5",
            },
        }
        suite = SuiteSpec.from_yaml(yaml_data)
        assert suite.sandbox is not None
        assert suite.sandbox.image is None
        assert suite.sandbox.letta_code_version == "0.30.5"

    def test_bundled_dockerfile_exists(self):
        """The Dockerfile must ship with the package so Image.from_dockerfile
        can resolve it when image is unset."""
        from pathlib import Path

        import letta_evals.sandbox as sandbox_pkg

        dockerfile = Path(sandbox_pkg.__file__).parent / "Dockerfile"
        assert dockerfile.is_file(), f"Bundled Dockerfile missing: {dockerfile}"
        contents = dockerfile.read_text()
        # The bundled Dockerfile is intentionally application-independent.
        assert "ARG LETTA_EVALS_VERSION" not in contents
        assert "ARG LETTA_CODE_VERSION" not in contents
        assert "npm install -g" not in contents
        assert '"typing_extensions>=4.15.0"' in contents
        assert "from typing_extensions import Sentinel" in contents
        assert "setup_22.x" in contents
        assert "major === 22 && minor < 19" in contents

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

    def test_project_root_resolves_relative_to_suite_dir(self, tmp_path):
        suite_dir = tmp_path / "suites" / "mini"
        suite_dir.mkdir(parents=True)
        yaml_data = _minimal_suite_yaml(project_root="../..")

        suite = SuiteSpec.from_yaml(yaml_data, base_dir=suite_dir)

        assert suite.sandbox.project_root == tmp_path.resolve()

    def test_absolute_project_root_is_preserved(self, tmp_path):
        yaml_data = _minimal_suite_yaml(project_root=str(tmp_path))
        suite = SuiteSpec.from_yaml(yaml_data, base_dir=tmp_path / "sub")
        assert suite.sandbox.project_root == tmp_path.resolve()

    def test_project_root_must_be_ancestor_of_suite(self, tmp_path):
        """A project_root that doesn't contain the suite fails at load, so
        `validate` catches it once instead of every sample failing later."""
        suite_dir = tmp_path / "suite"
        suite_dir.mkdir()
        (tmp_path / "other").mkdir()
        yaml_data = _minimal_suite_yaml(project_root="../other")

        with pytest.raises(ValueError, match="must be an ancestor"):
            SuiteSpec.from_yaml(yaml_data, base_dir=suite_dir, suite_path=suite_dir / "suite.yaml")

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


class TestUploadFilter:
    def test_default_excludes_drop_junk_but_keep_code_and_data(self):
        keep = build_upload_filter(ModalSandboxSpec(image="img:test"))
        # Kept: source, config, data.
        assert keep("pkg/mod.py")
        assert keep("pyproject.toml")
        assert keep("data/samples.jsonl")
        # Dropped: VCS, caches, virtualenvs, compiled/editor junk (at any depth).
        assert not keep(".git")
        assert not keep("pkg/__pycache__")
        assert not keep("pkg/mod.pyc")
        assert not keep("node_modules")

    def test_respects_gitignore_at_root(self, tmp_path):
        (tmp_path / ".gitignore").write_text("data/large/\n*.log\n")
        keep = build_upload_filter(ModalSandboxSpec(image="img:test"), root=tmp_path)
        assert keep("pkg/mod.py")
        assert not keep("data/large/blob.bin")
        assert not keep("run.log")

    def test_gitignore_ignored_when_respect_gitignore_false(self, tmp_path):
        (tmp_path / ".gitignore").write_text("*.log\n")
        keep = build_upload_filter(ModalSandboxSpec(image="img:test", respect_gitignore=False), root=tmp_path)
        assert keep("run.log")


class TestUploadDirFiltering:
    def test_tarball_excludes_filtered_members(self, tmp_path):
        """upload_dir wires the filter into tarfile.add: excluded subtrees never
        enter the streamed archive."""
        root = tmp_path / "project"
        (root / "pkg").mkdir(parents=True)
        (root / "pkg" / "mod.py").write_text("x = 1\n")
        (root / "pkg" / "__pycache__").mkdir()
        (root / "pkg" / "__pycache__" / "mod.pyc").write_text("junk")
        (root / ".git").mkdir()
        (root / ".git" / "config").write_text("[core]\n")
        (root / "data").mkdir()
        (root / "data" / "samples.jsonl").write_text("{}\n")

        captured: dict = {}

        class _TarCapturingSandbox(ModalSandbox):
            def __init__(self):
                self._sandbox = object()  # non-None sentinel; skip real Modal init
                self.session_id = "cap"

            async def upload_file(self, local: Path, remote: str) -> None:
                with tarfile.open(local, "r:gz") as tar:
                    captured["names"] = {n for n in tar.getnames() if n not in (".", "")}

            async def exec(self, command, env=None, timeout_sec=None):
                return ExecResult(stdout="", stderr="", return_code=0)

        sb = _TarCapturingSandbox()
        keep = build_upload_filter(ModalSandboxSpec(image="img:test"))
        anyio.run(sb.upload_dir, root, "/mnt/project", keep)

        names = captured["names"]
        assert "./pkg/mod.py" in names
        assert "./data/samples.jsonl" in names
        # Excluded subtrees are pruned wholesale — not even the dir entry ships.
        assert not any(".git" in n for n in names)
        assert not any("__pycache__" in n for n in names)
        assert not any(n.endswith(".pyc") for n in names)


class TestExecResult:
    def test_construct(self):
        r = ExecResult(stdout="ok", stderr="", return_code=0)
        assert r.stdout == "ok"
        assert r.return_code == 0


def test_letta_evals_package_spec_normalization():
    from letta_evals.sandbox.modal import _letta_evals_package_spec

    commit = "a" * 40
    direct = f"letta-evals @ git+https://github.com/letta-ai/letta-evals.git@{commit}"
    assert _letta_evals_package_spec("0.25.0") == "letta-evals==0.25.0"
    assert _letta_evals_package_spec(direct) == direct


def test_direct_url_log_summary_redacts_credentials_and_query():
    from letta_evals.sandbox.modal import _direct_url_log_summary

    summary = _direct_url_log_summary(
        {
            "url": "https://user:secret@example.com/org/repo.git?token=secret",
            "vcs_info": {"commit_id": "a" * 40, "vcs": "git"},
        }
    )
    assert summary == {"url": "https://example.com/org/repo.git", "commit_id": "a" * 40}


def _install_fake_modal(monkeypatch):
    """Inject a fake ``modal`` SDK and return it so tests can assert on calls."""
    import sys
    from unittest.mock import AsyncMock, MagicMock

    import letta_evals.sandbox.modal as modal_driver

    fake_modal = MagicMock(name="modal")
    fake_modal.App.lookup.aio = AsyncMock(return_value=MagicMock(name="app"))
    fake_image = MagicMock(name="image")
    fake_image.object_id = "im-xyz"
    fake_image.run_commands.return_value = fake_image
    fake_modal.Image.from_dockerfile = MagicMock(return_value=fake_image)
    fake_modal.Image.from_registry = MagicMock(return_value=fake_image)
    fake_sandbox = MagicMock(name="sandbox")
    fake_sandbox.object_id = "sb-xyz"
    fake_sandbox.terminate.aio = AsyncMock()
    fake_modal.Sandbox.create.aio = AsyncMock(return_value=fake_sandbox)
    runtime_verifier = AsyncMock()

    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setattr(modal_driver, "_check_modal_auth", lambda: None)
    monkeypatch.setattr(modal_driver.ModalSandbox, "_verify_runtime", runtime_verifier)
    fake_modal._runtime_verifier = runtime_verifier
    return fake_modal


class TestModalDriverImageBuild:
    """The driver puts exact runtime pins in literal Modal layer definitions."""

    @pytest.mark.asyncio
    async def test_exact_versions_become_explicit_image_layers(self, monkeypatch):
        from letta_evals.sandbox.modal import ModalSandbox

        fake_modal = _install_fake_modal(monkeypatch)
        spec = ModalSandboxSpec(
            letta_evals_version="0.24.0",
            letta_code_version="0.30.5",
            timeout_sec=60,
            cpu=1,
            memory_mb=512,
        )
        sandbox = ModalSandbox(spec=spec, session_id="unit-versioned-layers")

        await sandbox.start()

        fake_modal.Image.from_dockerfile.assert_called_once()
        _, kwargs = fake_modal.Image.from_dockerfile.call_args
        assert kwargs == {}
        layer_calls = fake_modal.Image.from_dockerfile.return_value.run_commands.call_args_list
        assert "letta-evals==0.24.0" in layer_calls[0].args[0]
        assert "from typing_extensions import Sentinel" in layer_calls[0].args[1]
        assert "@letta-ai/letta-code@0.30.5" in layer_calls[1].args[0]
        assert layer_calls[1].args[1:] == ("node --version", "letta --version")
        assert sandbox.image_id == "im-xyz"
        fake_modal.Image.from_registry.assert_not_called()
        fake_modal._runtime_verifier.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_changing_pins_changes_literal_layer_definitions(self, monkeypatch):
        from letta_evals.sandbox.modal import ModalSandbox

        fake_modal = _install_fake_modal(monkeypatch)
        image = fake_modal.Image.from_dockerfile.return_value

        pins = [
            ("0.24.0", "0.30.5"),
            ("0.25.0", "0.30.5"),  # only letta-evals changes
            ("0.24.0", "0.30.7"),  # only letta-code changes
        ]
        for evals_version, code_version in pins:
            spec = ModalSandboxSpec(
                letta_evals_version=evals_version,
                letta_code_version=code_version,
                timeout_sec=60,
            )
            await ModalSandbox(spec=spec, session_id=f"unit-{evals_version}-{code_version}").start()

        commands = [call.args[0] for call in image.run_commands.call_args_list]
        assert commands == [
            "python -m pip install --no-cache-dir --upgrade letta-evals==0.24.0",
            "npm install -g --omit=dev @letta-ai/letta-code@0.30.5",
            "python -m pip install --no-cache-dir --upgrade letta-evals==0.25.0",
            "npm install -g --omit=dev @letta-ai/letta-code@0.30.5",
            "python -m pip install --no-cache-dir --upgrade letta-evals==0.24.0",
            "npm install -g --omit=dev @letta-ai/letta-code@0.30.7",
        ]

    @pytest.mark.asyncio
    async def test_direct_git_pin_is_shell_quoted_in_layer(self, monkeypatch):
        from letta_evals.sandbox.modal import ModalSandbox

        fake_modal = _install_fake_modal(monkeypatch)
        commit = "a" * 40
        direct = f"letta-evals @ git+https://github.com/letta-ai/letta-evals.git@{commit}"
        spec = ModalSandboxSpec(
            letta_evals_version=direct,
            letta_code_version="0.30.5",
            timeout_sec=60,
        )

        await ModalSandbox(spec=spec, session_id="unit-direct-ref").start()

        evals_command = fake_modal.Image.from_dockerfile.return_value.run_commands.call_args_list[0].args[0]
        assert evals_command.endswith(f"'{direct}'")

    @pytest.mark.asyncio
    async def test_registry_image_skips_versioned_layers(self, monkeypatch):
        from letta_evals.sandbox.modal import ModalSandbox

        fake_modal = _install_fake_modal(monkeypatch)
        spec = ModalSandboxSpec(image="ghcr.io/custom/runtime:1.0", letta_code_version="0.30.5", timeout_sec=60)
        sandbox = ModalSandbox(spec=spec, session_id="unit-registry")

        await sandbox.start()

        fake_modal.Image.from_registry.assert_called_once_with("ghcr.io/custom/runtime:1.0")
        fake_modal.Image.from_dockerfile.assert_not_called()
        fake_modal.Image.from_registry.return_value.run_commands.assert_not_called()

    @pytest.mark.asyncio
    async def test_runtime_verification_failure_terminates_created_sandbox(self, monkeypatch):
        from letta_evals.sandbox.modal import ModalSandbox, VersionMismatch

        fake_modal = _install_fake_modal(monkeypatch)
        fake_modal._runtime_verifier.side_effect = VersionMismatch("stale runtime")
        spec = ModalSandboxSpec(letta_evals_version="0.25.0", letta_code_version="0.30.5")
        sandbox = ModalSandbox(spec=spec, session_id="unit-verification-failure")

        with pytest.raises(VersionMismatch, match="stale runtime"):
            await sandbox.start()

        created = fake_modal.Sandbox.create.aio.return_value
        created.terminate.aio.assert_awaited_once()
        assert sandbox.sandbox_id is None


class TestModalRuntimeVerification:
    @staticmethod
    def _probe_result(
        *,
        evals_version="0.25.0",
        code_version="0.30.5",
        node_version="v22.19.0",
        commit=None,
    ):
        import json

        direct_url = None
        if commit:
            direct_url = {
                "url": "https://github.com/letta-ai/letta-evals.git",
                "vcs_info": {"commit_id": commit, "vcs": "git"},
            }
        return ExecResult(
            stdout=json.dumps(
                {
                    "letta_evals_version": evals_version,
                    "letta_evals_direct_url": direct_url,
                    "letta_code_version": code_version,
                    "letta_version_output": code_version,
                    "node_version": node_version,
                }
            ),
            stderr="",
            return_code=0,
        )

    @pytest.mark.asyncio
    async def test_accepts_exact_versions(self, monkeypatch):
        from unittest.mock import AsyncMock

        sandbox = ModalSandbox(
            ModalSandboxSpec(letta_evals_version="0.25.0", letta_code_version="0.30.5"),
            session_id="verify-exact",
        )
        sandbox._sandbox = object()
        monkeypatch.setattr(sandbox, "exec", AsyncMock(return_value=self._probe_result()))

        await sandbox._verify_runtime()

    @pytest.mark.asyncio
    async def test_custom_image_allows_older_node_and_non_global_code_install(self, monkeypatch):
        from unittest.mock import AsyncMock

        sandbox = ModalSandbox(
            ModalSandboxSpec(image="img:test", letta_evals_version="0.25.0"),
            session_id="verify-custom",
        )
        sandbox._sandbox = object()
        probe = self._probe_result(code_version=None, node_version="v20.18.0")
        monkeypatch.setattr(sandbox, "exec", AsyncMock(return_value=probe))

        await sandbox._verify_runtime()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "probe_result",
        [
            ExecResult(stdout="", stderr="probe failed", return_code=1),
            ExecResult(stdout="not-json", stderr="", return_code=0),
        ],
    )
    async def test_rejects_failed_or_invalid_runtime_probe(self, monkeypatch, probe_result):
        from unittest.mock import AsyncMock

        from letta_evals.sandbox.modal import RuntimeProbeError

        sandbox = ModalSandbox(
            ModalSandboxSpec(letta_evals_version="0.25.0", letta_code_version="0.30.5"),
            session_id="verify-probe-failure",
        )
        sandbox._sandbox = object()
        monkeypatch.setattr(sandbox, "exec", AsyncMock(return_value=probe_result))

        with pytest.raises(RuntimeProbeError):
            await sandbox._verify_runtime()

    @pytest.mark.asyncio
    async def test_accepts_exact_git_commit(self, monkeypatch):
        from unittest.mock import AsyncMock

        commit = "a" * 40
        direct = f"letta-evals @ git+https://github.com/letta-ai/letta-evals.git@{commit}"
        sandbox = ModalSandbox(
            ModalSandboxSpec(letta_evals_version=direct, letta_code_version="0.30.5"),
            session_id="verify-commit",
        )
        sandbox._sandbox = object()
        monkeypatch.setattr(sandbox, "exec", AsyncMock(return_value=self._probe_result(commit=commit)))

        await sandbox._verify_runtime()

    @pytest.mark.asyncio
    async def test_rejects_git_commit_mismatch(self, monkeypatch):
        from unittest.mock import AsyncMock

        from letta_evals.sandbox.modal import VersionMismatch

        expected_commit = "a" * 40
        actual_commit = "b" * 40
        direct = f"letta-evals @ git+https://github.com/letta-ai/letta-evals.git@{expected_commit}"
        sandbox = ModalSandbox(
            ModalSandboxSpec(letta_evals_version=direct, letta_code_version="0.30.5"),
            session_id="verify-commit-mismatch",
        )
        sandbox._sandbox = object()
        monkeypatch.setattr(sandbox, "exec", AsyncMock(return_value=self._probe_result(commit=actual_commit)))

        with pytest.raises(VersionMismatch):
            await sandbox._verify_runtime()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "probe_result",
        [
            _probe_result(evals_version="0.24.0"),
            _probe_result(code_version="0.30.7"),
            _probe_result(node_version="v22.18.0"),
        ],
    )
    async def test_rejects_runtime_mismatches(self, monkeypatch, probe_result):
        from unittest.mock import AsyncMock

        from letta_evals.sandbox.modal import VersionMismatch

        sandbox = ModalSandbox(
            ModalSandboxSpec(letta_evals_version="0.25.0", letta_code_version="0.30.5"),
            session_id="verify-mismatch",
        )
        sandbox._sandbox = object()
        monkeypatch.setattr(sandbox, "exec", AsyncMock(return_value=probe_result))

        with pytest.raises(VersionMismatch):
            await sandbox._verify_runtime()


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

        # No image override: build the stable base plus exact runtime layers.
        spec = ModalSandboxSpec(
            letta_evals_version="0.25.0",
            letta_code_version="0.30.5",
            timeout_sec=120,
            cpu=1,
            memory_mb=512,
        )
        sandbox = ModalSandbox(spec=spec, session_id="unit-echo")
        await sandbox.start()
        try:
            assert sandbox.image_id and sandbox.image_id.startswith("im-")
            res = await sandbox.exec("echo hello")
            assert res.return_code == 0
            assert "hello" in res.stdout
            help_result = await sandbox.exec("letta --help")
            assert help_result.return_code == 0
            assert "--stateless" in help_result.stdout
        finally:
            await sandbox.stop()

    @pytest.mark.asyncio
    async def test_changing_code_pin_changes_image(self):
        from letta_evals.sandbox.modal import ModalSandbox

        image_ids = []
        for code_version in ("0.30.5", "0.30.7"):
            spec = ModalSandboxSpec(
                letta_evals_version="0.25.0",
                letta_code_version=code_version,
                timeout_sec=180,
                cpu=1,
                memory_mb=512,
            )
            sandbox = ModalSandbox(spec=spec, session_id=f"unit-pin-{code_version}")
            await sandbox.start()
            try:
                assert sandbox.image_id is not None
                image_ids.append(sandbox.image_id)
            finally:
                await sandbox.stop()

        assert image_ids[0] != image_ids[1]
