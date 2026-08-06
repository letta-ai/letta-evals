"""Modal sandbox driver.

One Modal sandbox per sample. Created on :meth:`ModalSandbox.start`, executes
the in-sandbox ``letta-evals run --sample`` invocation, and tears down on
:meth:`ModalSandbox.stop`. The Modal SDK (a letta-evals dependency) is
imported lazily so this module stays importable with no import-time cost
when the sandbox driver isn't used.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import tarfile
import tempfile
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlsplit, urlunsplit

from letta_evals.models import ModalSandboxSpec
from letta_evals.sandbox.base import AbstractSandbox, ExecResult, SandboxAuthError, SandboxNotInstalledError

logger = logging.getLogger(__name__)

_RUNTIME_PROBE_COMMAND = r"""python - <<'PY'
import importlib.metadata
import json
import pathlib
import subprocess


def run(*args):
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return (result.stdout or result.stderr).strip()


dist = importlib.metadata.distribution("letta-evals")
direct_url_text = dist.read_text("direct_url.json")
try:
    npm_root = pathlib.Path(run("npm", "root", "-g"))
    code_package = json.loads((npm_root / "@letta-ai" / "letta-code" / "package.json").read_text())
    code_version = code_package["version"]
except (OSError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError):
    code_version = None

print(json.dumps({
    "letta_evals_version": dist.version,
    "letta_evals_direct_url": json.loads(direct_url_text) if direct_url_text else None,
    "letta_code_version": code_version,
    "letta_version_output": run("letta", "--version"),
    "node_version": run("node", "--version"),
}, sort_keys=True))
PY"""


class VersionMismatch(RuntimeError):
    """Raised when a sandbox runtime does not match its requested pin."""


class RuntimeProbeError(RuntimeError):
    """Raised when the sandbox runtime cannot report its installed versions."""


def _is_direct_letta_evals_package_spec(spec: str) -> bool:
    return spec.startswith("git+") or "://" in spec or spec.startswith("letta-evals @")


def _letta_evals_package_spec(pin: str) -> str:
    """Normalize a version or direct reference into a pip package spec."""
    return pin if _is_direct_letta_evals_package_spec(pin) else f"letta-evals=={pin}"


def _requested_git_commit(spec: str) -> Optional[str]:
    if "git+" not in spec:
        return None
    url = spec.split("git+", 1)[1].split("#", 1)[0]
    ref = url.rsplit("@", 1)[-1]
    return ref if re.fullmatch(r"[0-9a-fA-F]{40}", ref) else None


def _node_version_tuple(version: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", version.strip())
    if match is None:
        raise RuntimeProbeError(f"Could not parse Node version: {version!r}")
    return tuple(int(part) for part in match.groups())


def _direct_url_log_summary(direct_url: object) -> object:
    """Keep useful source provenance in logs without URL credentials or queries."""
    if not isinstance(direct_url, dict):
        return direct_url
    raw_url = direct_url.get("url")
    if not isinstance(raw_url, str):
        return direct_url
    parsed = urlsplit(raw_url)
    hostname = parsed.hostname or ""
    netloc = f"{hostname}:{parsed.port}" if parsed.port else hostname
    safe_url = urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    vcs_info = direct_url.get("vcs_info")
    commit = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    return {"url": safe_url, "commit_id": commit}


def _import_modal():
    """Lazy import of the Modal SDK. Raises a friendly error if not installed."""
    try:
        import modal  # type: ignore
    except ImportError as e:
        raise SandboxNotInstalledError(
            "Modal SDK not found. It ships with letta-evals; reinstall with `pip install letta-evals`."
        ) from e
    return modal


def _check_modal_auth() -> None:
    """Pre-flight check for Modal credentials before any network call."""
    token_id = os.getenv("MODAL_TOKEN_ID")
    token_secret = os.getenv("MODAL_TOKEN_SECRET")
    if token_id and token_secret:
        return
    if (Path.home() / ".modal.toml").exists():
        return
    raise SandboxAuthError(
        "Modal authentication not found. Run `modal token new` or set "
        "MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables."
    )


class ModalSandbox(AbstractSandbox):
    """Single-container Modal sandbox driving one sample's worth of work."""

    def __init__(self, spec: ModalSandboxSpec, session_id: str):
        self.spec = spec
        self.session_id = session_id
        self._sandbox = None
        self._app = None
        self._image_id = None

    @property
    def sandbox_id(self) -> Optional[str]:
        if self._sandbox is None:
            return None
        return getattr(self._sandbox, "object_id", None)

    @property
    def image_id(self) -> Optional[str]:
        return self._image_id

    async def start(self) -> None:
        modal = _import_modal()
        _check_modal_auth()

        app = await modal.App.lookup.aio(name=self.spec.app_name, create_if_missing=True)
        if self.spec.image is None:
            # The Dockerfile is a stable OS/toolchain base. Application pins
            # live in literal Modal layer commands so each exact pin becomes
            # part of the layer definition and cannot reuse another version's
            # cached installation.
            dockerfile_path = Path(__file__).parent / "Dockerfile"
            evals_pin = self.spec.letta_evals_version
            code_pin = self.spec.letta_code_version
            assert evals_pin is not None and code_pin is not None  # enforced by ModalSandboxSpec

            evals_package = _letta_evals_package_spec(evals_pin)
            image = modal.Image.from_dockerfile(str(dockerfile_path)).run_commands(
                f"python -m pip install --no-cache-dir --upgrade {shlex.quote(evals_package)}",
                'python -c "from typing_extensions import Sentinel"',
            )
            code_package = f"@letta-ai/letta-code@{code_pin}"
            image = image.run_commands(
                f"npm install -g --omit=dev {shlex.quote(code_package)}",
                "node --version",
                "letta --version",
            )
        else:
            if self.spec.letta_code_version:
                logger.warning(
                    "sandbox.letta_code_version=%s is ignored because a pre-built "
                    "image (%s) is set; the registry image bakes in its own letta-code.",
                    self.spec.letta_code_version,
                    self.spec.image,
                )
            image = modal.Image.from_registry(self.spec.image)

        secrets = [modal.Secret.from_name(name) for name in self.spec.secrets]
        volumes = {
            mount_path: modal.Volume.from_name(name, create_if_missing=False)
            for mount_path, name in self.spec.volumes.items()
        }

        create_kwargs = dict(
            app=app,
            image=image,
            cpu=self.spec.cpu,
            memory=self.spec.memory_mb,
            timeout=self.spec.timeout_sec,
            block_network=self.spec.block_network,
            secrets=secrets,
            volumes=volumes,
        )
        if self.spec.idle_timeout_sec is not None:
            create_kwargs["idle_timeout"] = self.spec.idle_timeout_sec

        self._app = app
        self._sandbox = await modal.Sandbox.create.aio(**create_kwargs)
        self._image_id = getattr(image, "object_id", None)
        logger.info(
            "Started Modal sandbox %s (session=%s, image=%s)",
            self.sandbox_id,
            self.session_id,
            self.image_id,
        )
        try:
            await self._verify_runtime()
        except Exception:
            try:
                await self.stop()
            except Exception as cleanup_error:
                logger.warning("Failed to terminate sandbox after runtime verification error: %s", cleanup_error)
            raise

    async def _verify_runtime(self) -> None:
        probe = await self.exec(_RUNTIME_PROBE_COMMAND)
        if probe.return_code != 0:
            output = (probe.stderr or probe.stdout or "runtime probe returned non-zero").strip()
            raise RuntimeProbeError(f"Modal sandbox runtime probe failed: {output}")
        try:
            runtime = json.loads(probe.stdout)
        except (json.JSONDecodeError, TypeError) as e:
            raise RuntimeProbeError(f"Modal sandbox runtime probe returned invalid JSON: {probe.stdout!r}") from e

        logger.info(
            "Modal image %s runtime: letta-evals=%s direct_url=%s letta-code=%s letta=%r node=%s",
            self.image_id,
            runtime.get("letta_evals_version"),
            _direct_url_log_summary(runtime.get("letta_evals_direct_url")),
            runtime.get("letta_code_version"),
            runtime.get("letta_version_output"),
            runtime.get("node_version"),
        )

        evals_pin = self.spec.letta_evals_version
        if evals_pin:
            expected_commit = _requested_git_commit(evals_pin)
            if expected_commit:
                direct_url = runtime.get("letta_evals_direct_url") or {}
                actual_commit = (direct_url.get("vcs_info") or {}).get("commit_id")
                if not actual_commit or actual_commit.lower() != expected_commit.lower():
                    raise VersionMismatch(
                        f"Sandbox letta-evals commit {actual_commit!r} does not match pinned {expected_commit!r}."
                    )
            elif not _is_direct_letta_evals_package_spec(evals_pin):
                actual_version = runtime.get("letta_evals_version")
                if actual_version != evals_pin:
                    raise VersionMismatch(
                        f"Sandbox letta-evals version {actual_version!r} does not match pinned {evals_pin!r}."
                    )

        if self.spec.image is None:
            actual_code_version = runtime.get("letta_code_version")
            if actual_code_version != self.spec.letta_code_version:
                raise VersionMismatch(
                    f"Sandbox letta-code version {actual_code_version!r} does not match "
                    f"pinned {self.spec.letta_code_version!r}."
                )

        node_version = runtime.get("node_version")
        if self.spec.image is None and (
            not isinstance(node_version, str) or _node_version_tuple(node_version) < (22, 19, 0)
        ):
            raise VersionMismatch(f"Sandbox Node version {node_version!r} does not satisfy >=22.19.0.")

    async def exec(
        self,
        command: str,
        env: Optional[dict[str, str]] = None,
        timeout_sec: Optional[int] = None,
    ) -> ExecResult:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started — call start() first")
        modal = _import_modal()

        exec_secrets = []
        if env:
            exec_secrets.append(modal.Secret.from_dict(env))

        kwargs: dict[str, object] = {"secrets": exec_secrets}
        if timeout_sec is not None:
            kwargs["timeout"] = timeout_sec

        # Shell out so the in-sandbox command can be a full command line.
        process = await self._sandbox.exec.aio("sh", "-c", command, **kwargs)
        # Drain stdout/stderr fully before reading return code so we don't
        # truncate large outputs (matches harbor's pattern).
        stdout = await process.stdout.read.aio()
        stderr = await process.stderr.read.aio()
        return_code = await process.wait.aio()
        return ExecResult(stdout=stdout, stderr=stderr, return_code=return_code)

    async def upload_file(self, local: Path, remote: str) -> None:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started — call start() first")
        # filesystem.copy_from_local streams the file in chunks; it's the
        # supported API now that Sandbox.open()/FileIO are deprecated. The
        # remote parent dir must already exist (it does: /tmp for the suite
        # tarball, /mnt after upload_dir's mkdir for sample.json).
        await self._sandbox.filesystem.copy_from_local.aio(str(local), remote)

    async def upload_dir(
        self,
        local: Path,
        remote: str,
        path_filter: Optional[Callable[[str], bool]] = None,
    ) -> None:
        """Tar up ``local`` on the host, stream into the sandbox, extract at ``remote``.

        Avoids per-file SDK round trips (which add up fast for suites with
        large datasets) by going through a single tar exec call.

        ``path_filter(relpath)`` selects what enters the tarball: it receives
        each member's POSIX path relative to ``local`` and returns True to keep
        it. Dropping a directory prunes its whole subtree (``tarfile`` skips
        recursion into a filtered-out dir). None uploads everything.
        """
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started — call start() first")
        if not local.is_dir():
            raise ValueError(f"upload_dir: local path is not a directory: {local}")

        def _tar_filter(tarinfo: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
            if path_filter is None or tarinfo.name in (".", ""):
                return tarinfo
            relpath = tarinfo.name[2:] if tarinfo.name.startswith("./") else tarinfo.name
            return tarinfo if path_filter(relpath) else None

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tar_path = Path(tmp.name)
        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(local, arcname=".", filter=_tar_filter)

            remote_tar = f"/tmp/{self.session_id}-suite.tar.gz"
            await self.upload_file(tar_path, remote_tar)
            mkdir = await self.exec(f"mkdir -p {shlex.quote(remote)}")
            if mkdir.return_code != 0:
                raise RuntimeError(f"mkdir {remote} failed: {mkdir.stderr}")
            extract = await self.exec(
                f"tar -xzf {shlex.quote(remote_tar)} -C {shlex.quote(remote)} && rm -f {shlex.quote(remote_tar)}"
            )
            if extract.return_code != 0:
                raise RuntimeError(f"tar extract into {remote} failed: {extract.stderr}")
        finally:
            tar_path.unlink(missing_ok=True)

    async def download_file(self, remote: str, local: Path) -> None:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started — call start() first")
        local.parent.mkdir(parents=True, exist_ok=True)
        await self._sandbox.filesystem.copy_to_local.aio(remote, str(local))

    async def stop(self) -> None:
        if self._sandbox is None:
            return
        try:
            await self._sandbox.terminate.aio()
            logger.info("Terminated Modal sandbox %s", self.sandbox_id)
        finally:
            self._sandbox = None
