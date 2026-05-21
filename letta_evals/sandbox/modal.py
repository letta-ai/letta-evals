"""Modal sandbox driver.

One Modal sandbox per sample. Created on :meth:`ModalSandbox.start`, executes
the in-sandbox ``letta-evals run --sample`` invocation, and tears down on
:meth:`ModalSandbox.stop`. Modal SDK is imported lazily so missing
``letta-evals[modal]`` doesn't break the rest of letta-evals.
"""

from __future__ import annotations

import logging
import os
import shlex
import tarfile
import tempfile
from pathlib import Path
from typing import Optional

from letta_evals.models import ModalSandboxSpec
from letta_evals.sandbox.base import AbstractSandbox, ExecResult, SandboxAuthError, SandboxNotInstalledError

logger = logging.getLogger(__name__)


def _import_modal():
    """Lazy import of the Modal SDK. Raises a friendly error if not installed."""
    try:
        import modal  # type: ignore
    except ImportError as e:
        raise SandboxNotInstalledError(
            "Modal SDK not found. Install with `pip install letta-evals[modal]`."
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

    @property
    def sandbox_id(self) -> Optional[str]:
        if self._sandbox is None:
            return None
        return getattr(self._sandbox, "object_id", None)

    async def start(self) -> None:
        modal = _import_modal()
        _check_modal_auth()

        app = await modal.App.lookup.aio(name=self.spec.app_name, create_if_missing=True)
        if self.spec.image is None:
            # Default: build the base image from the Dockerfile bundled with
            # the package. Modal caches the build, so repeated sandboxes
            # don't pay the full build cost — only the first sandbox after
            # the Dockerfile changes does.
            dockerfile_path = Path(__file__).parent / "Dockerfile"
            image = modal.Image.from_dockerfile(str(dockerfile_path))
        else:
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
        logger.info(
            "Started Modal sandbox %s (session=%s, image=%s)",
            self.sandbox_id,
            self.session_id,
            self.spec.image,
        )

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
        with open(local, "rb") as src:
            data = src.read()
        # open.aio(...) is a coroutine returning the FileIO handle; await it
        # before entering the async context manager.
        async with await self._sandbox.open.aio(remote, "wb") as dst:
            await dst.write.aio(data)

    async def upload_dir(self, local: Path, remote: str) -> None:
        """Tar up ``local`` on the host, stream into the sandbox, extract at ``remote``.

        Avoids per-file SDK round trips (which add up fast for suites with
        large datasets) by going through a single tar exec call.
        """
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started — call start() first")
        if not local.is_dir():
            raise ValueError(f"upload_dir: local path is not a directory: {local}")

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tar_path = Path(tmp.name)
        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(local, arcname=".")

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
        async with await self._sandbox.open.aio(remote, "rb") as src:
            data = await src.read.aio()
        local.parent.mkdir(parents=True, exist_ok=True)
        with open(local, "wb") as dst:
            dst.write(data)

    async def stop(self) -> None:
        if self._sandbox is None:
            return
        try:
            await self._sandbox.terminate.aio()
            logger.info("Terminated Modal sandbox %s", self.sandbox_id)
        finally:
            self._sandbox = None
