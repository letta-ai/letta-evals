"""Abstract sandbox driver interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ExecResult:
    """Result of executing a command inside a sandbox."""

    stdout: str
    stderr: str
    return_code: int


class SandboxNotInstalledError(ImportError):
    """Raised when the SDK for the configured sandbox backend isn't importable."""


class SandboxAuthError(RuntimeError):
    """Raised when the sandbox backend is reachable but auth credentials are missing."""


class AbstractSandbox(ABC):
    """One unit of work: target + extractors + graders for a single sample."""

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def exec(
        self,
        command: str,
        env: Optional[dict[str, str]] = None,
        timeout_sec: Optional[int] = None,
    ) -> ExecResult: ...

    @abstractmethod
    async def upload_file(self, local: Path, remote: str) -> None: ...

    @abstractmethod
    async def upload_dir(self, local: Path, remote: str) -> None: ...

    @abstractmethod
    async def download_file(self, remote: str, local: Path) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @property
    @abstractmethod
    def sandbox_id(self) -> Optional[str]: ...
