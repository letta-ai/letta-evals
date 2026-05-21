"""Sandbox drivers for per-sample isolated execution.

The runner uses these when ``SuiteSpec.sandbox`` is set: each sample creates
a fresh sandbox via the appropriate driver, runs target + extractors +
graders inside it, then tears it down. Drivers are imported lazily so the
optional Modal SDK dependency does not impact installations that don't use
sandbox execution.
"""

from letta_evals.sandbox.base import AbstractSandbox, ExecResult, SandboxAuthError, SandboxNotInstalledError

__all__ = [
    "AbstractSandbox",
    "ExecResult",
    "SandboxAuthError",
    "SandboxNotInstalledError",
]
