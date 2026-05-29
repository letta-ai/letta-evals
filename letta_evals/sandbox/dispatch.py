import json
import logging
import os
import shlex
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

from letta_evals.models import Error, Sample, SampleResult, SuiteSpec, Timing
from letta_evals.sandbox.modal import ModalSandbox
from letta_evals.types import ErrorCategory

logger = logging.getLogger(__name__)

# Host env vars auto-forwarded into a Modal sandbox (when present) so the
# in-sandbox target/graders can authenticate and reach the right Letta server
# without pre-creating Modal Secrets. Only these explicit names are forwarded —
# never the whole env. LETTA_BASE_URL is forwarded as a convenience but is only
# used when the suite's target.base_url is unset (target.base_url takes
# precedence). Suite authors extend this via `sandbox.forward_env`.
DEFAULT_SANDBOX_FORWARD_ENV = (
    "LETTA_API_KEY",
    "LETTA_BASE_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "TINKER_API_KEY",
)


def suite_yaml_filename(suite: SuiteSpec) -> str:
    """Locate the suite YAML inside ``base_dir`` so we can reference it remotely.

    Convention: the suite file lives directly inside ``base_dir`` (this is how
    ``run_suite`` sets ``base_dir = suite_path.parent``). Prefer files whose
    name starts with ``suite`` and fall back to the first YAML file.
    """
    base = suite.base_dir
    if base is None:
        return "suite.yaml"
    candidates = sorted(list(base.glob("*.yaml")) + list(base.glob("*.yml")))
    if not candidates:
        return "suite.yaml"
    suite_like = [p for p in candidates if p.name.startswith("suite")]
    chosen = suite_like[0] if suite_like else candidates[0]
    return chosen.name


def sandbox_error_result(
    sample_id, t_sample_start: float, category: ErrorCategory, exception_type: str, message: str
) -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        trajectory=[],
        submissions={},
        grades={},
        timing=Timing(total=time.perf_counter() - t_sample_start, target=0.0),
        error=Error(
            category=category,
            exception_type=exception_type,
            message=message,
        ),
    )


def build_sandbox_command(suite: SuiteSpec, model_handle: Optional[str]) -> str:
    suite_yaml_remote = f"/mnt/suite/{suite_yaml_filename(suite)}"
    cmd_parts = [
        "letta-evals",
        "run",
        suite_yaml_remote,
        "--sample",
        "/mnt/sample.json",
        "--output-json",
        "/mnt/result.json",
    ]
    if isinstance(model_handle, str):
        cmd_parts += ["--model-handle", model_handle]

    inner_command = " ".join(shlex.quote(p) for p in cmd_parts)
    return f"cd /mnt/suite && {inner_command}"


def forwarded_sandbox_env(suite: SuiteSpec) -> Dict[str, str]:
    exec_env: Dict[str, str] = {}
    forward_names = list(DEFAULT_SANDBOX_FORWARD_ENV) + list(suite.sandbox.forward_env or [])
    for name in forward_names:
        value = os.environ.get(name)
        if value is not None:
            exec_env[name] = value
    return exec_env


async def run_sample_in_sandbox(
    suite: SuiteSpec,
    sample: Sample,
    model_handle: Optional[str],
    return_token_data: bool,
    t_sample_start: float,
) -> SampleResult:
    """Dispatch a single sample to a fresh Modal sandbox.

    Creates a sandbox, uploads the whole suite directory tree to /mnt/suite/,
    execs the in-sandbox ``letta-evals run --sample`` invocation, and
    round-trips the resulting SampleResult JSON. The sandbox is torn down in a
    finally block.

    v1 contract: the host only sees the final SampleResult per sample.
    Mid-sample progress callbacks are not emitted. Sandbox-internal stdout and
    stderr surface only on nonzero return code (wrapped into ``Error``).
    """
    sample_id = sample.id
    session_id = f"{suite.name}-{sample_id}-{uuid.uuid4().hex[:8]}"

    if suite.base_dir is None:
        return sandbox_error_result(
            sample_id,
            t_sample_start,
            ErrorCategory.UNKNOWN,
            "SuiteConfigurationError",
            "SuiteSpec.base_dir is unset — required for sandbox execution.",
        )

    sandbox = ModalSandbox(suite.sandbox, session_id=session_id)
    try:
        try:
            await sandbox.start()
        except Exception as e:
            logger.error(f"Sandbox start failed for sample {sample_id}: {e}")
            return sandbox_error_result(
                sample_id,
                t_sample_start,
                ErrorCategory.TARGET,
                type(e).__name__,
                str(e) or type(e).__name__,
            )

        logger.info("Sandbox %s started for sample %s", sandbox.sandbox_id, sample_id)

        if suite.sandbox.letta_evals_version:
            version_check = await sandbox.exec("letta-evals --version")
            expected = suite.sandbox.letta_evals_version
            if version_check.return_code != 0 or expected not in (version_check.stdout + version_check.stderr):
                message = (
                    f"Sandbox image's letta-evals version does not match "
                    f"pinned '{expected}': {version_check.stdout.strip() or version_check.stderr.strip()}"
                )
                return sandbox_error_result(
                    sample_id, t_sample_start, ErrorCategory.UNKNOWN, "VersionMismatch", message
                )

        await sandbox.upload_dir(suite.base_dir, "/mnt/suite")

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            tmp.write(Sample.model_validate(sample.model_dump()).model_dump_json())
            tmp_path = Path(tmp.name)
        try:
            await sandbox.upload_file(tmp_path, "/mnt/sample.json")
        finally:
            tmp_path.unlink(missing_ok=True)

        command = build_sandbox_command(suite, model_handle)
        exec_env = forwarded_sandbox_env(suite)
        if exec_env:
            logger.debug("Forwarding env vars to sandbox %s: %s", sandbox.sandbox_id, sorted(exec_env))

        exec_timeout = suite.sandbox.timeout_sec
        result_exec = await sandbox.exec(command, env=exec_env, timeout_sec=exec_timeout)

        if result_exec.return_code == -1:
            msg = f"Sandbox exec exceeded timeout_sec={exec_timeout}"
            logger.error(
                "Sandbox %s timed out for sample %s after %ss",
                sandbox.sandbox_id,
                sample_id,
                exec_timeout,
            )
            return sandbox_error_result(sample_id, t_sample_start, ErrorCategory.TARGET, "SandboxTimeout", msg)

        if result_exec.return_code != 0:
            msg = (result_exec.stderr or result_exec.stdout or "sandbox exec returned non-zero").strip()
            category = (
                ErrorCategory.GRADING if "grading" in msg.lower() or "rubric" in msg.lower() else ErrorCategory.TARGET
            )
            logger.error(
                "Sandbox %s exec failed for sample %s rc=%d: %s",
                sandbox.sandbox_id,
                sample_id,
                result_exec.return_code,
                msg[:1000],
            )
            return sandbox_error_result(sample_id, t_sample_start, category, "SandboxExecError", msg)

        with tempfile.NamedTemporaryFile("rb", suffix=".json", delete=False) as tmp_r:
            local_result_path = Path(tmp_r.name)
        try:
            try:
                await sandbox.download_file("/mnt/result.json", local_result_path)
            except Exception as e:
                return sandbox_error_result(
                    sample_id,
                    t_sample_start,
                    ErrorCategory.UNKNOWN,
                    "ResultDeserializationError",
                    f"Failed to download result: {e}",
                )

            try:
                with open(local_result_path, "r") as f:
                    result_data = json.load(f)
                sample_result = SampleResult.model_validate(result_data)
            except Exception as e:
                return sandbox_error_result(
                    sample_id,
                    t_sample_start,
                    ErrorCategory.UNKNOWN,
                    "ResultDeserializationError",
                    f"Failed to parse result JSON: {e}",
                )
        finally:
            local_result_path.unlink(missing_ok=True)

        logger.info(
            "Sandbox %s completed sample %s in %.2fs",
            sandbox.sandbox_id,
            sample_id,
            time.perf_counter() - t_sample_start,
        )
        return sample_result
    finally:
        try:
            await sandbox.stop()
        except Exception as e:
            logger.warning("Failed to terminate sandbox %s: %s", sandbox.sandbox_id, e)
