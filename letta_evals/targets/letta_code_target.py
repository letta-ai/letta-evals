import asyncio
import json
import logging
import os
import shlex
from pathlib import Path
from typing import Optional

import anyio
from letta_client import AsyncLetta

from letta_evals.execution.trace import extract_usage_stats
from letta_evals.models import Sample, TargetResult
from letta_evals.targets.errors import TargetError
from letta_evals.utils import load_object
from letta_evals.visualization.base import ProgressCallback

logger = logging.getLogger(__name__)


class LettaCodeTarget:
    """Letta code target that invokes the letta CLI command."""

    def __init__(
        self,
        client: AsyncLetta,
        model_handle: str,
        timeout: int = 300,
        max_retries: int = 0,
        base_url: Optional[str] = None,
        agent_script: Optional[str] = None,
        base_dir: Optional[Path] = None,
        flags: Optional[str] = None,
        permission_mode: Optional[str] = None,
        memory_workspace: bool = False,
        memory_dir: Optional[Path] = None,
    ):
        """Initialize the Letta Code target.

        Args:
            client: AsyncLetta client for retrieving messages after CLI execution
            model_handle: Model handle to use with letta code
            timeout: Command timeout in seconds (default: 300)
            max_retries: Number of retry attempts on failure
            base_url: Base URL for the Letta server (passed to CLI via env var)
            agent_script: Path to agent factory script (e.g., "setup_agent.py:setup_agent").
                If provided, the factory function is called to create an agent per sample,
                and --agent is used instead of --new-agent.
            base_dir: Base directory for resolving relative paths in agent_script
            flags: Additional CLI flags to pass to letta code, parsed with shell quoting
                rules (e.g., "--memfs --context-window 8000").
            permission_mode: Permission mode to pass through to Letta Code.
                Use a current Letta Code CLI mode such as "unrestricted",
                "standard", or "acceptEdits".
            memory_workspace: If true, configure MEMORY_DIR/LETTA_MEMORY_DIR
                and run the subprocess from that memory workspace when one can
                be resolved. This controls letta-evals workspace behavior only;
                use sandboxing if strict filesystem confinement is required.
            memory_dir: Optional explicit memory workspace root. If unset,
                memory_workspace uses a per-sample override or the factory-created
                agent's default ~/.letta/agents/<agent_id>/memory root.

        Agent factories may set ``sample.extra_vars["env"]`` (dict) to inject
        per-sample env vars into the subprocess; user keys win over target-managed
        ones. See ``_build_subprocess_env``.

        The CLI runs in the process's current working directory. For isolated
        execution, configure ``suite.sandbox`` to dispatch each sample to a
        fresh Modal sandbox — the image's ``WORKDIR`` becomes the per-sample
        working directory.
        """
        self.client = client
        self.model_handle = model_handle
        if permission_mode == "memory":
            raise ValueError(
                "permission_mode='memory' was removed from Letta Code. "
                "Use memory_workspace=True with a current permission_mode such as 'unrestricted'."
            )
        self.memory_workspace = memory_workspace
        self.permission_mode = permission_mode
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = base_url
        self.agent_script = agent_script
        # base_dir is the suite directory (set by SuiteSpec.from_yaml). It
        # resolves agent_script paths, the `{pwd}` prompt placeholder, and the
        # letta-code subprocess cwd. Sandbox isolation is handled by
        # suite.sandbox (Modal), which runs the in-sandbox CLI from /mnt/suite.
        self.base_dir = base_dir or Path.cwd()
        self.flags = shlex.split(flags) if flags else []
        self.memory_dir = memory_dir

    def _resolve_memory_workspace_dir(self, sample: Sample, agent_id: Optional[str]) -> Optional[Path]:
        """Resolve the memory workspace root for env/cwd configuration."""
        if not self.memory_workspace:
            return None

        sample_env = (sample.extra_vars or {}).get("env") or {}
        if isinstance(sample_env, dict):
            injected = sample_env.get("MEMORY_DIR") or sample_env.get("LETTA_MEMORY_DIR")
            if injected:
                return Path(str(injected)).expanduser()

        sample_memory_dir = (sample.extra_vars or {}).get("memory_dir")
        if sample_memory_dir:
            return Path(str(sample_memory_dir)).expanduser()

        if self.memory_dir:
            return self.memory_dir.expanduser()

        if agent_id:
            return Path.home() / ".letta" / "agents" / agent_id / "memory"

        return None

    def _build_subprocess_env(self, sample: Sample, agent_id: Optional[str]) -> dict[str, str]:
        """Build subprocess env: os.environ -> target-managed -> sample.extra_vars["env"]."""
        env = os.environ.copy()

        if self.base_url:
            env["LETTA_BASE_URL"] = self.base_url
            logger.info(f"Setting LETTA_BASE_URL={self.base_url} for letta CLI")

        memory_dir = self._resolve_memory_workspace_dir(sample, agent_id)
        if memory_dir:
            memory_dir.mkdir(parents=True, exist_ok=True)
            env["MEMORY_DIR"] = str(memory_dir)
            env["LETTA_MEMORY_DIR"] = str(memory_dir)

        sample_env = (sample.extra_vars or {}).get("env")
        if sample_env is not None:
            if not isinstance(sample_env, dict):
                raise TargetError(
                    f"sample.extra_vars['env'] must be a dict, got {type(sample_env).__name__} for sample {sample.id}",
                    agent_id=agent_id,
                )
            applied = []
            for k, v in sample_env.items():
                if not isinstance(k, str):
                    raise TargetError(
                        f"sample.extra_vars['env'] keys must be strings, got "
                        f"{type(k).__name__} ({k!r}) for sample {sample.id}",
                        agent_id=agent_id,
                    )
                # Coerce to str — env values must be strings; None becomes empty.
                env[k] = "" if v is None else str(v)
                applied.append(k)
            if applied:
                logger.info(f"Applied per-sample env overrides for sample {sample.id}: keys={sorted(applied)}")

        return env

    def _resolve_run_cwd(self, sample: Sample, agent_id: Optional[str]) -> str:
        """Resolve the subprocess working directory.

        When ``memory_workspace`` is true, run from the resolved memory root so
        relative file paths resolve there. Agent factories may point memory
        operations at a seeded repo by injecting ``MEMORY_DIR`` or
        ``memory_dir`` via ``sample.extra_vars``; otherwise an explicit target
        ``memory_dir`` or the factory-created agent's own memory root is used.
        """
        memory_dir = self._resolve_memory_workspace_dir(sample, agent_id)
        if memory_dir:
            return str(memory_dir)
        return str(self.base_dir)

    async def run(
        self,
        sample: Sample,
        progress_callback: Optional[ProgressCallback] = None,
        project_id: Optional[str] = None,
    ) -> TargetResult:
        """Run the letta CLI command on a sample and return execution metadata.

        The target deliberately does not fetch server-side trace fields such as
        messages, agent state, or token data. ``Runner`` owns those fetches so
        in-process and sandboxed runs share the same trace assembly path.
        """
        attempt = 0
        last_error = None

        while attempt <= self.max_retries:
            try:
                agent_id = None
                # Initialized up front so the error handler can read whatever
                # streamed before a failure (e.g. for best-effort usage).
                events = []

                # construct the letta-code CLI command (headless streaming JSON output).
                cmd = [
                    "letta",
                    "--output-format",
                    "stream-json",
                    "--model",
                    self.model_handle,
                ]

                # Only use --yolo when no explicit permission_mode is set,
                # since --yolo overrides --permission-mode (sets bypassPermissions).
                if not self.permission_mode:
                    cmd.append("--yolo")

                # If agent_script is provided, create agent via factory first
                factory_agent_id = None
                if self.agent_script:
                    agent_factory_func = load_object(self.agent_script, self.base_dir)
                    factory_agent_id = await agent_factory_func(self.client, sample)
                    logger.info(f"Created agent {factory_agent_id} via agent_factory for sample {sample.id}")
                    if progress_callback:
                        await progress_callback.agent_created(
                            sample.id, agent_id=factory_agent_id, model_handle=self.model_handle
                        )

                # construct prompt after factory call so agent_factory can modify sample if needed
                # handle single or multiple inputs
                inputs = sample.input if isinstance(sample.input, list) else [sample.input]

                # for multiple inputs, concatenate with newlines
                prompt = "\n".join(str(inp) for inp in inputs)
                prompt = prompt.replace("{pwd}", self.base_dir.resolve().as_posix())

                if factory_agent_id:
                    cmd.extend(["--agent", factory_agent_id])
                else:
                    cmd.append("--new-agent")

                # append any extra flags from suite config
                if self.flags:
                    cmd.extend(self.flags)

                # permission mode passed through to Letta Code. Memory workspace
                # behavior is handled by letta-evals and is not a CLI permission mode.
                if self.permission_mode:
                    cmd.extend(["--permission-mode", self.permission_mode])

                # Pass prompt via stdin to avoid OS ARG_MAX limits on large inputs.
                # letta-code headless mode reads from stdin when no positional prompt given.
                cmd.append("-p")
                prompt_bytes = prompt.encode("utf-8")

                logger.info(
                    f"Running letta command for sample {sample.id} (prompt via stdin, {len(prompt_bytes)} bytes)"
                )

                # Prepare environment variables for the subprocess.
                # Layers target-managed env (LETTA_BASE_URL, MEMORY_DIR) and per-sample
                # overrides from sample.extra_vars["env"] (see class docstring).
                env = self._build_subprocess_env(sample, factory_agent_id)

                agent_id = factory_agent_id
                stderr_chunks = []

                # Resolve the subprocess cwd (honors memory_workspace settings;
                # see _resolve_run_cwd).
                run_cwd = self._resolve_run_cwd(sample, factory_agent_id)

                # run the letta command with prompt piped via stdin
                #
                # NOTE: We bump the StreamReader limit well above asyncio's
                # 64 KiB default. letta-code emits one JSON event per line in
                # stream-json mode, and a single event (e.g. a Read of a large
                # file, verbose Bash output, or a base64-encoded image) can
                # easily exceed 64 KiB. Without this, `async for raw_line in
                # process.stdout` below raises LimitOverrunError on long-tool-
                # result turns and crashes the rollout.
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=run_cwd,
                    env=env,
                    limit=16 * 1024 * 1024,
                )
                # Write prompt to stdin and close it
                process.stdin.write(prompt_bytes)
                await process.stdin.drain()
                process.stdin.close()
                await process.stdin.wait_closed()

                # Read streaming JSON output line by line, capturing agent_id
                # from the init event as soon as it arrives. This ensures we
                # have the agent_id even if the process later times out.
                async def _read_stdout():
                    nonlocal agent_id
                    async for raw_line in process.stdout:
                        line = raw_line.decode().strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                            events.append(event)
                            if event.get("type") == "system" and event.get("subtype") == "init":
                                if not agent_id:
                                    agent_id = event.get("agent_id")
                                    logger.info(f"Captured agent_id {agent_id} from stream init event")
                                    if progress_callback and agent_id:
                                        await progress_callback.agent_created(
                                            sample.id, agent_id=agent_id, model_handle=self.model_handle
                                        )
                                if progress_callback and agent_id:
                                    await progress_callback.message_sending(
                                        sample.id, 1, len(inputs), agent_id=agent_id, model_handle=self.model_handle
                                    )
                        except json.JSONDecodeError:
                            logger.warning(f"Non-JSON stream output: {line[:200]}")

                async def _read_stderr():
                    async for raw_line in process.stderr:
                        stderr_chunks.append(raw_line.decode())

                try:
                    await asyncio.wait_for(
                        asyncio.gather(_read_stdout(), _read_stderr(), process.wait()),
                        timeout=self.timeout,
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    raise RuntimeError(f"Letta command timed out after {self.timeout} seconds")

                stderr_text = "".join(stderr_chunks)

                if process.returncode != 0:
                    # Surface the last stdout event so the real error isn't lost
                    last_stdout_event = ""
                    for ev in reversed(events):
                        try:
                            last_stdout_event = json.dumps(ev)[:500]
                        except Exception:
                            last_stdout_event = str(ev)[:500]
                        break

                    parts = [f"Letta command failed with return code {process.returncode}"]
                    if last_stdout_event:
                        parts.append(f"Last stdout event: {last_stdout_event}")
                    if stderr_text:
                        parts.append(f"Stderr: {stderr_text[:500]}")
                    raise RuntimeError(". ".join(parts))

                if not agent_id:
                    raise RuntimeError("No agent_id found in letta stream output")

                # Extract usage from the final stream result event (best-effort).
                usage_stats = extract_usage_stats(events)

                return TargetResult(
                    agent_id=agent_id,
                    model_handle=self.model_handle,
                    agent_usage=usage_stats,
                )

            except Exception as e:
                last_error = e
                attempt += 1

                if attempt > self.max_retries:
                    timeout_hint = f"Timed out after {self.timeout}s" if isinstance(e, TimeoutError) else ""
                    msg = str(e) or timeout_hint or type(e).__name__
                    err_agent_id = agent_id or factory_agent_id
                    # Best-effort: surface usage from the stream so far. Any
                    # server-side trace fields (partial trajectory/token data) are
                    # fetched by Runner from the agent_id when available.
                    partial_usage = extract_usage_stats(events)
                    raise TargetError(
                        msg,
                        agent_id=err_agent_id,
                        agent_usage=partial_usage,
                    ) from e

                backoff_time = 2 ** (attempt - 1)
                logger.warning(
                    f"Letta command failed for sample {sample.id} (attempt {attempt}/{self.max_retries + 1}). "
                    f"Agent: {agent_id or factory_agent_id or 'unknown'}. "
                    f"Error: {type(e).__name__}: {str(e)}. Retrying in {backoff_time}s..."
                )
                await anyio.sleep(backoff_time)

        raise last_error or RuntimeError("Unexpected failure in letta command retry loop")
