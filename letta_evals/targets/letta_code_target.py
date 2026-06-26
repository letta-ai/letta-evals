import asyncio
import json
import logging
import os
import shlex
from pathlib import Path
from typing import Any, Optional

import anyio
from letta_client import AsyncLetta

from letta_evals.models import Sample, TargetResult, TurnTokenData
from letta_evals.targets.errors import TargetError
from letta_evals.utils import list_all_agent_messages, load_object
from letta_evals.visualization.base import ProgressCallback

logger = logging.getLogger(__name__)


class LettaCodeTarget:
    """Letta code target that invokes the letta CLI command."""

    def __init__(
        self,
        client: AsyncLetta,
        model_handle: str,
        allowed_tools: Optional[list[str]] = None,
        disallowed_tools: Optional[list[str]] = None,
        timeout: int = 300,
        max_retries: int = 0,
        base_url: Optional[str] = None,
        agent_script: Optional[str] = None,
        base_dir: Optional[Path] = None,
        flags: Optional[str] = None,
        permission_mode: Optional[str] = None,
    ):
        """Initialize the Letta Code target.

        Args:
            client: AsyncLetta client for retrieving messages after CLI execution
            model_handle: Model handle to use with letta code
            allowed_tools: List of allowed tools (e.g., ["Bash", "Read"])
            disallowed_tools: List of disallowed tools
            timeout: Command timeout in seconds (default: 300)
            max_retries: Number of retry attempts on failure
            base_url: Base URL for the Letta server (passed to CLI via env var)
            agent_script: Path to agent factory script (e.g., "setup_agent.py:setup_agent").
                If provided, the factory function is called to create an agent per sample,
                and --agent is used instead of --new-agent.
            base_dir: Base directory for resolving relative paths in agent_script
            flags: Additional CLI flags to pass to letta code, parsed with shell quoting
                rules (e.g., "--memfs --context-window 8000").
            permission_mode: Permission mode for letta code (e.g., "memory" to scope
                writes to memory roots). When "memory", sets MEMORY_DIR env var.

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
        self.allowed_tools = allowed_tools
        self.disallowed_tools = disallowed_tools
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

    @staticmethod
    def _run_sort_key(run_summary: Any) -> tuple[str, str]:
        """Return a stable chronological sort key for run summaries.

        Letta server defaults for ``runs.list`` have varied across local
        development branches. Token data must be processed oldest-to-newest so
        Tinker sequence-extension can merge consecutive assistant generations.
        Normalize timestamps to strings so mixed SDK/server timestamp types do
        not make sorting fail.
        """
        created_at = getattr(run_summary, "created_at", None)
        if hasattr(created_at, "isoformat"):
            created_key = created_at.isoformat()
        elif created_at is None:
            created_key = ""
        else:
            created_key = str(created_at)
        return created_key, str(getattr(run_summary, "id", ""))

    def _build_subprocess_env(self, sample: Sample, agent_id: Optional[str]) -> dict[str, str]:
        """Build subprocess env: os.environ -> target-managed -> sample.extra_vars["env"]."""
        env = os.environ.copy()

        if self.base_url:
            env["LETTA_BASE_URL"] = self.base_url
            logger.info(f"Setting LETTA_BASE_URL={self.base_url} for letta CLI")

        if self.permission_mode == "memory" and agent_id:
            memory_dir = Path.home() / ".letta" / "agents" / agent_id / "memory"
            memory_dir.mkdir(parents=True, exist_ok=True)
            env["MEMORY_DIR"] = str(memory_dir)

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

        In ``permission_mode == "memory"`` we cd into the memory root so relative
        file paths resolve correctly. An agent factory may point memory operations
        at a different repo than the agent's own MemFS (e.g. a seeded "fake" repo)
        by injecting ``MEMORY_DIR`` via ``sample.extra_vars`` — honor that path so
        the subprocess starts in the repo it will actually read/write. Otherwise
        fall back to the agent's own memory root, or the configured base dir.
        """
        if self.permission_mode == "memory" and agent_id:
            injected_memory_dir = ((sample.extra_vars or {}).get("env") or {}).get("MEMORY_DIR") or (
                sample.extra_vars or {}
            ).get("memory_dir")
            return str(injected_memory_dir or Path.home() / ".letta" / "agents" / agent_id / "memory")
        return str(self.base_dir)

    async def run(
        self,
        sample: Sample,
        progress_callback: Optional[ProgressCallback] = None,
        project_id: Optional[str] = None,
        retrieve_agent_state: bool = False,
        return_token_data: bool = False,
    ) -> TargetResult:
        """Run the letta CLI command on a sample."""
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

                # permission mode (e.g., "memory" to scope writes to MEMORY_DIR)
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

                # Resolve the subprocess cwd (honors a factory-injected MEMORY_DIR
                # in memory permission mode; see _resolve_run_cwd).
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

                # retrieve the full message history using the agent_id
                trajectory = await self._fetch_trajectory(agent_id)

                # Extract usage from the final stream result event (best-effort).
                usage_stats = self._extract_usage_stats(events)

                # Fetch token-level data if requested (for RL training)
                token_data: Optional[list[TurnTokenData]] = None
                if return_token_data and agent_id:
                    token_data = await self._fetch_token_data(agent_id)

                # Retrieve agent state if needed (e.g., for memory block extractors)
                agent_state = None
                if retrieve_agent_state:
                    agent_state = await self.client.agents.retrieve(agent_id=agent_id, include=["agent.blocks"])

                return TargetResult(
                    trajectory=trajectory,
                    agent_id=agent_id,
                    model_handle=self.model_handle,
                    agent_usage=usage_stats,
                    agent_state=agent_state,
                    token_data=token_data,
                )

            except Exception as e:
                last_error = e
                attempt += 1

                if attempt > self.max_retries:
                    timeout_hint = f"Timed out after {self.timeout}s" if isinstance(e, TimeoutError) else ""
                    msg = str(e) or timeout_hint or type(e).__name__
                    err_agent_id = agent_id or factory_agent_id
                    # Best-effort: surface whatever the agent produced before failing —
                    # usage from the stream so far (no agent_id needed), plus the partial
                    # trajectory and token data. agent_id is usually known even on timeout
                    # (captured from the stream init event).
                    partial_usage = self._extract_usage_stats(events)
                    partial_trajectory = []
                    partial_token_data: Optional[list[TurnTokenData]] = None
                    if err_agent_id:
                        try:
                            partial_trajectory = await self._fetch_trajectory(err_agent_id)
                            if return_token_data:
                                partial_token_data = await self._fetch_token_data(err_agent_id)
                        except Exception as fetch_err:
                            logger.warning(f"Could not fetch partial trajectory for agent {err_agent_id}: {fetch_err}")
                    raise TargetError(
                        msg,
                        agent_id=err_agent_id,
                        partial_trajectory=partial_trajectory,
                        agent_usage=partial_usage,
                        token_data=partial_token_data,
                    ) from e

                backoff_time = 2 ** (attempt - 1)
                logger.warning(
                    f"Letta command failed for sample {sample.id} (attempt {attempt}/{self.max_retries + 1}). "
                    f"Agent: {agent_id or factory_agent_id or 'unknown'}. "
                    f"Error: {type(e).__name__}: {str(e)}. Retrying in {backoff_time}s..."
                )
                await anyio.sleep(backoff_time)

        raise last_error or RuntimeError("Unexpected failure in letta command retry loop")

    async def _fetch_trajectory(self, agent_id: str) -> list:
        """Fetch the agent's full message history as a single-turn trajectory.

        Single source of truth for the list + single-turn wrapping, called from
        both the success path and the error path (best-effort) so neither
        reimplements it.
        """
        logger.info(f"Retrieving messages for agent {agent_id}")
        messages = await list_all_agent_messages(self.client, agent_id)
        return [messages] if messages else []

    @staticmethod
    def _extract_usage_stats(events: list) -> Optional[list[dict]]:
        """Pull agent usage_statistics from the final stream ``result`` event.

        stream-json always emits the result event last. Returns ``None`` when no
        usage is present — e.g. the stream was cut short by a crash or timeout —
        so both the success path and the error path report usage the same way.
        """
        if events and events[-1].get("type") == "result" and "usage" in events[-1]:
            usage = events[-1]["usage"]
            return [
                {
                    "message_type": "usage_statistics",
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "cached_input_tokens": usage.get("cached_input_tokens", 0),
                    "cache_write_tokens": usage.get("cache_write_tokens", 0),
                    "reasoning_tokens": usage.get("reasoning_tokens", 0),
                }
            ]
        return None

    async def _fetch_token_data(self, agent_id: str) -> list[TurnTokenData]:
        """Fetch token-level data (IDs + logprobs) for a letta code agent.

        Retrieves messages with ``return_token_ids=True`` to get
        ``output_ids`` and ``output_token_logprobs`` per message.

        Stops at the first half-written turn — ``output_ids`` present but a
        shorter ``output_token_logprobs`` — returning only the clean prefix, so a
        partially-flushed generation can't corrupt Tinker's sequence-extension.
        """
        token_data: list[TurnTokenData] = []
        try:
            # Fetch ALL runs for this agent — client tools cause each tool-call
            # round-trip to be a separate run, so token IDs are scattered.
            try:
                runs_page = await self.client.runs.list(agent_id=agent_id, limit=100, order="asc")
            except TypeError:
                # Older generated clients may not expose the ``order`` kwarg.
                # Fall back to the legacy call and sort locally below.
                runs_page = await self.client.runs.list(agent_id=agent_id, limit=100)
            if not runs_page.items:
                return token_data

            # Token IDs are stored in run.metadata.result.turns (populated by SGLang native adapter)
            for run_summary in sorted(runs_page.items, key=self._run_sort_key):
                run = await self.client.runs.retrieve(run_id=run_summary.id)
                result = (run.metadata or {}).get("result", {})
                for turn in result.get("turns") or []:
                    output_ids = turn.get("output_ids")
                    role = turn.get("role", "assistant")
                    if output_ids:
                        logprobs = turn.get("output_token_logprobs")
                        if logprobs is not None and len(output_ids) != len(logprobs):
                            # Half-written generation: ids present but logprobs
                            # not fully flushed. Drop it and everything after.
                            logger.info(
                                f"Truncating token data at half-written turn in run {run_summary.id} "
                                f"for agent {agent_id}"
                            )
                            return token_data
                        # Assistant turn with token IDs from SGLang
                        token_data.append(
                            TurnTokenData(
                                role=role,
                                input_ids=turn.get("input_ids"),
                                output_ids=output_ids,
                                output_token_logprobs=logprobs,
                            )
                        )
                    elif role in ("tool", "tool_return", "tool_return_message") and turn.get("content"):
                        # Tool return turn — no output_ids, but content is needed
                        # for proper multi-turn token sequence reconstruction
                        token_data.append(
                            TurnTokenData(
                                role=role,
                                content=turn.get("content"),
                            )
                        )
        except Exception as e:
            logger.warning(f"Could not fetch token data for agent {agent_id}: {e}")
        return token_data
