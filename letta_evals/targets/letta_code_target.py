import asyncio
import json
import logging
import os
import shlex
from pathlib import Path
from typing import Optional

import anyio
from letta_client import AsyncLetta

from letta_evals.models import Sample, TargetResult, TurnTokenData
from letta_evals.targets.base import AbstractAgentTarget, TargetError
from letta_evals.utils import list_all_agent_messages, load_object
from letta_evals.visualization.base import ProgressCallback

logger = logging.getLogger(__name__)


class LettaCodeTarget(AbstractAgentTarget):
    """Letta code target that invokes the letta CLI command."""

    def __init__(
        self,
        client: AsyncLetta,
        model_handle: str = "anthropic/claude-sonnet-4-5-20250929",
        working_dir: Optional[Path] = None,
        sandbox: bool = True,
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
            working_dir: Working directory for letta command execution
            sandbox: If True, create a per-model subdirectory under working_dir
                for isolated execution. If False, use working_dir directly.
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
        self.base_dir = base_dir or Path.cwd()
        self.flags = shlex.split(flags) if flags else []

        # Resolve the working directory, optionally creating a per-model sandbox
        wd_base = working_dir or Path.cwd()
        if sandbox:
            model_name = model_handle.split("/")[-1]
            self.working_dir = wd_base / model_name
        else:
            self.working_dir = wd_base
        self.working_dir.mkdir(parents=True, exist_ok=True)

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
                            sample.id, agent_id=factory_agent_id, model_name=self.model_handle
                        )

                # construct prompt after factory call so agent_factory can modify sample if needed
                # handle single or multiple inputs
                inputs = sample.input if isinstance(sample.input, list) else [sample.input]

                # for multiple inputs, concatenate with newlines
                prompt = "\n".join(str(inp) for inp in inputs)
                prompt = prompt.replace("{pwd}", self.working_dir.resolve().as_posix())

                if factory_agent_id:
                    cmd.extend(["--agent", factory_agent_id])
                else:
                    cmd.append("--new-agent")

                # Use codex system prompt for GPT-style models (matches `letta --help` examples)
                if "gpt" in self.model_handle:
                    cmd.extend(["--system", "codex"])

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

                # Prepare environment variables for the subprocess
                # Pass base_url to letta CLI if specified
                env = os.environ.copy()
                if self.base_url:
                    env["LETTA_BASE_URL"] = self.base_url
                    logger.info(f"Setting LETTA_BASE_URL={self.base_url} for letta CLI")

                agent_id = factory_agent_id
                events = []
                stderr_chunks = []

                # When using memory permission mode, set MEMORY_DIR and cwd to the
                # agent's memory root so relative file paths resolve correctly.
                if self.permission_mode == "memory" and factory_agent_id:
                    memory_dir = Path.home() / ".letta" / "agents" / factory_agent_id / "memory"
                    memory_dir.mkdir(parents=True, exist_ok=True)
                    env["MEMORY_DIR"] = str(memory_dir)
                    run_cwd = str(memory_dir)
                else:
                    run_cwd = str(self.working_dir)

                # run the letta command with prompt piped via stdin
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=run_cwd,
                    env=env,
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
                                            sample.id, agent_id=agent_id, model_name=self.model_handle
                                        )
                                if progress_callback and agent_id:
                                    await progress_callback.message_sending(
                                        sample.id, 1, len(inputs), agent_id=agent_id, model_name=self.model_handle
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
                    logger.error(f"Letta command failed with return code {process.returncode}")
                    logger.error(f"Stderr: {stderr_text}")

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
                logger.info(f"Retrieving messages for agent {agent_id}")

                messages = await list_all_agent_messages(self.client, agent_id)

                # wrap messages in a single turn
                trajectory = [messages] if messages else []

                # Extract usage from the final result event.
                # stream-json always emits the result event last.
                usage_stats = []
                if events and events[-1].get("type") == "result" and "usage" in events[-1]:
                    usage = events[-1]["usage"]
                    usage_stats.append(
                        {
                            "message_type": "usage_statistics",
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                            "cached_input_tokens": usage.get("cached_input_tokens", 0),
                            "cache_write_tokens": usage.get("cache_write_tokens", 0),
                            "reasoning_tokens": usage.get("reasoning_tokens", 0),
                        }
                    )

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
                    model_name=self.model_handle,
                    agent_usage=usage_stats if usage_stats else None,
                    agent_state=agent_state,
                    token_data=token_data,
                )

            except Exception as e:
                last_error = e
                attempt += 1

                if attempt > self.max_retries:
                    logger.error(
                        f"Failed to run letta command for sample {sample.id} after {self.max_retries} retries. "
                        f"Agent: {agent_id or factory_agent_id or 'unknown'}. "
                        f"Final error: {type(e).__name__}: {str(e)}"
                    )
                    timeout_hint = f"Timed out after {self.timeout}s" if isinstance(e, TimeoutError) else ""
                    msg = str(e) or timeout_hint or type(e).__name__
                    raise TargetError(msg, agent_id=agent_id or factory_agent_id) from e

                backoff_time = 2 ** (attempt - 1)
                logger.warning(
                    f"Letta command failed for sample {sample.id} (attempt {attempt}/{self.max_retries + 1}). "
                    f"Agent: {agent_id or factory_agent_id or 'unknown'}. "
                    f"Error: {type(e).__name__}: {str(e)}. Retrying in {backoff_time}s..."
                )
                await anyio.sleep(backoff_time)

        raise last_error or RuntimeError("Unexpected failure in letta command retry loop")

    async def _fetch_token_data(self, agent_id: str) -> list[TurnTokenData]:
        """Fetch token-level data (IDs + logprobs) for a letta code agent.

        Retrieves messages with ``return_token_ids=True`` to get
        ``output_ids`` and ``output_token_logprobs`` per message.
        """
        token_data: list[TurnTokenData] = []
        try:
            # Fetch ALL runs for this agent — client tools cause each tool-call
            # round-trip to be a separate run, so token IDs are scattered.
            runs_page = await self.client.runs.list(agent_id=agent_id, limit=100)
            if not runs_page.items:
                return token_data

            # Token IDs are stored in run.metadata.result.turns (populated by SGLang native adapter)
            for run_summary in runs_page.items:
                run = await self.client.runs.retrieve(run_id=run_summary.id)
                result = (run.metadata or {}).get("result", {})
                for turn in result.get("turns") or []:
                    output_ids = turn.get("output_ids")
                    role = turn.get("role", "assistant")
                    if output_ids:
                        # Assistant turn with token IDs from SGLang
                        token_data.append(
                            TurnTokenData(
                                role=role,
                                output_ids=output_ids,
                                output_token_logprobs=turn.get("output_token_logprobs"),
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
