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
from letta_evals.utils import (
    fetch_token_data,
    list_all_agent_messages,
    list_run_ids,
    load_object,
)
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
        """
        self.client = client
        self.model_handle = model_handle
        self.allowed_tools = allowed_tools
        self.disallowed_tools = disallowed_tools
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

                # handle single or multiple inputs
                inputs = sample.input if isinstance(sample.input, list) else [sample.input]

                # for multiple inputs, concatenate with newlines
                prompt = "\n".join(str(inp) for inp in inputs)
                prompt = prompt.replace("{pwd}", self.working_dir.resolve().as_posix())

                # construct the letta-code CLI command (headless streaming JSON output).
                cmd = [
                    "letta",
                    "--yolo",
                    "--output-format",
                    "stream-json",
                    "--model",
                    self.model_handle,
                ]

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

                cmd.extend(["-p", prompt])

                logger.info(f"Running letta command for sample {sample.id}")

                # Prepare environment variables for the subprocess
                # Pass base_url to letta CLI if specified
                env = os.environ.copy()
                if self.base_url:
                    env["LETTA_BASE_URL"] = self.base_url
                    logger.info(f"Setting LETTA_BASE_URL={self.base_url} for letta CLI")

                agent_id = factory_agent_id
                events = []
                stderr_chunks = []

                # Use the agent's memory directory as cwd so Read/Write/Bash tools
                # resolve paths relative to the agent's memory filesystem.
                # Create it if needed — the letta CLI will populate it on startup.
                run_cwd = str(self.working_dir)
                if factory_agent_id:
                    agent_mem_dir = Path.home() / ".letta" / "agents" / factory_agent_id / "memory" / "system"
                    agent_mem_dir.mkdir(parents=True, exist_ok=True)
                    run_cwd = str(agent_mem_dir)

                # run the letta command
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=run_cwd,
                    env=env,
                )

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
                    raise RuntimeError(
                        f"Letta command failed with return code {process.returncode}. Stderr: {stderr_text[:500]}"
                    )

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
                run_ids: Optional[list[str]] = None
                token_data: Optional[list[TurnTokenData]] = None
                if return_token_data and agent_id:
                    run_ids = await list_run_ids(self.client, agent_id)
                    token_data = await fetch_token_data(self.client, run_ids)

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
                    run_ids=run_ids,
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
