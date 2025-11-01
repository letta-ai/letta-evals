import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from letta_client import AsyncLetta

from letta_evals.models import Sample, TargetResult
from letta_evals.targets.base import AbstractAgentTarget
from letta_evals.visualization.base import ProgressCallback

logger = logging.getLogger(__name__)


class LettaCodeTarget(AbstractAgentTarget):
    """Letta code target that invokes the letta CLI command."""

    def __init__(
        self,
        client: AsyncLetta,
        model_handle: str = "anthropic/claude-sonnet-4-5-20250929",
        working_dir: Optional[Path] = None,
        allowed_tools: Optional[list[str]] = None,
        disallowed_tools: Optional[list[str]] = None,
        timeout: int = 300,
        max_retries: int = 0,
    ):
        """Initialize the Letta Code target.

        Args:
            client: AsyncLetta client for retrieving messages after CLI execution
            model_handle: Model handle to use with letta code
            working_dir: Working directory for letta command execution
            allowed_tools: List of allowed tools (e.g., ["Bash", "Read"])
            disallowed_tools: List of disallowed tools
            timeout: Command timeout in seconds (default: 300)
            max_retries: Number of retry attempts on failure
        """
        self.client = client
        self.model_handle = model_handle
        self.working_dir = working_dir or Path.cwd()
        self.allowed_tools = allowed_tools
        self.disallowed_tools = disallowed_tools
        self.timeout = timeout
        self.max_retries = max_retries

    async def run(
        self,
        sample: Sample,
        progress_callback: Optional[ProgressCallback] = None,
        project_id: Optional[str] = None,
        retrieve_agent_state: bool = False,
    ) -> TargetResult:
        """Run the letta CLI command on a sample."""
        attempt = 0
        last_error = None

        while attempt <= self.max_retries:
            try:
                # handle single or multiple inputs
                inputs = sample.input if isinstance(sample.input, list) else [sample.input]

                if progress_callback:
                    await progress_callback.message_sending(sample.id, 1, len(inputs), model_name=self.model_handle)

                # for multiple inputs, concatenate with newlines
                prompt = "\n".join(str(inp) for inp in inputs)

                # construct the letta command with json output
                cmd = [
                    "letta",
                    "--new",
                    "--yolo",
                    "--output-format",
                    "json",
                    "--model",
                    self.model_handle,
                    "-p",
                    prompt,
                ]

                # add tool permissions if specified
                if self.allowed_tools:
                    cmd.extend(["--allowedTools", ",".join(self.allowed_tools)])
                if self.disallowed_tools:
                    cmd.extend(["--disallowedTools", ",".join(self.disallowed_tools)])

                logger.info(f"Running letta command for sample {sample.id}")

                # run the letta command
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.working_dir),
                )

                # wait for completion
                try:
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    raise RuntimeError(f"Letta command timed out after {self.timeout} seconds")

                stdout_text = stdout.decode() if stdout else ""
                stderr_text = stderr.decode() if stderr else ""

                if process.returncode != 0:
                    logger.error(f"Letta command failed with return code {process.returncode}")
                    logger.error(f"Stderr: {stderr_text}")
                    raise RuntimeError(
                        f"Letta command failed with return code {process.returncode}. Stderr: {stderr_text[:500]}"
                    )

                # parse the json output
                try:
                    result = json.loads(stdout_text)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON output: {stdout_text[:500]}")
                    raise RuntimeError(f"Failed to parse JSON output from letta command: {e}")

                # extract agent_id and other metadata
                agent_id = result.get("agent_id")
                if not agent_id:
                    raise RuntimeError(f"No agent_id found in letta output: {result}")

                # retrieve the full message history using the agent_id
                logger.info(f"Retrieving messages for agent {agent_id}")

                # retrieve messages from the agent's last run
                messages = await self.client.agents.messages.list(agent_id=agent_id)

                # wrap messages in a single turn
                trajectory = [messages] if messages else []

                # extract usage stats if available
                usage_stats = []
                if "usage" in result:
                    usage_stats.append(
                        {
                            "input_tokens": result["usage"].get("input_tokens", 0),
                            "output_tokens": result["usage"].get("output_tokens", 0),
                        }
                    )

                return TargetResult(
                    trajectory=trajectory,
                    agent_id=agent_id,
                    model_name=self.model_handle,
                    agent_usage=usage_stats if usage_stats else None,
                    agent_state=None,
                )

            except Exception as e:
                last_error = e
                attempt += 1

                if attempt > self.max_retries:
                    logger.error(
                        f"Failed to run letta command for sample {sample.id} after {self.max_retries} retries. "
                        f"Final error: {type(e).__name__}: {str(e)}"
                    )
                    raise

                backoff_time = 2 ** (attempt - 1)
                logger.warning(
                    f"Letta command failed for sample {sample.id} (attempt {attempt}/{self.max_retries + 1}). "
                    f"Error: {type(e).__name__}: {str(e)}. Retrying in {backoff_time}s..."
                )
                await asyncio.sleep(backoff_time)

        raise last_error or RuntimeError("Unexpected failure in letta command retry loop")
