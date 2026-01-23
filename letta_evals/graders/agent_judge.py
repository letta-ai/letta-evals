import asyncio
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from letta_client import AsyncLetta
from letta_client.types import MessageCreateParam
from letta_client.types.agents import ToolCall, ToolCallMessage

from letta_evals.extractors import extractor_requires_agent_state, get_extractor
from letta_evals.graders.base import Grader
from letta_evals.graders.prompt_utils import build_judge_prompt
from letta_evals.models import AgentState, GradeResult, LettaMessageUnion, Sample


class AgentJudgeGrader(Grader):
    """Grader that uses a Letta agent as a judge with custom rubric prompts."""

    def __init__(
        self,
        prompt: str,
        client: AsyncLetta,
        agent_file: Optional[Path] = None,
        agent_id: Optional[str] = None,
        project_id: Optional[str] = None,
        judge_tool_name: str = "submit_grade",
        extractor: str = "last_assistant",
        extractor_config: Optional[dict] = None,
        base_dir: Optional[Path] = None,
        rubric_vars: Optional[List[str]] = None,
        judge_target_kind: str = "letta_agent",
        working_dir: Optional[Path] = None,
        model_handle: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        if not agent_file and not agent_id:
            raise ValueError("Either agent_file or agent_id must be provided")
        if agent_file and agent_id:
            raise ValueError("Cannot provide both agent_file and agent_id")

        self.agent_file = agent_file
        self.agent_id = agent_id
        self.prompt = prompt
        self.client = client
        self.project_id = project_id
        self.judge_tool_name = judge_tool_name
        self.extractor_name = extractor
        self.base_dir = base_dir
        self.rubric_vars = rubric_vars or []
        self.extractor = get_extractor(extractor, extractor_config, base_dir=base_dir)
        self._requires_agent_state = extractor_requires_agent_state(extractor, base_dir=base_dir)
        self.judge_target_kind = judge_target_kind
        self.working_dir = working_dir
        self.model_handle = model_handle
        self.base_url = base_url

        # validate agent file contains the required tool with correct schema (only if agent_file is provided)
        if self.agent_file:
            self._validate_agent_file()

    @property
    def requires_agent_state(self) -> bool:
        """Whether this grader's extractor requires agent_state."""
        return self._requires_agent_state

    async def grade(
        self, sample: Sample, trajectory: List[List[LettaMessageUnion]], agent_state: Optional[AgentState] = None
    ) -> Tuple[GradeResult, str]:
        """Grade using agent judge with rubric."""
        # Validate trajectory before extraction
        if not trajectory or not any(turn for turn in trajectory if turn):
            return GradeResult(score=0.0, rationale="Empty trajectory - agent produced no messages"), ""

        submission = self.extractor(trajectory, agent_state=agent_state)

        # Validate submission after extraction
        if not submission:
            return GradeResult(score=0.0, rationale="Empty submission - extractor found no content"), ""

        judge_prompt = build_judge_prompt(
            self.prompt, sample, submission, self.rubric_vars, judge_tool_name=self.judge_tool_name
        )

        judge_agent_id = None
        try:
            if self.judge_target_kind == "letta_code":
                # Run judge using letta-code CLI
                score, rationale, judge_agent_id = await self._grade_with_letta_code(judge_prompt)
            else:
                # Run judge using API (default)
                if self.agent_id:
                    # Use provided agent_id directly
                    judge_agent_id = self.agent_id
                else:
                    # load judge agent from .af file
                    with open(self.agent_file, "rb") as f:
                        resp = await self.client.agents.import_file(
                            file=f, append_copy_suffix=False, override_existing_tools=False, project_id=self.project_id
                        )
                        if len(resp.agent_ids) > 1:
                            raise RuntimeError(
                                f"Expected single judge agent from .af file, got {len(resp.agent_ids)} agents"
                            )

                        judge_agent_id = resp.agent_ids[0]

                # send prompt to judge agent
                stream = await self.client.agents.messages.stream(
                    agent_id=judge_agent_id,
                    messages=[MessageCreateParam(role="user", content=judge_prompt)],
                    stream_tokens=False,
                )

                # consume stream
                run_id = None
                async for chunk in stream:
                    if hasattr(chunk, "run_id"):
                        run_id = chunk.run_id

                if not run_id:
                    raise RuntimeError("No run_id received from judge agent stream")

                messages_page = await self.client.runs.messages.list(run_id=run_id)
                score, rationale = self._parse_tool_calls(messages_page.items)

            metadata = {"judge_agent_id": judge_agent_id}
            if self.agent_file:
                metadata["agent_file"] = str(self.agent_file)
            if self.agent_id:
                metadata["agent_id"] = self.agent_id

            return GradeResult(
                score=score,
                rationale=rationale,
                metadata=metadata,
            ), submission

        except Exception as e:
            metadata = {"error": str(e)}
            if self.agent_file:
                metadata["agent_file"] = str(self.agent_file)
            if self.agent_id:
                metadata["agent_id"] = self.agent_id

            return GradeResult(
                score=0.0,
                rationale=f"Error during agent judge grading: {str(e)}",
                metadata=metadata,
            ), submission

    def _validate_agent_file(self) -> None:
        """Validate that the agent file contains the required tool with correct schema.

        Raises:
            FileNotFoundError: If agent file doesn't exist
            ValueError: If tool is missing or has incorrect schema
        """
        if not self.agent_file.exists():
            raise FileNotFoundError(f"Agent file not found: {self.agent_file}")

        # load and parse agent file
        try:
            with open(self.agent_file, "r") as f:
                agent_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in agent file {self.agent_file}: {e}")

        # find tools in agent file
        tools = agent_data.get("tools", [])
        if not tools:
            raise ValueError(f"Agent file {self.agent_file} contains no tools")

        # find the specified judge tool
        judge_tool = None
        for tool in tools:
            if tool.get("name") == self.judge_tool_name:
                judge_tool = tool
                break

        if not judge_tool:
            tool_names = [t.get("name") for t in tools]
            raise ValueError(
                f"Judge tool '{self.judge_tool_name}' not found in agent file {self.agent_file}. "
                f"Available tools: {tool_names}"
            )

        # validate tool has source_code with score and rationale parameters
        source_code = judge_tool.get("source_code", "")
        if not source_code:
            raise ValueError(
                f"Judge tool '{self.judge_tool_name}' in {self.agent_file} has no source_code. "
                f"Cannot validate parameter schema."
            )

        # check for score and rationale parameters in function signature
        if "score:" not in source_code and "score :" not in source_code:
            raise ValueError(
                f"Judge tool '{self.judge_tool_name}' in {self.agent_file} must have a 'score' parameter. "
                f"Expected signature: def {self.judge_tool_name}(score: float, rationale: str)"
            )

        if "rationale:" not in source_code and "rationale :" not in source_code:
            raise ValueError(
                f"Judge tool '{self.judge_tool_name}' in {self.agent_file} must have a 'rationale' parameter. "
                f"Expected signature: def {self.judge_tool_name}(score: float, rationale: str)"
            )

    def _parse_tool_calls(self, messages: List[LettaMessageUnion]) -> Tuple[float, str]:
        """Parse tool calls from messages to extract score and rationale.

        Args:
            messages: List of messages from the judge agent run

        Returns:
            Tuple of (score, rationale)

        Raises:
            ValueError: If submit_grade tool call not found or malformed
        """
        # Look through messages in reverse order to find the most recent tool call
        for msg in reversed(messages):
            if isinstance(msg, ToolCallMessage):
                # SDK v1.0 uses tool_calls (array), fall back to tool_call (singular) for compatibility
                tool_calls = msg.tool_calls if msg.tool_calls else ([msg.tool_call] if msg.tool_call else [])
                for tool_call in tool_calls:
                    # In SDK v1.0, tool_calls items are ToolCall objects
                    if isinstance(tool_call, dict):
                        tool_call = ToolCall(**tool_call)
                    if tool_call.name == self.judge_tool_name:
                        try:
                            args = json.loads(tool_call.arguments)
                            score = float(args.get("score", 0.0))
                            rationale = str(args.get("rationale", ""))
                            score = max(0.0, min(1.0, score))

                            return score, rationale
                        except (json.JSONDecodeError, ValueError, KeyError) as e:
                            raise ValueError(f"Failed to parse {self.judge_tool_name} tool call arguments: {e}")

        raise ValueError(f"No {self.judge_tool_name} tool call found in judge agent response")

    async def _grade_with_letta_code(self, judge_prompt: str) -> Tuple[float, str, str]:
        """Run judge agent using letta-code CLI for local file access.

        Returns:
            Tuple of (score, rationale, judge_agent_id)
        """
        # Import judge agent from .af file
        with open(self.agent_file, "rb") as f:
            resp = await self.client.agents.import_file(
                file=f, append_copy_suffix=False, override_existing_tools=False, project_id=self.project_id
            )
            if len(resp.agent_ids) > 1:
                raise RuntimeError(f"Expected single judge agent from .af file, got {len(resp.agent_ids)} agents")
            judge_agent_id = resp.agent_ids[0]

        # Construct the letta-code CLI command
        cmd = [
            "letta",
            "--agent",
            judge_agent_id,
            "--yolo",
            "--output-format",
            "json",
            "--model",
            self.model_handle,
            "-p",
            judge_prompt,
        ]

        # Prepare environment variables
        env = os.environ.copy()
        if self.base_url:
            env["LETTA_BASE_URL"] = self.base_url

        # Run the letta command
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.working_dir),
            env=env,
        )

        # Wait for completion (with timeout)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError("Letta-code judge command timed out after 300 seconds")

        stderr_text = stderr.decode() if stderr else ""

        if process.returncode != 0:
            raise RuntimeError(
                f"Letta-code judge command failed with return code {process.returncode}. Stderr: {stderr_text[:500]}"
            )

        # We already have judge_agent_id from the import, so we don't need to parse JSON output
        # The letta CLI may output JSON, but we can retrieve messages directly using the agent_id
        # Retrieve messages from the agent's last run
        messages_page = await self.client.agents.messages.list(agent_id=judge_agent_id)

        # Parse tool calls to extract score and rationale
        score, rationale = self._parse_tool_calls(messages_page.items)

        return score, rationale, judge_agent_id
