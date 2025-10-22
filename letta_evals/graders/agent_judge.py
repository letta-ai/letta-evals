import json
from pathlib import Path
from typing import List, Optional, Tuple

from letta_client import AgentState, AsyncLetta, LettaMessageUnion, MessageCreate, ToolCall, ToolCallMessage

from letta_evals.extractors import extractor_requires_agent_state, get_extractor
from letta_evals.graders.base import Grader
from letta_evals.graders.prompt_utils import build_judge_prompt
from letta_evals.models import GradeResult, Sample


class AgentJudgeGrader(Grader):
    """Grader that uses a Letta agent as a judge with custom rubric prompts."""

    def __init__(
        self,
        agent_file: Path,
        prompt: str,
        client: AsyncLetta,
        project_id: Optional[str] = None,
        judge_tool_name: str = "submit_grade",
        extractor: str = "last_assistant",
        extractor_config: Optional[dict] = None,
        base_dir: Optional[Path] = None,
        rubric_vars: Optional[List[str]] = None,
    ):
        self.agent_file = agent_file
        self.prompt = prompt
        self.client = client
        self.project_id = project_id
        self.judge_tool_name = judge_tool_name
        self.extractor_name = extractor
        self.base_dir = base_dir
        self.rubric_vars = rubric_vars or []
        self.extractor = get_extractor(extractor, extractor_config, base_dir=base_dir)
        self._requires_agent_state = extractor_requires_agent_state(extractor, base_dir=base_dir)

        # validate agent file contains the required tool with correct schema
        self._validate_agent_file()

    @property
    def requires_agent_state(self) -> bool:
        """Whether this grader's extractor requires agent_state."""
        return self._requires_agent_state

    async def grade(
        self, sample: Sample, trajectory: List[List[LettaMessageUnion]], agent_state: Optional[AgentState] = None
    ) -> Tuple[GradeResult, str]:
        """Grade using agent judge with rubric."""
        submission = self.extractor(trajectory, agent_state=agent_state)

        judge_prompt = build_judge_prompt(
            self.prompt, sample, submission, self.rubric_vars, judge_tool_name=self.judge_tool_name
        )

        judge_agent_id = None
        try:
            # load judge agent from .af file
            with open(self.agent_file, "rb") as f:
                resp = await self.client.agents.import_file(
                    file=f, append_copy_suffix=False, override_existing_tools=False, project_id=self.project_id
                )
                if len(resp.agent_ids) > 1:
                    raise RuntimeError(f"Expected single judge agent from .af file, got {len(resp.agent_ids)} agents")

                judge_agent_id = resp.agent_ids[0]

            # send prompt to judge agent
            stream = self.client.agents.messages.create_stream(
                agent_id=judge_agent_id,
                messages=[MessageCreate(role="user", content=judge_prompt)],
                stream_tokens=False,
            )

            # consume stream
            run_id = None
            async for chunk in stream:
                if hasattr(chunk, "run_id"):
                    run_id = chunk.run_id

            if not run_id:
                raise RuntimeError("No run_id received from judge agent stream")

            messages = await self.client.runs.messages.list(run_id=run_id)
            score, rationale = self._parse_tool_calls(messages)

            return GradeResult(
                score=score,
                rationale=rationale,
                metadata={"judge_agent_id": judge_agent_id, "agent_file": str(self.agent_file)},
            ), submission

        except Exception as e:
            return GradeResult(
                score=0.0,
                rationale=f"Error during agent judge grading: {str(e)}",
                metadata={"error": str(e), "agent_file": str(self.agent_file)},
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
        for msg in messages:
            if isinstance(msg, ToolCallMessage):
                for tool_call in msg.tool_calls:
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
