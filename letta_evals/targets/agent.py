from pathlib import Path
from typing import Optional

from letta_client import AsyncLetta, LlmConfig, MessageCreate

from letta_evals.models import Sample, TargetResult
from letta_evals.targets.base import Target
from letta_evals.types import ProgressCallback
from letta_evals.utils import load_object


class AgentTarget(Target):
    """Letta agent target for evaluation."""

    def __init__(
        self,
        client: AsyncLetta,
        agent_id: str = None,
        agent_file: Path = None,
        agent_script: str = None,
        base_dir: Path = None,
        llm_config: Optional[LlmConfig] = None,
    ):
        self.client = client
        self.agent_id = agent_id
        self.agent_file = agent_file
        self.agent_script = agent_script
        self.base_dir = base_dir or Path.cwd()
        self.llm_config = llm_config

    async def run(self, sample: Sample, progress_callback: Optional[ProgressCallback] = None) -> TargetResult:
        """Run the agent on a sample."""
        agent_id = self.agent_id

        if self.agent_file:
            with open(self.agent_file, "rb") as f:
                resp = await self.client.agents.import_file(
                    file=f, append_copy_suffix=False, override_existing_tools=False
                )
                if len(resp.agent_ids) > 1:
                    raise RuntimeError(
                        f"Expected single agent from .af file, got {len(resp.agent_ids)} agents. We don't support multi-agent evals yet."
                    )

                agent_id = resp.agent_ids[0]

        elif self.agent_script:
            agent_factory_func = load_object(self.agent_script, self.base_dir)
            agent_id = await agent_factory_func(self.client, sample)

        if self.llm_config and agent_id:
            await self.client.agents.modify(agent_id=agent_id, llm_config=self.llm_config)

        agent = await self.client.agents.retrieve(agent_id=agent_id, include_relationships=[])
        model_name = self.llm_config.model if self.llm_config else agent.llm_config.model

        # notify progress callback with model name
        if progress_callback and (self.agent_file or self.agent_script):
            await progress_callback.agent_loading(sample.id, model_name=model_name)

        trajectory = []
        usage_stats: list[dict] = []

        # Check if there's a contradicting fact to send before the questions
        contradicting_fact = None
        if sample.agent_args and "extra" in sample.agent_args and sample.agent_args["extra"]:
            contradicting_fact = sample.agent_args["extra"].get("contradicting_fact", None)

        inputs = sample.input if isinstance(sample.input, list) else [sample.input]
        total_messages = len(inputs)

        # Send contradicting fact first if available
        if contradicting_fact:
            try:
                stream = self.client.agents.messages.create_stream(
                    agent_id=agent_id,
                    messages=[
                        MessageCreate(
                            role="user",
                            content=f"Please update your knowledge with this new information: {contradicting_fact}",
                        )
                    ],
                    stream_tokens=True,
                )

                # Process the stream to ensure the message is sent
                async for chunk in stream:
                    # Process each chunk as needed - we just need to consume the stream
                    pass
            except Exception:
                # Continue even if there's an exception
                pass

        for i, input_msg in enumerate(inputs):
            if progress_callback:
                await progress_callback.message_sending(sample.id, i + 1, total_messages, model_name=model_name)

            stream = self.client.agents.messages.create_stream(
                agent_id=agent_id,
                messages=[MessageCreate(role="user", content=input_msg)],
            )

            messages = []

            prev_message_type = None
            async for chunk in stream:
                # handle usage statistics and skip other non-message types
                if hasattr(chunk, "message_type"):
                    if chunk.message_type == "usage_statistics":
                        # best-effort convert to JSON-serializable dict
                        usage_rec = None
                        if hasattr(chunk, "model_dump") and callable(getattr(chunk, "model_dump")):
                            try:
                                usage_rec = chunk.model_dump()
                            except Exception:
                                usage_rec = None
                        if usage_rec is None and hasattr(chunk, "dict") and callable(getattr(chunk, "dict")):
                            try:
                                usage_rec = chunk.dict()  # type: ignore[attr-defined]
                            except Exception:
                                usage_rec = None
                        if usage_rec is None and hasattr(chunk, "__dict__"):
                            try:
                                usage_rec = dict(chunk.__dict__)
                            except Exception:
                                usage_rec = None
                        if usage_rec is None:
                            # final fallback to string
                            usage_rec = {"raw": str(chunk)}
                        usage_stats.append(usage_rec)
                        continue
                    if chunk.message_type in ["stop_reason", "ping"]:
                        continue
                    current_message_type = chunk.message_type
                    if prev_message_type != current_message_type:
                        messages.append(chunk)
                    prev_message_type = current_message_type

            trajectory.append(messages)

        return TargetResult(trajectory=trajectory, agent_id=agent_id, model_name=model_name, agent_usage=usage_stats)
