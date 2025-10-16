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

    async def run(
        self, sample: Sample, progress_callback: Optional[ProgressCallback] = None, project_id: Optional[str] = None
    ) -> TargetResult:
        """Run the agent on a sample."""
        agent_id = self.agent_id

        if self.agent_file:
            with open(self.agent_file, "rb") as f:
                resp = await self.client.agents.import_file(
                    file=f, append_copy_suffix=False, override_existing_tools=False, project_id=project_id
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

        if progress_callback and (self.agent_file or self.agent_script):
            await progress_callback.agent_loading(sample.id, model_name=model_name)

        trajectory = []
        usage_stats: list[dict] = []

        inputs = sample.input if isinstance(sample.input, list) else [sample.input]
        total_messages = len(inputs)

        for i, input_msg in enumerate(inputs):
            if progress_callback:
                await progress_callback.message_sending(sample.id, i + 1, total_messages, model_name=model_name)

            stream = self.client.agents.messages.create_stream(
                agent_id=agent_id,
                messages=[MessageCreate(role="user", content=str(input_msg))],
                stream_tokens=True,
            )

            run_id = None
            async for chunk in stream:
                # derive run_id from very first chunk, all should have the same
                if not run_id:
                    run_id = chunk.run_id

                # handle usage statistics in a streaming fashion
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

            if not run_id:
                raise RuntimeError("Unexpected error: no run ID was returned from streaming chunks.")

            # TODO: Set limit here potentially, this is capped to 100
            messages = await self.client.runs.messages.list(run_id=run_id)
            trajectory.append(messages)

        return TargetResult(trajectory=trajectory, agent_id=agent_id, model_name=model_name, agent_usage=usage_stats)
