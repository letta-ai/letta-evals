from pathlib import Path
from typing import Optional

from letta_client import AsyncLetta, MessageCreate

from letta_evals.models import Sample, TargetResult
from letta_evals.targets.base import Target


class AgentTarget(Target):
    """Letta agent target for evaluation."""

    def __init__(
        self, base_url: str, agent_id: str = None, agent_file: Path = None, api_key: str = None, timeout: float = 300.0
    ):
        self.base_url = base_url
        self.agent_id = agent_id
        self.agent_file = agent_file

        self.client = AsyncLetta(base_url=self.base_url, token=api_key, timeout=timeout)

    async def run(
        self, sample: Sample, progress_callback: Optional[object] = None, sample_id: Optional[int] = None
    ) -> TargetResult:
        """Run the agent on a sample."""
        agent_id = self.agent_id

        if self.agent_file:
            if progress_callback and sample_id is not None:
                await progress_callback.agent_loading(sample_id)

            with open(self.agent_file, "rb") as f:
                resp = await self.client.agents.import_file(
                    file=f, append_copy_suffix=False, override_existing_tools=False
                )
                if len(resp.agent_ids) > 1:
                    raise RuntimeError(
                        f"Expected single agent from .af file, got {len(resp.agent_ids)} agents. We don't support multi-agent evals yet."
                    )

                agent_id = resp.agent_ids[0]

        trajectory = []

        inputs = sample.input if isinstance(sample.input, list) else [sample.input]
        total_messages = len(inputs)

        for i, input_msg in enumerate(inputs):
            if progress_callback and sample_id is not None:
                await progress_callback.message_sending(sample_id, i + 1, total_messages)

            letta_resp = await self.client.agents.messages.create(
                agent_id=agent_id,
                messages=[MessageCreate(role="user", content=input_msg)],
            )
            trajectory.append(letta_resp.messages)

        return TargetResult(trajectory=trajectory, agent_id=agent_id)
