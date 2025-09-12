from pathlib import Path
from typing import Optional

from letta_client import AsyncLetta, MessageCreate

from letta_evals.models import Sample, TargetResult
from letta_evals.targets.base import Target
from letta_evals.types import ProgressCallback
from letta_evals.utils.module_loader import load_object


class AgentTarget(Target):
    """Letta agent target for evaluation."""

    def __init__(
        self,
        base_url: str,
        agent_id: str = None,
        agent_file: Path = None,
        agent_script: str = None,
        api_key: str = None,
        timeout: float = 300.0,
    ):
        self.base_url = base_url
        self.agent_id = agent_id
        self.agent_file = agent_file
        self.agent_script = agent_script

        self.client = AsyncLetta(base_url=self.base_url, token=api_key, timeout=timeout)

    async def run(
        self, sample: Sample, progress_callback: Optional[ProgressCallback] = None, sample_id: Optional[int] = None
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

        elif self.agent_script:
            if progress_callback and sample_id is not None:
                await progress_callback.agent_loading(sample_id)

            # load the agent factory class from the script
            base_dir = Path.cwd()  # use current working directory as base
            factory_class = load_object(self.agent_script, base_dir)

            # instantiate the factory and create the agent
            factory = factory_class()
            agent_id = await factory.create(self.client)

        trajectory = []

        inputs = sample.input if isinstance(sample.input, list) else [sample.input]
        total_messages = len(inputs)

        for i, input_msg in enumerate(inputs):
            if progress_callback and sample_id is not None:
                await progress_callback.message_sending(sample_id, i + 1, total_messages)

            stream = self.client.agents.messages.create_stream(
                agent_id=agent_id,
                messages=[MessageCreate(role="user", content=input_msg)],
            )

            messages = []
            prev_message_type = None
            async for chunk in stream:
                # skip non-message types like stop_reason and usage_statistics
                if hasattr(chunk, "message_type"):
                    if chunk.message_type in ["stop_reason", "usage_statistics"]:
                        continue
                    current_message_type = chunk.message_type
                    if prev_message_type != current_message_type:
                        messages.append(chunk)
                    prev_message_type = current_message_type

            trajectory.append(messages)

        return TargetResult(trajectory=trajectory, agent_id=agent_id)
