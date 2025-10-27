import logging
from pathlib import Path
from typing import Optional

import anyio
from letta_client import AsyncLetta, LlmConfig, MessageCreate

from letta_evals.models import Sample, TargetResult
from letta_evals.targets.base import AbstractAgentTarget
from letta_evals.utils import load_object
from letta_evals.visualization.base import ProgressCallback

logger = logging.getLogger(__name__)


class LettaAgentTarget(AbstractAgentTarget):
    """Letta agent target for evaluation."""

    def __init__(
        self,
        client: AsyncLetta,
        agent_id: Optional[str] = None,
        agent_file: Optional[Path] = None,
        agent_script: Optional[str] = None,
        base_dir: Optional[Path] = None,
        llm_config: Optional[LlmConfig] = None,
        model_handle: Optional[str] = None,
        max_retries: int = 0,
    ):
        self.client = client
        self.agent_id = agent_id
        self.agent_file = agent_file
        self.agent_script = agent_script
        self.base_dir = base_dir or Path.cwd()
        self.llm_config = llm_config
        self.model_handle = model_handle
        self.max_retries = max_retries

    async def run(
        self,
        sample: Sample,
        progress_callback: Optional[ProgressCallback] = None,
        project_id: Optional[str] = None,
        retrieve_agent_state: bool = False,
    ) -> TargetResult:
        """Run the agent on a sample."""
        attempt = 0
        last_error = None

        while attempt <= self.max_retries:
            agent_id = self.agent_id
            agent_id_to_cleanup = None

            try:
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
                        agent_id_to_cleanup = agent_id

                elif self.agent_script:
                    agent_factory_func = load_object(self.agent_script, self.base_dir)
                    agent_id = await agent_factory_func(self.client, sample)
                    agent_id_to_cleanup = agent_id

                if self.llm_config and agent_id:
                    await self.client.agents.modify(agent_id=agent_id, llm_config=self.llm_config)
                elif self.model_handle and agent_id:
                    await self.client.agents.modify(agent_id=agent_id, model=self.model_handle)

                agent = await self.client.agents.retrieve(agent_id=agent_id, include_relationships=[])
                if self.llm_config:
                    model_name = self.llm_config.model
                elif self.model_handle:
                    model_name = self.model_handle
                else:
                    model_name = agent.llm_config.model

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
                    chunks = []
                    async for chunk in stream:
                        # derive run_id from very first chunk, all should have the same
                        # defensive for now, letta server needs fix to standardize run_id
                        chunks.append(chunk)

                        if not run_id and hasattr(chunk, "run_id"):
                            run_id = chunk.run_id

                        if hasattr(chunk, "message_type"):
                            if chunk.message_type == "usage_statistics":
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
                                    usage_rec = {"raw": str(chunk)}
                                usage_stats.append(usage_rec)
                                continue

                    if not run_id:
                        raise RuntimeError(f"Unexpected error: no run ID was found from streaming chunks: {chunks}")

                    # TODO: Set limit here potentially, this is capped to 100
                    messages = await self.client.runs.messages.list(run_id=run_id)
                    trajectory.append(messages)

                final_agent_state = None
                if retrieve_agent_state:
                    final_agent_state = await self.client.agents.retrieve(agent_id=agent_id, include_relationships=[])

                return TargetResult(
                    trajectory=trajectory,
                    agent_id=agent_id,
                    model_name=model_name,
                    agent_usage=usage_stats,
                    agent_state=final_agent_state,
                )

            except Exception as e:
                last_error = e
                attempt += 1

                if attempt > self.max_retries:
                    logger.error(
                        f"Failed to run agent for sample {sample.id} after {self.max_retries} retries. "
                        f"Final error: {type(e).__name__}: {str(e)}"
                    )
                    raise

                if agent_id_to_cleanup:
                    try:
                        await self.client.agents.delete(agent_id=agent_id_to_cleanup)
                        logger.info(f"Cleaned up agent {agent_id_to_cleanup} after failed attempt {attempt}")
                    except Exception as cleanup_error:
                        logger.warning(
                            f"Failed to cleanup agent {agent_id_to_cleanup}: {type(cleanup_error).__name__}: {str(cleanup_error)}"
                        )

                backoff_time = 2 ** (attempt - 1)
                logger.warning(
                    f"Agent run failed for sample {sample.id} (attempt {attempt}/{self.max_retries + 1}). "
                    f"Error: {type(e).__name__}: {str(e)}. Retrying in {backoff_time}s..."
                )
                await anyio.sleep(backoff_time)

        raise last_error or RuntimeError("Unexpected failure in agent run retry loop")
