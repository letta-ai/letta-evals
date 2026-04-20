import logging
from pathlib import Path
from typing import Optional

import anyio
from letta_client import AsyncLetta
from letta_client.types import LlmConfig, MessageCreateParam

from letta_evals.models import Sample, TargetResult, TurnTokenData
from letta_evals.targets.base import AbstractAgentTarget, TargetError
from letta_evals.utils import consume_stream_with_resumes, list_all_run_messages, load_object
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
        timeout: Optional[int] = None,
    ):
        self.client = client
        self.agent_id = agent_id
        self.agent_file = agent_file
        self.agent_script = agent_script
        self.base_dir = base_dir or Path.cwd()
        self.llm_config = llm_config
        self.model_handle = model_handle
        self.max_retries = max_retries
        self.timeout = timeout

    async def run(
        self,
        sample: Sample,
        progress_callback: Optional[ProgressCallback] = None,
        project_id: Optional[str] = None,
        retrieve_agent_state: bool = False,
        return_token_data: bool = False,
    ) -> TargetResult:
        """Run the agent on a sample."""
        attempt = 0
        last_error = None

        while attempt <= self.max_retries:
            agent_id = self.agent_id
            agent_id_to_cleanup = None

            try:
                with anyio.fail_after(self.timeout):
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
                        # Workaround for letta-client SDK bug: serialize with aliases
                        # The SDK doesn't use by_alias=True, causing model_endpoint_type -> api_model_endpoint_type
                        llm_config_dict = self.llm_config.model_dump(by_alias=True, exclude_none=True)
                        await self.client.agents.update(agent_id=agent_id, llm_config=llm_config_dict)
                    elif self.model_handle and agent_id:
                        await self.client.agents.update(agent_id=agent_id, model=self.model_handle, parallel_tool_calls=True)

                    agent = await self.client.agents.retrieve(agent_id=agent_id, include=[])
                    if self.llm_config:
                        model_name = self.llm_config.model
                    elif self.model_handle:
                        model_name = self.model_handle
                    else:
                        model_name = agent.llm_config.model

                    if progress_callback and (self.agent_file or self.agent_script):
                        await progress_callback.agent_created(sample.id, agent_id=agent_id, model_name=model_name)

                    trajectory = []
                    usage_stats: list[dict] = []

                    inputs = sample.input if isinstance(sample.input, list) else [sample.input]
                    total_messages = len(inputs)

                    for i, input_msg in enumerate(inputs):
                        if progress_callback:
                            await progress_callback.message_sending(
                                sample.id, i + 1, total_messages, agent_id=agent_id, model_name=model_name
                            )

                        stream = await self.client.agents.messages.create(
                            agent_id=agent_id,
                            messages=[MessageCreateParam(role="user", content=str(input_msg))],
                            streaming=True,
                            background=True,
                            stream_tokens=True,
                            include_pings=True,
                            max_steps=100,
                        )

                        async def _on_chunk(chunk):
                            if not hasattr(chunk, "message_type"):
                                return

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
                                return

                            if chunk.message_type == "error_message":
                                detail = getattr(chunk, "detail", None) or getattr(chunk, "message", None) or str(chunk)
                                raise RuntimeError(f"Error for sample {sample.id}: {detail}")

                        async def _resume_stream(rid: str, seq_id: int):
                            return await self.client.runs.messages.stream(
                                rid,
                                starting_after=seq_id,
                                include_pings=True,
                            )

                        run_id, _ = await consume_stream_with_resumes(
                            stream,
                            resume_stream=_resume_stream,
                            on_chunk=_on_chunk,
                            max_resumes=5,
                            log=logger,
                            description=f"Stream for sample {sample.id}",
                        )

                        if not run_id:
                            raise RuntimeError("Unexpected error: no run ID was found from background stream")

                        messages = await list_all_run_messages(self.client, run_id)
                        trajectory.append(messages)

                    # Fetch token-level data if requested (for RL training)
                    token_data: Optional[list[TurnTokenData]] = None
                    if return_token_data and run_id:
                        token_data = await self._fetch_token_data([run_id])

                    final_agent_state = None
                    if retrieve_agent_state:
                        final_agent_state = await self.client.agents.retrieve(
                            agent_id=agent_id, include=["agent.blocks"]
                        )

                    return TargetResult(
                        trajectory=trajectory,
                        agent_id=agent_id,
                        model_name=model_name,
                        agent_usage=usage_stats,
                        agent_state=final_agent_state,
                        token_data=token_data,
                    )

            except Exception as e:
                last_error = e
                attempt += 1

                if attempt > self.max_retries:
                    timeout_hint = f"Timed out after {self.timeout}s" if isinstance(e, TimeoutError) else ""
                    msg = str(e) or timeout_hint or type(e).__name__
                    raise TargetError(msg, agent_id=agent_id) from e

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

    async def _fetch_token_data(self, run_ids: list[str]) -> list[TurnTokenData]:
        """Fetch token-level data (IDs + logprobs) from the runs API.

        For each run, re-fetches messages with ``return_token_ids=True`` to
        obtain ``output_ids`` and ``output_token_logprobs`` per message.
        """
        token_data: list[TurnTokenData] = []
        for run_id in run_ids:
            try:
                messages = await list_all_run_messages(
                    self.client,
                    run_id,
                    params={"return_token_ids": "true"},
                )
                for msg in messages:
                    output_ids = getattr(msg, "output_ids", None)
                    output_token_logprobs = getattr(msg, "output_token_logprobs", None)
                    if output_ids:
                        token_data.append(
                            TurnTokenData(
                                role=getattr(msg, "role", "assistant"),
                                output_ids=output_ids,
                                output_token_logprobs=output_token_logprobs,
                            )
                        )
            except Exception as e:
                logger.warning(f"Could not fetch token data for run {run_id}: {e}")
        return token_data
