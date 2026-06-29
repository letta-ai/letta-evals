"""Execution trace extraction helpers.

These helpers turn raw execution output (stream events) and server-side
agent state keyed by ``agent_id`` into the trace fields persisted on
``SampleResult``. They intentionally live outside targets so in-process and
sandboxed runs share the same fetch path.
"""

import logging
from typing import Any, Optional

from letta_evals.models import TurnTokenData
from letta_evals.utils import list_all_agent_messages

logger = logging.getLogger(__name__)


def extract_usage_stats(events: list) -> Optional[list[dict]]:
    """Pull agent usage_statistics from the final stream ``result`` event.

    stream-json always emits the result event last. Returns ``None`` when no
    usage is present — e.g. the stream was cut short by a crash or timeout —
    so both the success path and the error path report usage the same way.
    """
    if events and events[-1].get("type") == "result" and "usage" in events[-1]:
        usage = events[-1]["usage"]
        return [
            {
                "message_type": "usage_statistics",
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "cached_input_tokens": usage.get("cached_input_tokens", 0),
                "cache_write_tokens": usage.get("cache_write_tokens", 0),
                "reasoning_tokens": usage.get("reasoning_tokens", 0),
            }
        ]
    return None


async def fetch_trajectory(client: Any, agent_id: str) -> list:
    """Fetch the agent's full message history as a single-turn trajectory.

    Single source of truth for the list + single-turn wrapping, called from
    both the success path and the error path (best-effort) so neither
    reimplements it.
    """
    logger.info(f"Retrieving messages for agent {agent_id}")
    messages = await list_all_agent_messages(client, agent_id)
    return [messages] if messages else []


def _run_sort_key(run_summary: Any) -> tuple[str, str]:
    """Return a stable chronological sort key for run summaries.

    Letta server defaults for ``runs.list`` have varied across local
    development branches. Token data must be processed oldest-to-newest so
    Tinker sequence-extension can merge consecutive assistant generations.
    Normalize timestamps to strings so mixed SDK/server timestamp types do
    not make sorting fail.
    """
    created_at = getattr(run_summary, "created_at", None)
    if hasattr(created_at, "isoformat"):
        created_key = created_at.isoformat()
    elif created_at is None:
        created_key = ""
    else:
        created_key = str(created_at)
    return created_key, str(getattr(run_summary, "id", ""))


async def fetch_token_data(client: Any, agent_id: str) -> list[TurnTokenData]:
    """Fetch token-level data (IDs + logprobs) for a letta code agent.

    Reads token IDs and logprobs from ``run.metadata.result.turns``, which is
    populated by Letta's SGLang-native adapter for token-aware model runs.

    Stops at the first half-written turn — ``output_ids`` present but a
    shorter ``output_token_logprobs`` — returning only the clean prefix, so a
    partially-flushed generation can't corrupt Tinker's sequence-extension.
    """
    token_data: list[TurnTokenData] = []
    try:
        # Fetch ALL runs for this agent — client tools cause each tool-call
        # round-trip to be a separate run, so token IDs are scattered.
        try:
            runs_page = await client.runs.list(agent_id=agent_id, limit=100, order="asc")
        except TypeError:
            # Older generated clients may not expose the ``order`` kwarg.
            # Fall back to the legacy call and sort locally below.
            runs_page = await client.runs.list(agent_id=agent_id, limit=100)
        if not runs_page.items:
            return token_data

        # Token IDs are stored in run.metadata.result.turns (populated by SGLang native adapter)
        for run_summary in sorted(runs_page.items, key=_run_sort_key):
            run = await client.runs.retrieve(run_id=run_summary.id)
            result = (run.metadata or {}).get("result", {})
            for turn in result.get("turns") or []:
                output_ids = turn.get("output_ids")
                role = turn.get("role", "assistant")
                if output_ids:
                    logprobs = turn.get("output_token_logprobs")
                    if logprobs is not None and len(output_ids) != len(logprobs):
                        # Half-written generation: ids present but logprobs
                        # not fully flushed. Drop it and everything after.
                        logger.info(
                            f"Truncating token data at half-written turn in run {run_summary.id} for agent {agent_id}"
                        )
                        return token_data
                    # Assistant turn with token IDs from SGLang
                    token_data.append(
                        TurnTokenData(
                            role=role,
                            input_ids=turn.get("input_ids"),
                            output_ids=output_ids,
                            output_token_logprobs=logprobs,
                        )
                    )
                elif role in ("tool", "tool_return", "tool_return_message") and turn.get("content"):
                    # Tool return turn — no output_ids, but content is needed
                    # for proper multi-turn token sequence reconstruction
                    token_data.append(
                        TurnTokenData(
                            role=role,
                            content=turn.get("content"),
                        )
                    )
    except Exception as e:
        logger.warning(f"Could not fetch token data for agent {agent_id}: {e}")
    return token_data


async def fetch_agent_state(client: Any, agent_id: str) -> Any:
    """Fetch the final agent state, including memory blocks, for graders that need it."""
    return await client.agents.retrieve(agent_id=agent_id, include=["agent.blocks"])
