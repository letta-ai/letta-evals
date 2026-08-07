"""Execution trace extraction helpers.

These helpers turn raw execution output (stream events) and server-side
agent state keyed by ``agent_id`` into the trace fields persisted on
``SampleResult``. They intentionally live outside targets so in-process and
sandboxed runs share the same fetch path.
"""

import asyncio
import json
import logging
import random
from typing import Any, Optional

from letta_client import APIConnectionError, APITimeoutError

from letta_evals.models import TurnTokenData
from letta_evals.utils import list_all_agent_messages

logger = logging.getLogger(__name__)

_TOKEN_FETCH_MAX_ATTEMPTS = 3
_TOKEN_FETCH_RETRY_BASE_SECONDS = 0.5
_RETRYABLE_TOKEN_FETCH_ERRORS = (APIConnectionError, APITimeoutError)


class TokenDataFetchError(RuntimeError):
    """Raised when a complete, internally consistent token trace cannot be fetched."""

    def __init__(
        self,
        agent_id: str,
        *,
        completed_runs: int,
        total_runs: int | None,
        failed_run_id: str | None,
        reason: str,
    ) -> None:
        total = "unknown" if total_runs is None else str(total_runs)
        failed_run = f", failed_run_id={failed_run_id}" if failed_run_id else ""
        super().__init__(
            f"Atomic token-data fetch failed for agent {agent_id}: {reason} "
            f"(completed_runs={completed_runs}/{total}{failed_run})"
        )
        self.agent_id = agent_id
        self.completed_runs = completed_runs
        self.total_runs = total_runs
        self.failed_run_id = failed_run_id


def extract_usage_stats(last_line: str) -> Optional[list[dict]]:
    """Pull agent usage_statistics from the stream's final line.

    stream-json emits the ``result`` event last. Returns ``None`` when the
    line is empty, unparseable, or not a result event (e.g. a stream cut
    short by a crash or timeout).
    """
    try:
        event = json.loads(last_line) if last_line else None
    except json.JSONDecodeError as err:
        logger.warning(f"Unparseable final stream line ({err.msg} at pos {err.pos}): {last_line[:200]}")
        return None
    if not isinstance(event, dict) or event.get("type") != "result" or "usage" not in event:
        return None
    usage = event["usage"]
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


async def _list_agent_runs(client: Any, agent_id: str) -> list[Any]:
    """List every run for an agent, including pages beyond the API limit."""
    try:
        runs_page = await client.runs.list(agent_id=agent_id, limit=100, order="asc")
    except TypeError:
        # Older generated clients may not expose the ``order`` kwarg.
        runs_page = await client.runs.list(agent_id=agent_id, limit=100)

    runs: list[Any] = []
    seen_run_ids: set[str] = set()
    while True:
        new_runs = [run for run in runs_page.items if str(run.id) not in seen_run_ids]
        if runs_page.items and not new_runs:
            # Defensive stop if an older server ignores the pagination cursor.
            break
        runs.extend(new_runs)
        seen_run_ids.update(str(run.id) for run in new_runs)

        has_next_page = getattr(runs_page, "has_next_page", None)
        get_next_page = getattr(runs_page, "get_next_page", None)
        if not callable(has_next_page) or not callable(get_next_page) or not runs_page.has_next_page():
            break
        runs_page = await runs_page.get_next_page()

    return runs


async def _fetch_token_data_once(client: Any, agent_id: str) -> list[TurnTokenData]:
    """Fetch one complete token-data snapshot or raise without returning a prefix."""
    try:
        # Client tools cause each tool-call round-trip to be a separate run, so
        # token IDs are scattered across the agent's runs.
        run_summaries = sorted(await _list_agent_runs(client, agent_id), key=_run_sort_key)
    except _RETRYABLE_TOKEN_FETCH_ERRORS as exc:
        raise TokenDataFetchError(
            agent_id,
            completed_runs=0,
            total_runs=None,
            failed_run_id=None,
            reason=f"{type(exc).__name__}: {exc}",
        ) from exc

    token_data: list[TurnTokenData] = []

    for completed_runs, run_summary in enumerate(run_summaries):
        try:
            run = await client.runs.retrieve(run_id=run_summary.id)
        except _RETRYABLE_TOKEN_FETCH_ERRORS as exc:
            raise TokenDataFetchError(
                agent_id,
                completed_runs=completed_runs,
                total_runs=len(run_summaries),
                failed_run_id=str(run_summary.id),
                reason=f"{type(exc).__name__}: {exc}",
            ) from exc

        result = (run.metadata or {}).get("result", {})
        for turn in result.get("turns") or []:
            output_ids = turn.get("output_ids")
            role = turn.get("role", "assistant")
            if output_ids:
                logprobs = turn.get("output_token_logprobs")
                if logprobs is not None and len(output_ids) != len(logprobs):
                    raise TokenDataFetchError(
                        agent_id,
                        completed_runs=completed_runs,
                        total_runs=len(run_summaries),
                        failed_run_id=str(run_summary.id),
                        reason=(f"half-written turn has {len(output_ids)} output IDs but {len(logprobs)} logprobs"),
                    )
                token_data.append(
                    TurnTokenData(
                        role=role,
                        input_ids=turn.get("input_ids"),
                        output_ids=output_ids,
                        output_token_logprobs=logprobs,
                    )
                )
            elif role in ("tool", "tool_return", "tool_return_message") and turn.get("content"):
                token_data.append(
                    TurnTokenData(
                        role=role,
                        content=turn.get("content"),
                    )
                )

    return token_data


async def fetch_token_data(
    client: Any,
    agent_id: str,
    *,
    max_attempts: int = _TOKEN_FETCH_MAX_ATTEMPTS,
    retry_base_seconds: float = _TOKEN_FETCH_RETRY_BASE_SECONDS,
) -> list[TurnTokenData]:
    """Fetch token IDs and logprobs atomically, retrying the complete snapshot.

    Each attempt starts with a fresh accumulator. A transport failure or
    half-written turn discards that attempt's prefix. After ``max_attempts``
    failures, ``TokenDataFetchError`` propagates so callers can retry the full
    rollout rather than training or evaluating on incomplete token data.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    last_error: TokenDataFetchError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await _fetch_token_data_once(client, agent_id)
        except TokenDataFetchError as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            delay = retry_base_seconds * (2 ** (attempt - 1))
            if delay > 0:
                delay += random.uniform(0.0, retry_base_seconds)
            logger.warning(
                "Token-data fetch attempt %d/%d failed for agent %s; discarding partial data and retrying in %.2fs: %r",
                attempt,
                max_attempts,
                agent_id,
                delay,
                exc.__cause__ or exc,
            )
            await asyncio.sleep(delay)

    assert last_error is not None
    raise last_error


async def fetch_agent_state(client: Any, agent_id: str) -> Any:
    """Fetch the final agent state, including memory blocks, for graders that need it."""
    return await client.agents.retrieve(agent_id=agent_id, include=["agent.blocks"])
