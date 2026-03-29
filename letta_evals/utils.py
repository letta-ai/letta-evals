import importlib.util
import logging
import re
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import anyio
import httpx

from letta_evals.constants import (
    MODEL_COSTS,
    MODEL_NAME_MAPPING,
    TURN_FAIL_SYMBOL,
    TURN_PASS_SYMBOL,
    TURN_PENDING_SYMBOL,
)
from letta_evals.models import Sample, TurnTokenData

# Pattern to match reasoning effort suffixes
_EFFORT_PATTERN = re.compile(r"-(low|medium|high|xhigh|max)$")
_TOKEN_DATA_TOOL_ROLES = {"tool", "tool_return", "tool_return_message"}

logger = logging.getLogger(__name__)


def _timestamp_sort_key(value: Any) -> str:
    """Normalize timestamp-like values for deterministic sorting."""
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return str(value)


def sort_items_oldest_first(items: List[Any]) -> List[Any]:
    """Return items sorted by created_at/date then id (oldest to newest)."""

    def _sort_key(item: Any) -> tuple[str, str]:
        created_at = getattr(item, "created_at", None) or getattr(item, "date", None)
        return _timestamp_sort_key(created_at), str(getattr(item, "id", ""))

    return sorted(items, key=_sort_key)


def dedupe_items_by_id(items: List[Any]) -> List[Any]:
    """Dedupe object list by id while preserving first-seen order."""
    deduped_items: List[Any] = []
    seen_ids: set[str] = set()

    for item in items:
        item_id = getattr(item, "id", None)
        if item_id and item_id in seen_ids:
            continue
        if item_id:
            seen_ids.add(item_id)
        deduped_items.append(item)

    return deduped_items


def extract_run_ids_from_summaries(run_summaries: List[Any]) -> List[str]:
    """Extract deterministic, unique run IDs from run-summary objects."""
    ordered = sort_items_oldest_first(run_summaries)
    deduped = dedupe_items_by_id(ordered)
    return [run_id for run_id in (getattr(summary, "id", None) for summary in deduped) if run_id]


def _build_turn_token_data(
    *,
    role: str,
    output_ids: Optional[List[int]],
    output_token_logprobs: Optional[List[Any]],
    content: Any,
) -> Optional[TurnTokenData]:
    if output_ids:
        return TurnTokenData(
            role=role,
            output_ids=output_ids,
            output_token_logprobs=output_token_logprobs,
        )

    if role in _TOKEN_DATA_TOOL_ROLES and content is not None:
        text_content = content if isinstance(content, str) else str(content)
        if text_content:
            return TurnTokenData(role=role, content=text_content)

    return None


def extract_token_data_from_message(msg: Any) -> Optional[TurnTokenData]:
    """Extract TurnTokenData from a message object."""
    return _build_turn_token_data(
        role=getattr(msg, "role", "assistant"),
        output_ids=getattr(msg, "output_ids", None),
        output_token_logprobs=getattr(msg, "output_token_logprobs", None),
        content=getattr(msg, "content", None),
    )


def extract_token_data_from_messages(messages: List[Any]) -> List[TurnTokenData]:
    """Extract token data from a list of message objects."""
    return [td for msg in messages if (td := extract_token_data_from_message(msg)) is not None]


def extract_token_data_from_turn(turn: Dict[str, Any]) -> Optional[TurnTokenData]:
    """Extract TurnTokenData from a run.metadata.result.turn entry."""
    return _build_turn_token_data(
        role=turn.get("role", "assistant"),
        output_ids=turn.get("output_ids"),
        output_token_logprobs=turn.get("output_token_logprobs"),
        content=turn.get("content"),
    )


def extract_token_data_from_turns(turns: List[Dict[str, Any]]) -> List[TurnTokenData]:
    """Extract token data from run.metadata.result.turns entries."""
    return [td for turn in turns if (td := extract_token_data_from_turn(turn)) is not None]


async def list_run_ids(client: Any, agent_id: str) -> List[str]:
    """List run IDs for an agent in deterministic oldest->newest order."""
    try:
        runs_page = await client.runs.list(agent_id=agent_id, limit=200)
        run_summaries = list(getattr(runs_page, "items", None) or [])
    except Exception as e:
        logger.warning(f"Could not fetch run IDs for agent {agent_id}: {e}")
        return []
    return extract_run_ids_from_summaries(run_summaries)


async def fetch_token_data(client: Any, run_ids: List[str]) -> List[TurnTokenData]:
    """Fetch token data for runs in parallel, trying metadata then messages.

    For each run, first checks ``run.metadata.result.turns`` (populated by
    native-token adapters like SGLang).  If that's empty, falls back to
    re-fetching messages with ``return_token_ids=true``.
    """
    import asyncio

    async def _fetch_one(run_id: str) -> List[TurnTokenData]:
        # Try metadata path first (single API call, used by native adapters)
        run = await client.runs.retrieve(run_id=run_id)
        turns = ((run.metadata or {}).get("result") or {}).get("turns") or []
        if turns:
            return extract_token_data_from_turns(turns)

        # Fall back to messages path (used by hosted LLM endpoints)
        messages = await list_all_run_messages(
            client, run_id, params={"return_token_ids": "true"},
        )
        return extract_token_data_from_messages(messages)

    results = await asyncio.gather(*[_fetch_one(rid) for rid in run_ids], return_exceptions=True)
    token_data: List[TurnTokenData] = []
    for rid, result in zip(run_ids, results):
        if isinstance(result, Exception):
            logger.warning(f"Could not fetch token data for run {rid}: {result}")
        else:
            token_data.extend(result)
    return token_data


def load_object(spec: str, base_dir: Path = None) -> Any:
    """Load a Python object from a file path specification."""
    if not spec:
        raise ValueError("Empty specification provided")

    if ":" not in spec:
        raise ImportError(f"'{spec}' appears to be a simple name, not a file path")

    file_path, obj_name = spec.rsplit(":", 1)
    path = Path(file_path)

    # resolve relative paths
    if not path.is_absolute():
        if base_dir is None:
            raise ValueError(f"Relative path provided but no base_dir: {file_path}")
        path = (base_dir / path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix != ".py":
        raise ValueError(f"File must be a Python file (.py), got: {path}")

    module_name = f"_dynamic_{path.stem}_{id(path)}"
    spec_loader = importlib.util.spec_from_file_location(module_name, path)
    if spec_loader is None or spec_loader.loader is None:
        raise ImportError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec_loader)
    sys.modules[module_name] = module
    spec_loader.loader.exec_module(module)

    if not hasattr(module, obj_name):
        available = [name for name in dir(module) if not name.startswith("_")]
        raise AttributeError(f"Module '{path}' has no attribute '{obj_name}'. Available: {', '.join(available[:10])}")

    return getattr(module, obj_name)


def normalize_model_name(model_name: str) -> str:
    """
    Normalize model names to match MODEL_COSTS keys.

    Args:
        model_name: Raw model name (e.g., "gpt-4.1-mini", "openai/gpt-4.1", "claude-sonnet-4-5-20250929")

    Returns:
        Normalized model name that can be found in MODEL_COSTS
    """
    # Direct match in MODEL_COSTS
    if model_name in MODEL_COSTS:
        return model_name

    # Try the mapping (handles base names like "gpt-4.1-mini" -> "openai/gpt-4.1-mini-2025-04-14")
    if model_name in MODEL_NAME_MAPPING:
        return MODEL_NAME_MAPPING[model_name]

    # If it has a provider prefix (e.g., "openai/gpt-4.1"), strip it and try mapping
    if "/" in model_name:
        model_part = model_name.split("/", 1)[1]
        if model_part in MODEL_NAME_MAPPING:
            return MODEL_NAME_MAPPING[model_part]

    # Try with provider prefix for common patterns
    if model_name.startswith("claude"):
        prefixed = f"anthropic/{model_name}"
        if prefixed in MODEL_COSTS:
            return prefixed
    elif model_name.startswith("gpt"):
        prefixed = f"openai/{model_name}"
        if prefixed in MODEL_COSTS:
            return prefixed
    elif model_name.startswith("gemini"):
        prefixed = f"google_ai/{model_name}"
        if prefixed in MODEL_COSTS:
            return prefixed

    # Strip effort-level suffix (e.g. "gpt-5.2-high" -> "gpt-5.2") and retry
    stripped = _EFFORT_PATTERN.sub("", model_name)
    if stripped != model_name:
        return normalize_model_name(stripped)

    # No match found
    return model_name


def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate the cost for a model's token usage.

    Args:
        model_name: Name of the model (will be normalized if needed)
        prompt_tokens: Number of prompt tokens used
        completion_tokens: Number of completion tokens used

    Returns:
        Total cost in dollars, or 0.0 if model pricing is not available

    Note:
        Returns 0.0 if model pricing is not found in MODEL_COSTS instead of raising an error.
        This allows evaluation to continue even for new/unknown models.
    """
    # Normalize model name (resolve aliases and add provider prefix if needed)
    normalized_name = normalize_model_name(model_name)

    # Check if we have pricing for this model
    if normalized_name not in MODEL_COSTS:
        logger.debug(f"No pricing information available for model: {normalized_name} (original: {model_name})")
        return 0.0

    model_costs = MODEL_COSTS[normalized_name]
    prompt_cost = model_costs["prompt_tokens"] * prompt_tokens / 1_000_000
    completion_cost = model_costs["completion_tokens"] * completion_tokens / 1_000_000
    return prompt_cost + completion_cost


def extract_token_counts(agent_usage: Optional[List[dict]]) -> tuple[int, int, int, int, int]:
    """
    Extract total token counts from agent_usage data.

    Args:
        agent_usage: List of usage statistics from the agent run

    Returns:
        Tuple of (total_prompt_tokens, total_completion_tokens, cached_input_tokens, cache_write_tokens, reasoning_tokens)
    """
    if not agent_usage:
        return 0, 0, 0, 0, 0

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cached_input_tokens = 0
    total_cache_write_tokens = 0
    total_reasoning_tokens = 0

    for usage_record in agent_usage:
        if usage_record.get("message_type") == "usage_statistics":
            # Handle None values explicitly: .get() returns None if key exists with None value
            # Using 'or 0' ensures we treat None, missing keys, and falsy values as 0
            total_prompt_tokens += usage_record.get("prompt_tokens") or 0
            total_completion_tokens += usage_record.get("completion_tokens") or 0

            # Extract cached input tokens - check both top-level and nested prompt_tokens_details
            cached_input = usage_record.get("cached_input_tokens") or 0
            if cached_input == 0:
                # Check nested prompt_tokens_details structure
                prompt_details = usage_record.get("prompt_tokens_details") or {}
                if isinstance(prompt_details, dict):
                    # Try different field names used by different providers
                    cached_input = (
                        prompt_details.get("cached_tokens")  # OpenAI/Gemini
                        or prompt_details.get("cache_read_tokens")  # Anthropic
                        or prompt_details.get("cached_input_tokens")
                        or 0
                    )
            total_cached_input_tokens += cached_input

            # Extract cache write tokens - check both top-level and nested
            cache_write = usage_record.get("cache_write_tokens") or 0
            if cache_write == 0:
                prompt_details = usage_record.get("prompt_tokens_details") or {}
                if isinstance(prompt_details, dict):
                    cache_write = prompt_details.get("cache_creation_tokens") or 0
            total_cache_write_tokens += cache_write

            # Extract reasoning tokens - check both top-level and nested completion_tokens_details
            reasoning = usage_record.get("reasoning_tokens") or 0
            if reasoning == 0:
                completion_details = usage_record.get("completion_tokens_details") or {}
                if isinstance(completion_details, dict):
                    reasoning = completion_details.get("reasoning_tokens") or 0
            total_reasoning_tokens += reasoning

    return (
        total_prompt_tokens,
        total_completion_tokens,
        total_cached_input_tokens,
        total_cache_write_tokens,
        total_reasoning_tokens,
    )


def calculate_cost_from_agent_usage(model_name: str, agent_usage: Optional[List[dict]]) -> float:
    """
    Calculate total cost from agent_usage data.

    Args:
        model_name: Name of the model
        agent_usage: List of usage statistics from the agent run

    Returns:
        Total cost in dollars for the entire agent run
    """
    if not agent_usage:
        return 0.0

    total_cost = 0.0
    for usage_record in agent_usage:
        if usage_record.get("message_type") == "usage_statistics":
            # Handle None values explicitly: .get() returns None if key exists with None value
            prompt_tokens = usage_record.get("prompt_tokens") or 0
            completion_tokens = usage_record.get("completion_tokens") or 0
            total_cost += calculate_cost(model_name, prompt_tokens, completion_tokens)

    return total_cost


def _is_retryable_http_error(e: Exception) -> bool:
    """Return True for transient httpx/network errors.

    We intentionally keep this conservative (network/protocol + retriable HTTP statuses)
    so we don't mask real logic/config errors.
    """

    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        return status >= 500 or status in (408, 429)

    return isinstance(e, httpx.HTTPError)


async def _retry_async(
    fn: Callable[[], Awaitable[Any]],
    *,
    max_attempts: int,
    backoff_base_s: float,
    backoff_max_s: float,
    description: str,
) -> Any:
    """Retry an async function on transient HTTP errors."""

    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as e:
            attempt += 1
            should_retry = _is_retryable_http_error(e)

            if (not should_retry) or attempt >= max_attempts:
                logger.error(f"{description} failed after {attempt} attempt(s): {type(e).__name__}: {e}")
                raise

            sleep_s = min(backoff_max_s, backoff_base_s * (2 ** (attempt - 1)))
            logger.debug(
                f"{description} failed (attempt {attempt}/{max_attempts}); "
                f"retrying in {sleep_s:.2f}s: {type(e).__name__}: {e}"
            )
            await anyio.sleep(sleep_s)


async def list_all_run_messages(
    client: Any,
    run_id: str,
    *,
    page_limit: int = 200,
    max_attempts_per_page: int = 5,
    backoff_base_s: float = 0.5,
    backoff_max_s: float = 8.0,
    params: Optional[Dict[str, str]] = None,
) -> List[Any]:
    """List all messages for a run using pagination + retries.

    This avoids a single huge `list(limit=1000)` response, which is more prone to
    `RemoteProtocolError: incomplete chunked read` when message payloads are large.

    Returns a flat list of messages in ascending created_at order.
    """

    messages: List[Any] = []
    after: Optional[str] = None

    # Loop until the API returns an empty page.
    while True:

        async def _list_page() -> Any:
            kwargs: Dict[str, Any] = {"run_id": run_id, "limit": page_limit, "order": "asc"}
            if after is not None:
                kwargs["after"] = after
            if params:
                kwargs["extra_query"] = params
            return await client.runs.messages.list(**kwargs)

        page = await _retry_async(
            _list_page,
            max_attempts=max_attempts_per_page,
            backoff_base_s=backoff_base_s,
            backoff_max_s=backoff_max_s,
            description=f"runs.messages.list(run_id={run_id}, after={after})",
        )

        items = getattr(page, "items", None) or []
        if not items:
            break

        messages.extend(items)

        last_id = getattr(items[-1], "id", None)
        if not last_id or last_id == after:
            # No usable cursor (or cursor didn't advance) — avoid infinite loop.
            break
        after = last_id

    return sort_items_oldest_first(dedupe_items_by_id(messages))


async def list_all_agent_messages(
    client: Any,
    agent_id: str,
    *,
    page_limit: int = 200,
    max_attempts_per_page: int = 5,
    backoff_base_s: float = 0.5,
    backoff_max_s: float = 8.0,
) -> List[Any]:
    """List all messages for an agent using pagination + retries."""

    messages: List[Any] = []
    after: Optional[str] = None

    while True:

        async def _list_page() -> Any:
            kwargs = {"agent_id": agent_id, "limit": page_limit, "order": "asc"}
            if after is not None:
                kwargs["after"] = after
            return await client.agents.messages.list(**kwargs)

        page = await _retry_async(
            _list_page,
            max_attempts=max_attempts_per_page,
            backoff_base_s=backoff_base_s,
            backoff_max_s=backoff_max_s,
            description=f"agents.messages.list(agent_id={agent_id}, after={after})",
        )

        items = getattr(page, "items", None) or []
        if not items:
            break

        messages.extend(items)

        last_id = getattr(items[-1], "id", None)
        if not last_id or last_id == after:
            break
        after = last_id

    return messages


async def consume_stream_with_resumes(
    stream_iter: Any,
    *,
    resume_stream: Callable[[str, int], Awaitable[Any]],
    on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
    max_resumes: int = 5,
    backoff_base_s: float = 0.5,
    backoff_max_s: float = 8.0,
    log: Optional[logging.Logger] = None,
    description: str = "stream",
) -> tuple[Optional[str], Optional[int]]:
    """Consume an SSE stream, resuming from seq_id on transient disconnects.

    This helper is used by both targets and graders.

    Args:
        stream_iter: The initial async iterator returned by `agents.messages.create(..., streaming=True)`
        resume_stream: Function that returns a *new* stream iterator given (run_id, last_seq_id)
        on_chunk: Optional per-chunk handler. Should raise RuntimeError for fatal stream-level errors.
        max_resumes: Maximum number of resume attempts
        backoff_base_s: Exponential backoff base
        backoff_max_s: Maximum backoff sleep
        log: Logger for debug messages (defaults to this module's logger)
        description: Human-readable description for log messages

    Returns:
        (run_id, last_seq_id)
    """

    log = log or logger

    run_id: Optional[str] = None
    last_seq_id: Optional[int] = None

    async def _consume(single_iter: Any) -> None:
        nonlocal run_id, last_seq_id
        async for chunk in single_iter:
            rid = getattr(chunk, "run_id", None)
            if rid:
                run_id = rid
            if hasattr(chunk, "seq_id"):
                last_seq_id = getattr(chunk, "seq_id")

            if on_chunk is not None:
                await on_chunk(chunk)

    resume_attempt = 0
    current_iter = stream_iter
    cancel_exc = anyio.get_cancelled_exc_class()

    while True:
        try:
            await _consume(current_iter)
            break
        except cancel_exc:
            raise
        except RuntimeError:
            raise
        except Exception as stream_err:
            # Only resume on transient network/protocol errors.
            if not _is_retryable_http_error(stream_err):
                raise
            if not (run_id and last_seq_id is not None):
                raise

            resume_attempt += 1
            if resume_attempt > max_resumes:
                raise

            backoff_s = min(backoff_max_s, backoff_base_s * (2 ** (resume_attempt - 1)))
            log.debug(
                f"{description} disconnected; resuming (attempt {resume_attempt}/{max_resumes}) "
                f"from seq_id {last_seq_id}: {stream_err}"
            )
            await anyio.sleep(backoff_s)
            current_iter = await resume_stream(run_id, last_seq_id)

    return run_id, last_seq_id


def is_per_turn_evaluation(sample: Sample) -> bool:
    """Check if sample requires per-turn evaluation.

    Per-turn evaluation is used when both input and ground_truth are lists,
    allowing each turn in a multi-turn conversation to be evaluated against
    its own ground truth.

    Args:
        sample: The evaluation sample to check

    Returns:
        True if both input and ground_truth are lists (per-turn mode),
        False otherwise (standard evaluation mode)
    """
    return isinstance(sample.input, list) and isinstance(sample.ground_truth, list)


def build_turn_symbols(scores: List[Optional[float]], pass_threshold: float = 1.0) -> str:
    """Build a string of symbols representing turn scores.

    Args:
        scores: List of turn scores (None for ungraded turns)
        pass_threshold: Score threshold for pass (default 1.0)

    Returns:
        Space-separated string of symbols (e.g., "✓ ✓ ✗ …")
    """
    symbols = []
    for score in scores:
        if score is None:
            symbols.append(TURN_PENDING_SYMBOL)
        elif score >= pass_threshold:
            symbols.append(TURN_PASS_SYMBOL)
        else:
            symbols.append(TURN_FAIL_SYMBOL)
    return " ".join(symbols)


def calculate_turn_average(scores: List[Optional[float]]) -> float:
    """Calculate average of non-None turn scores.

    Args:
        scores: List of turn scores (None for ungraded turns)

    Returns:
        Average score, or 0.0 if no graded turns
    """
    graded = [sc for sc in scores if sc is not None]
    return sum(graded) / len(graded) if graded else 0.0


def build_turn_summary(scores: List[float], pass_threshold: float = 1.0) -> str:
    """Build a summary string for completed per-turn evaluation.

    Args:
        scores: List of turn scores (all graded)
        pass_threshold: Score threshold for pass (default 1.0)

    Returns:
        Summary string like "2/3 passed: ✓ ✓ ✗"
    """
    turns_passed = sum(1 for sc in scores if sc >= pass_threshold)
    total_turns = len(scores)
    symbols = build_turn_symbols(scores, pass_threshold)
    return f"{turns_passed}/{total_turns} passed: {symbols}"
