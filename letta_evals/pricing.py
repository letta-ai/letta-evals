"""Model pricing loaded from BerriAI/litellm pricing JSON.

The pricing table is fetched from upstream once per process. Disk cache
(`~/.cache/letta_evals/model_prices.json`) lives for 30 minutes; when the cache
is fresh we skip the network. On cache miss we refetch and rewrite. If a fetch
fails *and* a stale cache exists, we use the stale cache and log a warning;
otherwise the underlying network error propagates.

Use ``MODEL_PRICE_OVERRIDES`` for models the upstream JSON hasn't published yet
(internal/preview names). Overrides win over the JSON.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

LITELLM_PRICING_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
CACHE_DIR = Path(os.environ.get("LETTA_EVALS_CACHE_DIR", str(Path.home() / ".cache" / "letta_evals")))
CACHE_FILE = CACHE_DIR / "model_prices.json"
CACHE_TTL_SECONDS = 30 * 60  # 30 minutes
NETWORK_TIMEOUT_SECONDS = 5

# Pattern to match reasoning effort suffixes (-low, -medium, -high, -xhigh, -max)
_EFFORT_PATTERN = re.compile(r"-(low|medium|high|xhigh|max)$")
# Pattern to match date suffixes: -YYYY-MM-DD or -YYYYMMDD
_DATE_PATTERN = re.compile(r"-\d{4}-\d{2}-\d{2}$|-\d{8}$")

# In-process cache: parsed pricing table. Loaded lazily.
_PRICING: Optional[Dict[str, "ModelPricing"]] = None


@dataclass(frozen=True)
class ModelPricing:
    """Per-token costs for a model. ``None`` for cache fields means 'not provided'."""

    input_per_token: float
    output_per_token: float
    cache_read_per_token: Optional[float] = None
    cache_creation_per_token: Optional[float] = None
    # Tiered pricing (e.g. Anthropic Sonnet/Opus 4.5 and Gemini 3 Pro charge ~2x above 200k context)
    threshold_tokens: Optional[int] = None
    input_above_threshold_per_token: Optional[float] = None
    output_above_threshold_per_token: Optional[float] = None
    cache_read_above_threshold_per_token: Optional[float] = None
    cache_creation_above_threshold_per_token: Optional[float] = None


# Manual overrides for models that haven't shipped to litellm yet, or where we
# want to pin a different rate. Keys are checked verbatim before any normalization,
# so use the same canonical form your runner emits (e.g. "openai/gpt-5.2-experimental").
# Empty by default - litellm covers everything we use today.
MODEL_PRICE_OVERRIDES: Dict[str, ModelPricing] = {}


# Provider-prefix translation: maps our prefix -> ordered list of litellm prefixes to try.
# Each candidate is tried both with and without date-stripping. An empty prefix means
# "bare key" (no provider segment).
_PROVIDER_CANDIDATES: Dict[str, List[str]] = {
    "anthropic": [""],
    "openai": [""],
    "google_ai": ["", "gemini/", "vertex_ai/"],
    "deepseek": ["deepseek/", ""],
    "mistralai": ["mistral/"],
    "moonshotai": ["moonshot/"],
    "z-ai": ["zai/"],
    "minimax": ["minimax/"],
}

# When no provider prefix is given, try these in order based on bare-name patterns.
_BARE_NAME_PROVIDERS: Dict[str, List[str]] = {
    "claude": ["", "anthropic/"],
    "gpt": [""],
    "gemini": ["", "gemini/", "vertex_ai/"],
    "deepseek": ["deepseek/", ""],
    "mistral": ["mistral/", ""],
    "kimi": ["moonshot/"],
    "glm": ["zai/", ""],
    "minimax": ["minimax/", ""],
}


def _build_pricing_from_json(raw: dict) -> Dict[str, ModelPricing]:
    """Convert the raw litellm JSON into a dict of model -> ModelPricing."""
    out: Dict[str, ModelPricing] = {}
    for key, entry in raw.items():
        if key == "sample_spec" or not isinstance(entry, dict):
            continue
        input_cost = entry.get("input_cost_per_token")
        output_cost = entry.get("output_cost_per_token")
        if input_cost is None or output_cost is None:
            # Embeddings, audio, etc. - skip anything without both required fields.
            continue

        threshold = None
        input_above = entry.get("input_cost_per_token_above_200k_tokens")
        output_above = entry.get("output_cost_per_token_above_200k_tokens")
        cache_read_above = entry.get("cache_read_input_token_cost_above_200k_tokens")
        cache_create_above = entry.get("cache_creation_input_token_cost_above_200k_tokens")
        if any(v is not None for v in (input_above, output_above, cache_read_above, cache_create_above)):
            threshold = 200_000

        out[key] = ModelPricing(
            input_per_token=float(input_cost),
            output_per_token=float(output_cost),
            cache_read_per_token=_as_float_or_none(entry.get("cache_read_input_token_cost")),
            cache_creation_per_token=_as_float_or_none(entry.get("cache_creation_input_token_cost")),
            threshold_tokens=threshold,
            input_above_threshold_per_token=_as_float_or_none(input_above),
            output_above_threshold_per_token=_as_float_or_none(output_above),
            cache_read_above_threshold_per_token=_as_float_or_none(cache_read_above),
            cache_creation_above_threshold_per_token=_as_float_or_none(cache_create_above),
        )
    return out


def _as_float_or_none(v) -> Optional[float]:
    return float(v) if v is not None else None


def _read_disk_cache() -> Optional[dict]:
    if not CACHE_FILE.exists():
        return None
    try:
        with CACHE_FILE.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to read pricing cache at {CACHE_FILE}: {e}")
        return None


def _write_disk_cache(raw: dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to temp then rename
        tmp = CACHE_FILE.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(raw, f)
        tmp.replace(CACHE_FILE)
    except OSError as e:
        logger.warning(f"Failed to write pricing cache to {CACHE_FILE}: {e}")


def _fetch_upstream() -> dict:
    """Fetch the litellm pricing JSON. Raises on network errors."""
    resp = httpx.get(LITELLM_PRICING_URL, timeout=NETWORK_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()


def _cache_is_fresh() -> bool:
    if not CACHE_FILE.exists():
        return False
    try:
        age = time.time() - CACHE_FILE.stat().st_mtime
        return age < CACHE_TTL_SECONDS
    except OSError:
        return False


def load_pricing_table() -> Dict[str, ModelPricing]:
    """Load the pricing table, populating the in-process cache on first call."""
    global _PRICING
    if _PRICING is not None:
        return _PRICING

    raw: Optional[dict] = None

    if _cache_is_fresh():
        raw = _read_disk_cache()
        if raw is not None:
            logger.debug(f"Using fresh pricing cache from {CACHE_FILE}")

    if raw is None:
        try:
            raw = _fetch_upstream()
            _write_disk_cache(raw)
            logger.debug(f"Fetched fresh pricing from {LITELLM_PRICING_URL}")
        except Exception as e:
            stale = _read_disk_cache()
            if stale is not None:
                logger.warning(
                    f"Failed to fetch pricing from upstream ({e}); falling back to stale cache at {CACHE_FILE}"
                )
                raw = stale
            else:
                raise

    _PRICING = _build_pricing_from_json(raw)
    return _PRICING


def _candidate_keys(model_handle: str) -> List[str]:
    """Generate the ordered list of litellm keys to probe for a Letta model handle."""
    candidates: List[str] = [model_handle]

    if "/" in model_handle:
        provider, model_part = model_handle.split("/", 1)
        prefixes = _PROVIDER_CANDIDATES.get(provider, [""])
        for prefix in prefixes:
            candidates.append(f"{prefix}{model_part}")
            stripped = _DATE_PATTERN.sub("", model_part)
            if stripped != model_part:
                candidates.append(f"{prefix}{stripped}")
        # Also try the bare model_part
        candidates.append(model_part)
    else:
        # No provider prefix - try each known pattern based on the leading token
        for marker, prefixes in _BARE_NAME_PROVIDERS.items():
            if model_handle.startswith(marker):
                for prefix in prefixes:
                    candidates.append(f"{prefix}{model_handle}")
                    stripped = _DATE_PATTERN.sub("", model_handle)
                    if stripped != model_handle:
                        candidates.append(f"{prefix}{stripped}")
                break

    # De-duplicate while preserving order
    seen: set = set()
    deduped: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def resolve_model(model_handle: str) -> Optional[ModelPricing]:
    """Resolve a Letta-style model handle to a ModelPricing entry, or None if unknown.

    Resolution order:
        1. ``MODEL_PRICE_OVERRIDES`` exact match.
        2. Strip effort suffix (-low, -medium, -high, -xhigh, -max) and recurse.
        3. Try provider-prefix candidates against the litellm JSON.
    """
    if not model_handle:
        return None

    if model_handle in MODEL_PRICE_OVERRIDES:
        return MODEL_PRICE_OVERRIDES[model_handle]

    # Strip effort suffix and try again (recursively, once)
    stripped = _EFFORT_PATTERN.sub("", model_handle)
    if stripped != model_handle:
        return resolve_model(stripped)

    table = load_pricing_table()
    for candidate in _candidate_keys(model_handle):
        if candidate in table:
            return table[candidate]

    return None


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------


def _read_nested(record: dict, parent_key: str, candidate_keys: List[str]) -> int:
    """Read a token count from a nested usage detail record (e.g. prompt_tokens_details)."""
    details = record.get(parent_key) or {}
    if not isinstance(details, dict):
        return 0
    for k in candidate_keys:
        v = details.get(k)
        if v:
            return v
    return 0


def _bill_record(
    pricing: ModelPricing,
    non_cached: int,
    cached: int,
    cache_write: int,
    completion: int,
) -> float:
    """Compute the dollar cost of a single LLM call given a ModelPricing entry."""
    total_input = non_cached + cached + cache_write
    use_tier = pricing.threshold_tokens is not None and total_input > pricing.threshold_tokens

    if use_tier and pricing.input_above_threshold_per_token is not None:
        in_rate = pricing.input_above_threshold_per_token
    else:
        in_rate = pricing.input_per_token

    if use_tier and pricing.output_above_threshold_per_token is not None:
        out_rate = pricing.output_above_threshold_per_token
    else:
        out_rate = pricing.output_per_token

    if use_tier and pricing.cache_read_above_threshold_per_token is not None:
        cache_read_rate = pricing.cache_read_above_threshold_per_token
    elif pricing.cache_read_per_token is not None:
        cache_read_rate = pricing.cache_read_per_token
    else:
        # No cache pricing available: bill cached input at full input rate.
        cache_read_rate = in_rate

    if use_tier and pricing.cache_creation_above_threshold_per_token is not None:
        cache_create_rate = pricing.cache_creation_above_threshold_per_token
    elif pricing.cache_creation_per_token is not None:
        cache_create_rate = pricing.cache_creation_per_token
    else:
        # OpenAI-style models don't have cache writes; rate is 0.
        cache_create_rate = 0.0

    return non_cached * in_rate + cached * cache_read_rate + cache_write * cache_create_rate + completion * out_rate


def calculate_cost_from_agent_usage(model_handle: str, agent_usage: Optional[List[dict]]) -> float:
    """Calculate total cost from agent_usage data.

    Bills cache reads, cache writes, and tiered (>200k context) pricing per LLM
    call, using rates from the litellm pricing JSON.

    Args:
        model_handle: Name of the model
        agent_usage: List of usage statistics from the agent run

    Returns:
        Total cost in dollars for the entire agent run, or 0.0 if model pricing
        is unavailable (logs a debug message).
    """
    if not agent_usage:
        return 0.0

    pricing = resolve_model(model_handle)
    if pricing is None:
        logger.debug(f"No pricing information available for model: {model_handle}")
        return 0.0

    total_cost = 0.0
    for record in agent_usage:
        if record.get("message_type") != "usage_statistics":
            continue

        prompt_tokens = record.get("prompt_tokens") or 0
        completion_tokens = record.get("completion_tokens") or 0

        cached_input = record.get("cached_input_tokens") or _read_nested(
            record,
            "prompt_tokens_details",
            ["cached_tokens", "cache_read_tokens", "cache_read_input_tokens", "cached_input_tokens"],
        )
        cache_write = record.get("cache_write_tokens") or _read_nested(
            record,
            "prompt_tokens_details",
            ["cache_creation_tokens", "cache_creation_input_tokens"],
        )

        # Letta normalizes prompt_tokens differently per provider:
        #   Anthropic: prompt_tokens = non_cached + cache_read + cache_write
        #   OpenAI:    prompt_tokens = non_cached + cache_read (no cache_write)
        # Subtracting both cached buckets gives non-cached input in either case.
        non_cached = max(prompt_tokens - cached_input - cache_write, 0)

        total_cost += _bill_record(pricing, non_cached, cached_input, cache_write, completion_tokens)

    return total_cost
