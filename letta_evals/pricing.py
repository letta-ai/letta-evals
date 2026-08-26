"""Cost computation at litellm catalog rates.

Rates for a model handle come from, in order:

1. A suite-declared ``target.pricing`` entry: an exact litellm catalog key,
   or explicit rates. Explicit config never falls through to guessing — a
   catalog key that isn't in the catalog is unknown cost.
2. The model behind the handle as the server reports it in the agent's
   ``llm_config`` (model name, endpoint type, provider name). The server has
   already resolved runtime selectors — effort suffixes, provider routes —
   to the underlying model, so new handles price themselves.

There is no handle-string guessing: without either source, cost is ``None``
("unknown"), never ``0.0``. ``calculate_cost_from_agent_usage`` never raises;
catalog load failures also degrade to unknown cost.

The catalog is litellm's pricing JSON, fetched once per process and disk-cached
for 30 minutes (a stale cache is used when the refetch fails). Entries are kept
as litellm's raw dicts: ``input_cost_per_token``, ``output_cost_per_token``,
``cache_read_input_token_cost``, ``cache_creation_input_token_cost``, and their
``*_above_200k_tokens`` tier variants.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import httpx

logger = logging.getLogger(__name__)

LITELLM_PRICING_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
CACHE_FILE = (
    Path(os.environ.get("LETTA_EVALS_CACHE_DIR", str(Path.home() / ".cache" / "letta_evals"))) / "model_prices.json"
)
CACHE_TTL_SECONDS = 30 * 60  # 30 minutes
NETWORK_TIMEOUT_SECONDS = 5

# A target.pricing entry: an exact litellm catalog key, or a litellm-shaped
# rates dict (ModelPricingSpec.to_entry() produces the latter).
PricingOverrides = Dict[str, Union[str, Dict[str, float]]]

# Parsed catalog, loaded lazily, plus a lowercase index (catalog casing is
# inconsistent with what providers report, e.g. litellm's `minimax/MiniMax-M3`
# vs a server-reported `minimax/minimax-m3`; exact matches win).
_TABLE: Optional[Dict[str, dict]] = None
_TABLE_LOWER: Optional[Dict[str, dict]] = None

# Keys already warned about in this process, so unresolved pricing logs once
# per handle rather than once per sample.
_WARNED: set = set()

# litellm prefixes to try for an endpoint type, after the provider-name prefix.
# Needed where litellm's namespace differs from Letta's (bare keys for openai/
# anthropic, gemini/ for google) and for BYOK providers whose custom
# provider_name matches nothing in the catalog.
_ENDPOINT_TYPE_PREFIXES: Dict[str, List[str]] = {
    "openai": ["", "openai/"],
    "anthropic": ["", "anthropic/"],
    "google_ai": ["", "gemini/", "vertex_ai/"],
    "google_vertex": ["vertex_ai/", "gemini/", ""],
    "azure": ["azure/"],
    "groq": ["groq/"],
    "mistral": ["mistral/"],
    "together": ["together_ai/"],
    "bedrock": ["bedrock/", ""],
    "deepseek": ["deepseek/", ""],
    "xai": ["xai/"],
}


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(message)


def _fetch_upstream() -> dict:
    """Fetch the litellm pricing JSON. Raises on network errors."""
    resp = httpx.get(LITELLM_PRICING_URL, timeout=NETWORK_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()


def _read_cache(max_age: Optional[float]) -> Optional[dict]:
    """Read the disk cache; None if absent, unreadable, or older than max_age."""
    try:
        if max_age is not None and time.time() - CACHE_FILE.stat().st_mtime >= max_age:
            return None
        return json.loads(CACHE_FILE.read_text())
    except (OSError, ValueError):
        return None


def _write_cache(raw: dict) -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_FILE.with_suffix(".json.tmp")  # atomic: write temp, rename
        tmp.write_text(json.dumps(raw))
        tmp.replace(CACHE_FILE)
    except OSError as e:
        logger.warning(f"Failed to write pricing cache to {CACHE_FILE}: {e}")


def load_pricing_table() -> Dict[str, dict]:
    """The litellm catalog as raw per-model dicts, loaded once per process.

    Keeps only entries with both input and output rates (drops embeddings,
    audio, and litellm's sample_spec). Raises only when the fetch fails and no
    disk cache exists at all.
    """
    global _TABLE, _TABLE_LOWER
    if _TABLE is not None:
        return _TABLE

    raw = _read_cache(max_age=CACHE_TTL_SECONDS)
    if raw is None:
        try:
            raw = _fetch_upstream()
            _write_cache(raw)
        except Exception as err:
            raw = _read_cache(max_age=None)
            if raw is None:
                raise
            logger.warning(f"Failed to fetch pricing from upstream ({err}); using stale cache at {CACHE_FILE}")

    _TABLE = {
        key: entry
        for key, entry in raw.items()
        if isinstance(entry, dict)
        and entry.get("input_cost_per_token") is not None
        and entry.get("output_cost_per_token") is not None
    }
    _TABLE_LOWER = {}
    for key, entry in _TABLE.items():
        _TABLE_LOWER.setdefault(key.lower(), entry)
    return _TABLE


def _lookup(key: str) -> Optional[dict]:
    """Look up one catalog key: exact match first, then case-insensitive."""
    table = load_pricing_table()
    return table.get(key) or (_TABLE_LOWER or {}).get(key.lower())


def _find_by_identity(
    model_name: Optional[str], provider_type: Optional[str], provider_name: Optional[str]
) -> Optional[dict]:
    """Probe the catalog for a server-reported model name under the
    provider-name prefix, the endpoint-type prefixes, and bare.

    The name is probed exactly as reported — no suffix stripping: names that
    look decorated (o3-mini-high, qwen3-max, dated snapshots) are distinct
    catalog identities, and a near-miss priced as a different model is worse
    than an unknown cost.
    """
    name = (model_name or "").strip()
    if not name:
        return None

    prefixes: List[str] = []
    if provider_name:
        prefixes.append(provider_name.strip().lower() + "/")
    prefixes += _ENDPOINT_TYPE_PREFIXES.get(provider_type or "", [])
    if "" not in prefixes:
        prefixes.append("")

    for prefix in prefixes:
        entry = _lookup(prefix + name)
        if entry is not None:
            return entry
    return None


def _resolve_entry(
    model_handle: str,
    model_name: Optional[str],
    provider_type: Optional[str],
    provider_name: Optional[str],
    overrides: Optional[PricingOverrides],
) -> Optional[dict]:
    """Suite override first, then the server-reported identity."""
    override = (overrides or {}).get(model_handle)
    if isinstance(override, dict):
        return override
    if isinstance(override, str):
        entry = _lookup(override)
        if entry is None:
            _warn_once(
                f"override:{model_handle}",
                f"target.pricing maps '{model_handle}' to '{override}', which is not in the "
                f"pricing catalog; cost recorded as unknown",
            )
        # Explicit config never falls through to guessing.
        return entry
    return _find_by_identity(model_name, provider_type, provider_name)


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


def _rate(entry: dict, key: str, tier: bool, default: float = 0.0) -> float:
    """Per-token rate for one bucket, honoring the >200k tier variant when present."""
    if tier and entry.get(key + "_above_200k_tokens") is not None:
        return entry[key + "_above_200k_tokens"]
    v = entry.get(key)
    return v if v is not None else default


def calculate_cost_from_agent_usage(
    model_handle: str,
    agent_usage: Optional[List[dict]],
    *,
    model_name: Optional[str] = None,
    provider_type: Optional[str] = None,
    provider_name: Optional[str] = None,
    overrides: Optional[PricingOverrides] = None,
) -> Optional[float]:
    """Total dollar cost of an agent run, billing cache reads, cache writes,
    and tiered (>200k context) pricing per LLM call.

    ``model_name`` / ``provider_type`` / ``provider_name`` are the
    server-reported identity of the model behind the handle, from the agent's
    ``llm_config``.

    Returns 0.0 for an empty usage list, and None when rates cannot be
    resolved (cost unknown, warned once per handle). Never raises — a failure
    to load the pricing catalog also degrades to None.
    """
    if not agent_usage:
        return 0.0

    try:
        entry = _resolve_entry(model_handle, model_name, provider_type, provider_name, overrides)
    except Exception as e:
        _warn_once(
            f"catalog-error:{model_handle}",
            f"Could not load pricing catalog while pricing model '{model_handle}' ({e}); cost recorded as unknown",
        )
        return None
    if entry is None:
        _warn_once(
            model_handle,
            f"No pricing found for model '{model_handle}'; cost recorded as unknown. "
            f"Map it to a catalog entry or explicit rates via target.pricing in the suite YAML.",
        )
        return None

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

        tier = non_cached + cached_input + cache_write > 200_000
        in_rate = _rate(entry, "input_cost_per_token", tier)
        total_cost += (
            non_cached * in_rate
            # No cache-read pricing published: bill cached input at the full input rate.
            + cached_input * _rate(entry, "cache_read_input_token_cost", tier, default=in_rate)
            # OpenAI-style models have no cache writes; rate 0 when unpublished.
            + cache_write * _rate(entry, "cache_creation_input_token_cost", tier)
            + completion_tokens * _rate(entry, "output_cost_per_token", tier)
        )

    return total_cost
