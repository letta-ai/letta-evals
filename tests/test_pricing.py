"""Tests for letta_evals.pricing and the cache-aware cost calculation."""

from __future__ import annotations

from typing import Dict

import pytest

from letta_evals import pricing
from letta_evals.pricing import ModelPricing, calculate_cost_from_agent_usage, resolve_model

# A trimmed fixture mirroring the schema in
# https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
# Costs are per-token (the upstream convention). Matching the real prices for the
# most common models we use today.
FIXTURE_RAW: Dict[str, dict] = {
    "claude-opus-4-5": {
        "input_cost_per_token": 5e-06,
        "output_cost_per_token": 2.5e-05,
        "cache_read_input_token_cost": 5e-07,
        "cache_creation_input_token_cost": 6.25e-06,
    },
    "claude-sonnet-4-5": {
        "input_cost_per_token": 3e-06,
        "output_cost_per_token": 1.5e-05,
        "cache_read_input_token_cost": 3e-07,
        "cache_creation_input_token_cost": 3.75e-06,
        # Tiered: doubles above 200k context
        "input_cost_per_token_above_200k_tokens": 6e-06,
        "output_cost_per_token_above_200k_tokens": 2.25e-05,
        "cache_read_input_token_cost_above_200k_tokens": 6e-07,
        "cache_creation_input_token_cost_above_200k_tokens": 7.5e-06,
    },
    "claude-sonnet-4-5-20250929": {
        # Date-stamped variant: same prices.
        "input_cost_per_token": 3e-06,
        "output_cost_per_token": 1.5e-05,
        "cache_read_input_token_cost": 3e-07,
        "cache_creation_input_token_cost": 3.75e-06,
    },
    "gpt-5-2025-08-07": {
        "input_cost_per_token": 1.25e-06,
        "output_cost_per_token": 1e-05,
        "cache_read_input_token_cost": 1.25e-07,
    },
    "gemini-3-pro-preview": {
        "input_cost_per_token": 2e-06,
        "output_cost_per_token": 1.2e-05,
        "cache_read_input_token_cost": 2e-07,
    },
    "deepseek/deepseek-chat": {
        "input_cost_per_token": 2.8e-07,
        "output_cost_per_token": 4.2e-07,
        "cache_read_input_token_cost": 2.8e-08,
    },
    "zai/glm-4.6": {
        "input_cost_per_token": 5e-07,
        "output_cost_per_token": 1.75e-06,
    },
    # Embedding-only entry; should be skipped by the loader.
    "text-embedding-3-small": {
        "input_cost_per_token": 2e-08,
    },
    # The litellm JSON has a sample_spec key; loader should skip it.
    "sample_spec": {"note": "documentation only"},
}


@pytest.fixture(autouse=True)
def _reset_pricing_module():
    """Reset module-level pricing cache and overrides between tests."""
    pricing._PRICING = None
    saved_overrides = dict(pricing.MODEL_PRICE_OVERRIDES)
    yield
    pricing._PRICING = None
    pricing.MODEL_PRICE_OVERRIDES.clear()
    pricing.MODEL_PRICE_OVERRIDES.update(saved_overrides)


@pytest.fixture(autouse=True)
def _stub_loader(monkeypatch):
    """Stub _fetch_upstream and disk cache so tests never touch the network."""
    monkeypatch.setattr(pricing, "_fetch_upstream", lambda: FIXTURE_RAW)
    monkeypatch.setattr(pricing, "_read_disk_cache", lambda: None)
    monkeypatch.setattr(pricing, "_write_disk_cache", lambda raw: None)
    monkeypatch.setattr(pricing, "_cache_is_fresh", lambda: False)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def test_resolve_anthropic_provider_prefix():
    p = resolve_model("anthropic/claude-opus-4-5")
    assert p is not None
    assert p.input_per_token == 5e-06
    assert p.output_per_token == 2.5e-05


def test_resolve_anthropic_with_date_suffix():
    p = resolve_model("anthropic/claude-sonnet-4-5-20250929")
    assert p is not None
    assert p.input_per_token == 3e-06


def test_resolve_anthropic_strips_date_to_match_undated_key():
    # The fixture has `claude-sonnet-4-5` (undated); query with a different date
    # should fall back to the undated key after date stripping.
    p = resolve_model("anthropic/claude-sonnet-4-5-20991231")
    assert p is not None
    assert p.input_per_token == 3e-06


def test_resolve_openai_with_date_suffix():
    p = resolve_model("openai/gpt-5-2025-08-07")
    assert p is not None
    assert p.input_per_token == 1.25e-06


def test_resolve_google_ai_to_bare():
    p = resolve_model("google_ai/gemini-3-pro-preview")
    assert p is not None
    assert p.input_per_token == 2e-06


def test_resolve_z_ai_remaps_to_zai_prefix():
    p = resolve_model("z-ai/glm-4.6")
    assert p is not None
    assert p.input_per_token == 5e-07


def test_resolve_deepseek_keeps_provider_prefix():
    p = resolve_model("deepseek/deepseek-chat")
    assert p is not None
    assert p.input_per_token == 2.8e-07


def test_resolve_strips_effort_suffix():
    p = resolve_model("anthropic/claude-opus-4-5-high")
    assert p is not None
    assert p.input_per_token == 5e-06

    p = resolve_model("openai/gpt-5-2025-08-07-xhigh")
    assert p is not None
    assert p.input_per_token == 1.25e-06


def test_resolve_unknown_model_returns_none():
    assert resolve_model("anthropic/totally-fake-model") is None
    assert resolve_model("") is None


def test_resolve_skips_embedding_only_entry_via_explicit_match():
    # Embedding entries lack output_cost_per_token; they're filtered out.
    assert resolve_model("text-embedding-3-small") is None


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------

def test_override_takes_precedence_over_json():
    pricing.MODEL_PRICE_OVERRIDES["anthropic/claude-opus-4-5"] = ModelPricing(
        input_per_token=999e-06,
        output_per_token=999e-06,
    )
    p = resolve_model("anthropic/claude-opus-4-5")
    assert p is not None
    assert p.input_per_token == 999e-06


def test_override_for_internal_model_not_in_json():
    pricing.MODEL_PRICE_OVERRIDES["openai/gpt-5.99-internal"] = ModelPricing(
        input_per_token=1e-06,
        output_per_token=2e-06,
    )
    p = resolve_model("openai/gpt-5.99-internal")
    assert p is not None
    assert p.output_per_token == 2e-06


# ---------------------------------------------------------------------------
# Cost billing
# ---------------------------------------------------------------------------

def _usage(prompt: int, completion: int, *, cached: int = 0, cache_write: int = 0) -> dict:
    """Build a single usage_statistics record."""
    return {
        "message_type": "usage_statistics",
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_input_tokens": cached,
        "cache_write_tokens": cache_write,
    }


def test_simple_billing_no_cache():
    # 1000 input * $3e-06 + 500 output * $1.5e-05 = $0.003 + $0.0075 = $0.0105
    cost = calculate_cost_from_agent_usage(
        "anthropic/claude-sonnet-4-5",
        [_usage(1000, 500)],
    )
    assert cost == pytest.approx(0.0105)


def test_billing_with_anthropic_cache_read():
    # Anthropic semantics: prompt_tokens already INCLUDES cached.
    # 1000 cache_read at $3e-07 + 500 output at $1.5e-05 = $0.0003 + $0.0075 = $0.0078
    cost = calculate_cost_from_agent_usage(
        "anthropic/claude-sonnet-4-5",
        [_usage(1000, 500, cached=1000)],
    )
    assert cost == pytest.approx(0.0078)


def test_billing_with_anthropic_cache_write():
    # 500 non_cached * $3e-06 + 500 cache_write * $3.75e-06 + 200 output * $1.5e-05
    # = $0.0015 + $0.001875 + $0.003 = $0.006375
    cost = calculate_cost_from_agent_usage(
        "anthropic/claude-sonnet-4-5",
        [_usage(prompt=1000, completion=200, cached=0, cache_write=500)],
    )
    assert cost == pytest.approx(0.0015 + 0.001875 + 0.003)


def test_billing_with_openai_cache_read():
    # OpenAI semantics: prompt_tokens = non_cached + cache_read (no cache_write).
    # 800 non_cached * $1.25e-06 + 200 cache_read * $1.25e-07 + 100 output * $1e-05
    # = $0.001 + $0.000025 + $0.001 = $0.002025
    cost = calculate_cost_from_agent_usage(
        "openai/gpt-5-2025-08-07",
        [_usage(prompt=1000, completion=100, cached=200)],
    )
    assert cost == pytest.approx(0.001 + 0.000025 + 0.001)


def test_billing_falls_back_to_input_rate_when_no_cache_pricing():
    # zai/glm-4.6 fixture has no cache_read_input_token_cost; cached tokens
    # should bill at the regular input rate.
    cost = calculate_cost_from_agent_usage(
        "z-ai/glm-4.6",
        [_usage(prompt=1000, completion=100, cached=200)],
    )
    # 800 non_cached * $5e-07 + 200 cached * $5e-07 (fallback) + 100 * $1.75e-06
    expected = 800 * 5e-07 + 200 * 5e-07 + 100 * 1.75e-06
    assert cost == pytest.approx(expected)


def test_tiered_billing_above_threshold():
    # Sonnet 4.5 above 200k: input doubles to $6e-06, output to $2.25e-05.
    # 250_000 prompt (all non-cached) * $6e-06 + 1000 output * $2.25e-05
    # = $1.5 + $0.0225 = $1.5225
    cost = calculate_cost_from_agent_usage(
        "anthropic/claude-sonnet-4-5",
        [_usage(prompt=250_000, completion=1000)],
    )
    assert cost == pytest.approx(250_000 * 6e-06 + 1000 * 2.25e-05)


def test_tiered_billing_at_threshold_uses_base_rate():
    # Exactly 200k - not above, base rate applies.
    cost = calculate_cost_from_agent_usage(
        "anthropic/claude-sonnet-4-5",
        [_usage(prompt=200_000, completion=1000)],
    )
    assert cost == pytest.approx(200_000 * 3e-06 + 1000 * 1.5e-05)


def test_unknown_model_returns_zero():
    cost = calculate_cost_from_agent_usage(
        "anthropic/some-future-model",
        [_usage(1000, 500)],
    )
    assert cost == 0.0


def test_empty_agent_usage_returns_zero():
    assert calculate_cost_from_agent_usage("anthropic/claude-opus-4-5", None) == 0.0
    assert calculate_cost_from_agent_usage("anthropic/claude-opus-4-5", []) == 0.0


def test_multi_record_aggregation():
    # Two LLM calls in the same run.
    cost = calculate_cost_from_agent_usage(
        "anthropic/claude-sonnet-4-5",
        [
            _usage(1000, 500),
            _usage(2000, 100),
        ],
    )
    expected = (1000 * 3e-06 + 500 * 1.5e-05) + (2000 * 3e-06 + 100 * 1.5e-05)
    assert cost == pytest.approx(expected)


def test_skips_non_usage_records():
    # Only message_type == 'usage_statistics' is billed.
    cost = calculate_cost_from_agent_usage(
        "anthropic/claude-sonnet-4-5",
        [
            {"message_type": "stop_reason", "stop_reason": "end_turn"},
            _usage(1000, 500),
        ],
    )
    assert cost == pytest.approx(1000 * 3e-06 + 500 * 1.5e-05)


def test_reads_nested_cache_tokens_from_prompt_details():
    # If cached_input_tokens isn't on the top-level record, we read it from
    # prompt_tokens_details (under any of several provider-specific keys).
    record = {
        "message_type": "usage_statistics",
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "prompt_tokens_details": {"cached_tokens": 300},
    }
    cost = calculate_cost_from_agent_usage("openai/gpt-5-2025-08-07", [record])
    expected = 700 * 1.25e-06 + 300 * 1.25e-07 + 500 * 1e-05
    assert cost == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Loader behavior
# ---------------------------------------------------------------------------

def test_load_pricing_skips_sample_spec_and_embedding_only():
    table = pricing.load_pricing_table()
    assert "sample_spec" not in table
    assert "text-embedding-3-small" not in table
    assert "claude-opus-4-5" in table


def test_load_pricing_caches_in_process(monkeypatch):
    fetch_count = {"n": 0}

    def counting_fetch():
        fetch_count["n"] += 1
        return FIXTURE_RAW

    monkeypatch.setattr(pricing, "_fetch_upstream", counting_fetch)
    pricing._PRICING = None

    pricing.load_pricing_table()
    pricing.load_pricing_table()
    pricing.load_pricing_table()
    assert fetch_count["n"] == 1


def test_load_falls_back_to_stale_cache_on_fetch_error(monkeypatch):
    pricing._PRICING = None

    def boom():
        raise RuntimeError("network down")

    # Simulate stale cache returning the fixture
    monkeypatch.setattr(pricing, "_fetch_upstream", boom)
    monkeypatch.setattr(pricing, "_read_disk_cache", lambda: FIXTURE_RAW)
    monkeypatch.setattr(pricing, "_cache_is_fresh", lambda: False)

    table = pricing.load_pricing_table()
    assert "claude-opus-4-5" in table


def test_load_propagates_error_when_no_disk_cache(monkeypatch):
    pricing._PRICING = None

    def boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(pricing, "_fetch_upstream", boom)
    monkeypatch.setattr(pricing, "_read_disk_cache", lambda: None)
    monkeypatch.setattr(pricing, "_cache_is_fresh", lambda: False)

    with pytest.raises(RuntimeError, match="network down"):
        pricing.load_pricing_table()
