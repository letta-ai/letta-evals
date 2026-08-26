"""Tests for letta_evals.pricing: rate resolution and cost calculation."""

from __future__ import annotations

from typing import Dict

import pytest

from letta_evals import pricing
from letta_evals.pricing import calculate_cost_from_agent_usage

# A trimmed fixture mirroring the schema in
# https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
# Costs are per-token (the upstream convention), matching real prices for the
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
    "gpt-5-2025-08-07": {
        "input_cost_per_token": 1.25e-06,
        "output_cost_per_token": 1e-05,
        "cache_read_input_token_cost": 1.25e-07,
    },
    "zai/glm-4.6": {
        "input_cost_per_token": 5e-07,
        "output_cost_per_token": 1.75e-06,
    },
    # Provider-routed entry.
    "openrouter/z-ai/glm-4.6": {
        "input_cost_per_token": 6e-07,
        "output_cost_per_token": 2e-06,
    },
    # Mixed-case catalog key (litellm carries e.g. minimax/MiniMax-M3 while
    # servers report minimax/minimax-m3); resolution is case-insensitive.
    "minimax/MiniMax-M3": {
        "input_cost_per_token": 3e-07,
        "output_cost_per_token": 1.2e-06,
    },
    # Embedding-only entry; should be dropped by the loader.
    "text-embedding-3-small": {
        "input_cost_per_token": 2e-08,
    },
    # The litellm JSON has a sample_spec key; loader should drop it.
    "sample_spec": {"note": "documentation only"},
}


@pytest.fixture(autouse=True)
def _reset_pricing_module():
    """Reset the module-level catalog cache and warn-once state between tests."""
    pricing._TABLE = None
    pricing._TABLE_LOWER = None
    pricing._WARNED.clear()
    yield
    pricing._TABLE = None
    pricing._TABLE_LOWER = None
    pricing._WARNED.clear()


@pytest.fixture(autouse=True)
def _stub_loader(monkeypatch):
    """Stub the fetch and disk cache so tests never touch the network."""
    monkeypatch.setattr(pricing, "_fetch_upstream", lambda: FIXTURE_RAW)
    monkeypatch.setattr(pricing, "_read_cache", lambda max_age: None)
    monkeypatch.setattr(pricing, "_write_cache", lambda raw: None)


def _usage(prompt: int, completion: int, *, cached: int = 0, cache_write: int = 0) -> dict:
    """Build a single usage_statistics record."""
    return {
        "message_type": "usage_statistics",
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_input_tokens": cached,
        "cache_write_tokens": cache_write,
    }


# Server-reported identities (from llm_config) used throughout. Billing 1M
# input tokens makes cost equal the input rate in dollars per Mtok.
SONNET = dict(model_name="claude-sonnet-4-5", provider_type="anthropic", provider_name="anthropic")
GPT5 = dict(model_name="gpt-5-2025-08-07", provider_type="openai", provider_name="openai")
GLM = dict(model_name="glm-4.6", provider_name="zai")
MTOK_INPUT = [_usage(1_000_000, 0)]


# ---------------------------------------------------------------------------
# Billing math
# ---------------------------------------------------------------------------


def test_simple_billing_no_cache():
    # 1000 input * $3e-06 + 500 output * $1.5e-05 = $0.003 + $0.0075 = $0.0105
    cost = calculate_cost_from_agent_usage("anthropic/claude-sonnet-4-5", [_usage(1000, 500)], **SONNET)
    assert cost == pytest.approx(0.0105)


def test_billing_with_anthropic_cache_read():
    # Anthropic semantics: prompt_tokens already INCLUDES cached.
    # 1000 cache_read at $3e-07 + 500 output at $1.5e-05 = $0.0003 + $0.0075 = $0.0078
    cost = calculate_cost_from_agent_usage("anthropic/claude-sonnet-4-5", [_usage(1000, 500, cached=1000)], **SONNET)
    assert cost == pytest.approx(0.0078)


def test_billing_with_anthropic_cache_write():
    # 500 non_cached * $3e-06 + 500 cache_write * $3.75e-06 + 200 output * $1.5e-05
    cost = calculate_cost_from_agent_usage(
        "anthropic/claude-sonnet-4-5", [_usage(prompt=1000, completion=200, cache_write=500)], **SONNET
    )
    assert cost == pytest.approx(0.0015 + 0.001875 + 0.003)


def test_billing_with_openai_cache_read():
    # OpenAI semantics: prompt_tokens = non_cached + cache_read (no cache_write).
    # 800 non_cached * $1.25e-06 + 200 cache_read * $1.25e-07 + 100 output * $1e-05
    cost = calculate_cost_from_agent_usage(
        "openai/gpt-5-2025-08-07", [_usage(prompt=1000, completion=100, cached=200)], **GPT5
    )
    assert cost == pytest.approx(0.001 + 0.000025 + 0.001)


def test_billing_falls_back_to_input_rate_when_no_cache_pricing():
    # zai/glm-4.6 fixture has no cache_read_input_token_cost; cached tokens
    # should bill at the regular input rate.
    cost = calculate_cost_from_agent_usage("z-ai/glm-4.6", [_usage(prompt=1000, completion=100, cached=200)], **GLM)
    expected = 800 * 5e-07 + 200 * 5e-07 + 100 * 1.75e-06
    assert cost == pytest.approx(expected)


def test_tiered_billing_above_threshold():
    # Sonnet 4.5 above 200k: input doubles to $6e-06, output to $2.25e-05.
    cost = calculate_cost_from_agent_usage(
        "anthropic/claude-sonnet-4-5", [_usage(prompt=250_000, completion=1000)], **SONNET
    )
    assert cost == pytest.approx(250_000 * 6e-06 + 1000 * 2.25e-05)


def test_tiered_billing_at_threshold_uses_base_rate():
    # Exactly 200k - not above, base rate applies.
    cost = calculate_cost_from_agent_usage(
        "anthropic/claude-sonnet-4-5", [_usage(prompt=200_000, completion=1000)], **SONNET
    )
    assert cost == pytest.approx(200_000 * 3e-06 + 1000 * 1.5e-05)


def test_empty_agent_usage_returns_zero():
    assert calculate_cost_from_agent_usage("anthropic/claude-opus-4-5", None) == 0.0
    assert calculate_cost_from_agent_usage("anthropic/claude-opus-4-5", []) == 0.0


def test_multi_record_aggregation():
    cost = calculate_cost_from_agent_usage(
        "anthropic/claude-sonnet-4-5", [_usage(1000, 500), _usage(2000, 100)], **SONNET
    )
    expected = (1000 * 3e-06 + 500 * 1.5e-05) + (2000 * 3e-06 + 100 * 1.5e-05)
    assert cost == pytest.approx(expected)


def test_skips_non_usage_records():
    cost = calculate_cost_from_agent_usage(
        "anthropic/claude-sonnet-4-5",
        [{"message_type": "stop_reason", "stop_reason": "end_turn"}, _usage(1000, 500)],
        **SONNET,
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
    cost = calculate_cost_from_agent_usage("openai/gpt-5-2025-08-07", [record], **GPT5)
    assert cost == pytest.approx(700 * 1.25e-06 + 300 * 1.25e-07 + 500 * 1e-05)


# ---------------------------------------------------------------------------
# Identity resolution (the server-reported model behind a handle)
# ---------------------------------------------------------------------------


def test_identity_prices_an_opaque_routing_handle():
    # The handle alone means nothing; provider_name becomes the catalog prefix.
    cost = calculate_cost_from_agent_usage(
        "glm-4.6-openrouter", MTOK_INPUT, model_name="z-ai/glm-4.6", provider_type="openai", provider_name="openrouter"
    )
    assert cost == pytest.approx(0.6)


def test_identity_endpoint_type_maps_to_bare_key():
    # BYOK providers have custom names; provider_type='anthropic' finds bare keys.
    cost = calculate_cost_from_agent_usage(
        "my-anthropic/claude-opus-4-5",
        MTOK_INPUT,
        model_name="claude-opus-4-5",
        provider_type="anthropic",
        provider_name="my-anthropic",
    )
    assert cost == pytest.approx(5.0)


def test_identity_names_match_exactly_no_suffix_stripping():
    # Dated snapshots resolve under their exact catalog key.
    cost = calculate_cost_from_agent_usage("h", MTOK_INPUT, model_name="gpt-5-2025-08-07", provider_type="openai")
    assert cost == pytest.approx(1.25)
    # A near-miss is unknown, never guessed: names that look decorated
    # (a different date, an -xhigh/-max tail) may be distinct models.
    assert (
        calculate_cost_from_agent_usage(
            "h2", MTOK_INPUT, model_name="claude-opus-4-5-20991231", provider_type="anthropic"
        )
        is None
    )
    assert (
        calculate_cost_from_agent_usage("h3", MTOK_INPUT, model_name="claude-opus-4-5-xhigh", provider_type="anthropic")
        is None
    )


def test_identity_matches_catalog_case_insensitively():
    # Server reports lowercase; catalog key is minimax/MiniMax-M3.
    cost = calculate_cost_from_agent_usage("minimax-m3", MTOK_INPUT, model_name="minimax-m3", provider_name="minimax")
    assert cost == pytest.approx(0.3)
    # Routed models embed the provider path in the name (openrouter reports
    # minimax/minimax-m3); the bare-name candidate matches case-insensitively.
    cost = calculate_cost_from_agent_usage(
        "minimax-m3", MTOK_INPUT, model_name="minimax/minimax-m3", provider_name="openrouter"
    )
    assert cost == pytest.approx(0.3)


def test_unknown_identity_or_no_identity_is_unknown_cost(caplog):
    # No guessing from the handle string, however plausible it looks.
    with caplog.at_level("WARNING", logger="letta_evals.pricing"):
        assert calculate_cost_from_agent_usage("anthropic/claude-opus-4-5", MTOK_INPUT) is None
        assert (
            calculate_cost_from_agent_usage("h", MTOK_INPUT, model_name="totally-fake", provider_type="openai") is None
        )
    assert "anthropic/claude-opus-4-5" in caplog.text


def test_unknown_pricing_warns_once_per_handle(caplog):
    with caplog.at_level("WARNING", logger="letta_evals.pricing"):
        calculate_cost_from_agent_usage("some-future-model", [_usage(1000, 500)])
        calculate_cost_from_agent_usage("some-future-model", [_usage(2000, 100)])
    assert caplog.text.count("some-future-model") == 1


def test_catalog_load_failure_degrades_to_unknown_cost(monkeypatch, caplog):
    # No cache + fetch failure raises from load_pricing_table, but the cost
    # calculation must degrade to unknown, never fail a sample.
    def boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(pricing, "_fetch_upstream", boom)
    with caplog.at_level("WARNING", logger="letta_evals.pricing"):
        cost = calculate_cost_from_agent_usage(
            "anthropic/claude-opus-4-5", [_usage(1000, 500)], model_name="claude-opus-4-5", provider_type="anthropic"
        )
    assert cost is None
    assert "network down" in caplog.text


# ---------------------------------------------------------------------------
# Overrides (target.pricing) and precedence
# ---------------------------------------------------------------------------


def test_override_catalog_key():
    cost = calculate_cost_from_agent_usage("my-alias", MTOK_INPUT, overrides={"my-alias": "claude-opus-4-5"})
    assert cost == pytest.approx(5.0)


def test_override_explicit_rates():
    overrides = {"private-model": {"input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06}}
    cost = calculate_cost_from_agent_usage("private-model", [_usage(1_000_000, 1_000_000)], overrides=overrides)
    assert cost == pytest.approx(3.0)


def test_override_wins_over_identity():
    cost = calculate_cost_from_agent_usage(
        "glm-4.6-openrouter",
        MTOK_INPUT,
        model_name="z-ai/glm-4.6",
        provider_name="openrouter",
        overrides={"glm-4.6-openrouter": "zai/glm-4.6"},
    )
    assert cost == pytest.approx(0.5)  # the zai/ entry, not the openrouter/ one


def test_override_bad_catalog_key_is_unknown_not_guessed(caplog):
    # Explicit config must fail loudly, not fall through to the identity.
    with caplog.at_level("WARNING", logger="letta_evals.pricing"):
        cost = calculate_cost_from_agent_usage(
            "anthropic/claude-opus-4-5",
            MTOK_INPUT,
            model_name="claude-opus-4-5",
            provider_type="anthropic",
            overrides={"anthropic/claude-opus-4-5": "not-a-real-key"},
        )
    assert cost is None
    assert "not-a-real-key" in caplog.text


# ---------------------------------------------------------------------------
# Loader behavior
# ---------------------------------------------------------------------------


def test_load_pricing_drops_sample_spec_and_embedding_only():
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
    pricing.load_pricing_table()
    pricing.load_pricing_table()
    pricing.load_pricing_table()
    assert fetch_count["n"] == 1


def test_load_falls_back_to_stale_cache_on_fetch_error(monkeypatch):
    def boom():
        raise RuntimeError("network down")

    # Fresh cache misses (max_age set); stale read (max_age=None) returns the fixture.
    monkeypatch.setattr(pricing, "_fetch_upstream", boom)
    monkeypatch.setattr(pricing, "_read_cache", lambda max_age: FIXTURE_RAW if max_age is None else None)

    table = pricing.load_pricing_table()
    assert "claude-opus-4-5" in table


def test_load_propagates_error_when_no_disk_cache(monkeypatch):
    def boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(pricing, "_fetch_upstream", boom)

    with pytest.raises(RuntimeError, match="network down"):
        pricing.load_pricing_table()
