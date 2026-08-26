"""Tests for the suite-level target.pricing declaration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from letta_evals.models import (
    LettaCodeTargetSpec,
    MetricRewardSpec,
    ModelPricingSpec,
    SuiteSpec,
    ToolGraderSpec,
)
from letta_evals.types import GraderKind, RewardKind

# ---------------------------------------------------------------------------
# Validation and conversion
# ---------------------------------------------------------------------------


def test_explicit_rates_form_is_valid():
    spec = ModelPricingSpec(input_per_mtok=2.0, output_per_mtok=10.0, cache_read_per_mtok=0.2)
    assert spec.output_per_mtok == 10.0


def test_missing_output_rate_rejected():
    with pytest.raises(ValidationError):
        ModelPricingSpec(input_per_mtok=2.0)


def test_negative_rates_rejected():
    with pytest.raises(ValidationError):
        ModelPricingSpec(input_per_mtok=-1.0, output_per_mtok=10.0)


def test_to_entry_converts_per_mtok_to_per_token():
    entry = ModelPricingSpec(
        input_per_mtok=2.0, output_per_mtok=10.0, cache_read_per_mtok=0.2, cache_write_per_mtok=2.5
    ).to_entry()
    assert entry["input_cost_per_token"] == pytest.approx(2e-06)
    assert entry["output_cost_per_token"] == pytest.approx(1e-05)
    assert entry["cache_read_input_token_cost"] == pytest.approx(2e-07)
    assert entry["cache_creation_input_token_cost"] == pytest.approx(2.5e-06)


def test_to_entry_omits_unset_cache_rates():
    entry = ModelPricingSpec(input_per_mtok=2.0, output_per_mtok=10.0).to_entry()
    assert "cache_read_input_token_cost" not in entry
    assert "cache_creation_input_token_cost" not in entry


# ---------------------------------------------------------------------------
# Suite serialization round trip (what sandbox dispatch relies on)
# ---------------------------------------------------------------------------


def _make_suite(target: LettaCodeTargetSpec) -> SuiteSpec:
    return SuiteSpec(
        name="pricing-round-trip",
        dataset="fake.jsonl",
        target=target,
        graders={"accuracy": ToolGraderSpec(kind=GraderKind.TOOL, function="exact_match")},
        reward=MetricRewardSpec(kind=RewardKind.METRIC, metric_key="accuracy"),
    )


def test_pricing_survives_suite_round_trip():
    suite = _make_suite(
        LettaCodeTargetSpec(
            model_handles=["glm-4.6-openrouter", "my-private-model"],
            pricing={
                "glm-4.6-openrouter": "openrouter/z-ai/glm-4.6",
                "my-private-model": ModelPricingSpec(input_per_mtok=1.0, output_per_mtok=4.0),
            },
        )
    )
    restored = SuiteSpec.model_validate(suite.model_dump(mode="json"))
    assert restored.target.pricing == suite.target.pricing
    assert restored.target.pricing["glm-4.6-openrouter"] == "openrouter/z-ai/glm-4.6"
    assert restored.target.pricing["my-private-model"].to_entry()["input_cost_per_token"] == pytest.approx(1e-06)


def test_pricing_parses_from_yaml_shape():
    yaml_data = {
        "name": "pricing-yaml",
        "dataset": "fake.jsonl",
        "target": {
            "kind": "letta_code",
            "model_handles": ["glm-4.6-openrouter"],
            "pricing": {
                "glm-4.6-openrouter": "openrouter/z-ai/glm-4.6",
                "my-private-model": {"input_per_mtok": 1.0, "output_per_mtok": 4.0},
            },
        },
        "graders": {"accuracy": {"kind": "tool", "function": "exact_match"}},
        "reward": {"kind": "metric", "metric_key": "accuracy"},
    }
    suite = SuiteSpec.from_yaml(yaml_data)
    assert suite.target.pricing["glm-4.6-openrouter"] == "openrouter/z-ai/glm-4.6"
    assert suite.target.pricing["my-private-model"].output_per_mtok == 4.0
