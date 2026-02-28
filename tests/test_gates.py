"""Unit tests for gate-related helper functions."""

import pytest

from letta_evals.models import _compare, normalize_weights
from letta_evals.types import MetricOp

# ── _compare ──


class TestCompare:
    def test_gt(self):
        assert _compare(1.0, MetricOp.GT, 0.5) is True
        assert _compare(0.5, MetricOp.GT, 0.5) is False
        assert _compare(0.0, MetricOp.GT, 0.5) is False

    def test_gte(self):
        assert _compare(1.0, MetricOp.GTE, 0.5) is True
        assert _compare(0.5, MetricOp.GTE, 0.5) is True
        assert _compare(0.0, MetricOp.GTE, 0.5) is False

    def test_lt(self):
        assert _compare(0.0, MetricOp.LT, 0.5) is True
        assert _compare(0.5, MetricOp.LT, 0.5) is False

    def test_lte(self):
        assert _compare(0.0, MetricOp.LTE, 0.5) is True
        assert _compare(0.5, MetricOp.LTE, 0.5) is True
        assert _compare(1.0, MetricOp.LTE, 0.5) is False

    def test_eq(self):
        assert _compare(0.5, MetricOp.EQ, 0.5) is True
        assert _compare(0.6, MetricOp.EQ, 0.5) is False

    def test_boundary_values(self):
        assert _compare(0.0, MetricOp.GTE, 0.0) is True
        assert _compare(1.0, MetricOp.LTE, 1.0) is True
        assert _compare(0.0, MetricOp.GT, 0.0) is False


# ── normalize_weights ──


class TestNormalizeWeights:
    def test_basic(self):
        result = normalize_weights({"a": 1.0, "b": 1.0})
        assert result == {"a": 0.5, "b": 0.5}

    def test_unequal_weights(self):
        result = normalize_weights({"a": 3.0, "b": 1.0})
        assert result["a"] == pytest.approx(0.75)
        assert result["b"] == pytest.approx(0.25)

    def test_already_normalized(self):
        result = normalize_weights({"a": 0.6, "b": 0.4})
        assert result["a"] == pytest.approx(0.6)
        assert result["b"] == pytest.approx(0.4)

    def test_single_weight(self):
        result = normalize_weights({"only": 5.0})
        assert result == {"only": 1.0}

    def test_zero_weights_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            normalize_weights({"a": 0.0, "b": 0.0})

    def test_preserves_keys(self):
        result = normalize_weights({"accuracy": 0.6, "quality": 0.4})
        assert set(result.keys()) == {"accuracy", "quality"}

    def test_negative_weights_accepted(self):
        """Negative weights are allowed as long as sum is non-zero."""
        result = normalize_weights({"a": -1.0, "b": 3.0})
        assert result["a"] == pytest.approx(-0.5)
        assert result["b"] == pytest.approx(1.5)

    def test_negative_weights_zero_sum_raises(self):
        """Negative weights that cancel out to zero should raise."""
        with pytest.raises(ValueError, match="non-zero"):
            normalize_weights({"a": 1.0, "b": -1.0})
