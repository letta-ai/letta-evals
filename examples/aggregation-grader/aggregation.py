"""Custom aggregation function for combining multiple metrics."""

from typing import Dict

from letta_evals.decorators import aggregation


@aggregation
def weighted_average_aggregate(metrics: Dict[str, float]) -> float:
    """
    Aggregate multiple metrics with custom weights.

    Args:
        metrics: Dict[str, float] containing scores from dependent graders

    Returns:
        float: Aggregated score between 0.0 and 1.0
    """
    contains_score = metrics.get('contains_check', 0.0)
    exact_score = metrics.get('exact_check', 0.0)

    # Weighted average: 70% contains, 30% exact
    weighted_score = 0.7 * contains_score + 0.3 * exact_score

    return weighted_score
