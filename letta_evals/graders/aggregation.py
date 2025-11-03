"""Aggregation grader that combines multiple metrics using custom Python code."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from letta_client import AgentState, LettaMessageUnion

from letta_evals.decorators import AGGREGATION_REGISTRY
from letta_evals.graders.base import Grader
from letta_evals.models import GradeResult, Sample
from letta_evals.utils import load_object


class AggregationGrader(Grader):
    """Aggregation grader that combines multiple metrics using custom Python code.

    This grader does not process trajectories directly. Instead, it waits for other
    graders to complete and then aggregates their scores using custom Python code.
    """

    def __init__(
        self,
        function: str,
        depends_on: List[str],
        base_dir: Optional[Path] = None,
    ):
        """Initialize aggregation grader.

        Args:
            function: Path to Python file containing aggregation function (e.g., 'aggregation.py:my_aggregate')
            depends_on: List of metric keys (grader names) that this aggregation depends on
            base_dir: Base directory for resolving relative paths
        """
        self.function_name = function
        self.depends_on = depends_on
        self.base_dir = base_dir
        self._aggregate_func = None
        self._load_function()

    def _load_function(self):
        """Load the aggregation function from registry or file path."""
        # First check if it's in the registry (like ToolGrader does)
        if self.function_name in AGGREGATION_REGISTRY:
            self._aggregate_func = AGGREGATION_REGISTRY[self.function_name]
        elif ":" in self.function_name:
            # Load from file path
            func = load_object(self.function_name, base_dir=self.base_dir)

            if not callable(func):
                raise ValueError(f"Loaded object {self.function_name} is not callable")

            # Check if it has the decorator marker (optional but recommended)
            if not hasattr(func, "_is_aggregation"):
                # Allow it but maybe warn in the future
                pass

            self._aggregate_func = func
        else:
            raise ValueError(
                f"Aggregation function '{self.function_name}' not found in registry. "
                f"Either use @aggregation decorator or specify as 'file.py:function_name'"
            )

    @property
    def requires_agent_state(self) -> bool:
        """Aggregation grader does not require agent state."""
        return False

    async def grade(
        self,
        sample: Sample,
        trajectory: List[List[LettaMessageUnion]],
        agent_state: Optional[AgentState] = None,
        dependent_grades: Optional[Dict[str, GradeResult]] = None,
    ) -> Tuple[GradeResult, str]:
        """Grade by aggregating scores from dependent graders.

        Args:
            sample: The sample being graded (not used directly)
            trajectory: The agent trajectory (not used directly)
            agent_state: The agent state (not used)
            dependent_grades: Dictionary mapping metric keys to their GradeResults

        Returns:
            Tuple of (GradeResult with aggregated score, empty string for submission)
        """
        if dependent_grades is None:
            raise ValueError("AggregationGrader requires dependent_grades to be provided")

        # Check that all dependencies are present
        missing = set(self.depends_on) - set(dependent_grades.keys())
        if missing:
            raise ValueError(f"Missing required metrics for aggregation: {missing}")

        # Extract scores from dependent grades
        metrics = {key: dependent_grades[key].score for key in self.depends_on}

        try:
            aggregated_score = self._aggregate_func(metrics)
            if not isinstance(aggregated_score, (int, float)):
                raise ValueError(f"Aggregate function must return a number, got {type(aggregated_score)}")

            aggregated_score = float(aggregated_score)
            if not (0.0 <= aggregated_score <= 1.0):
                raise ValueError(f"Aggregate function must return a value between 0.0 and 1.0, got {aggregated_score}")

            rationale = f"Aggregated from metrics: {', '.join(f'{k}={v:.3f}' for k, v in metrics.items())} -> {aggregated_score:.3f}"

            return GradeResult(
                score=aggregated_score,
                rationale=rationale,
                metadata={"input_metrics": metrics}
            ), ""

        except Exception as e:
            raise ValueError(f"Error executing aggregate function: {e}")