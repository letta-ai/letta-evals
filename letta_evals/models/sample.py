"""Dataset sample model.

A :class:`Sample` is the unit of evaluation input: an ID, one or more user
inputs, optional ground truth, and optional grader/extractor variables.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, model_validator

SampleId = Union[int, str]


class Sample(BaseModel):
    """Single evaluation sample."""

    id: SampleId = Field(description="Sample ID from the dataset, or row index if the dataset does not provide one")
    input: Union[str, List[str]] = Field(description="Input message(s) to send to the agent")
    ground_truth: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Expected ground_truth response for grading. Can be a list for per-turn evaluation in multi-turn conversations.",
    )
    agent_args: Optional[Dict[str, Any]] = Field(default=None, description="Custom arguments for agent creation")
    rubric_vars: Optional[Dict[str, Any]] = Field(
        default=None, description="Variables for prompt substitution in rubric graders"
    )
    extra_vars: Optional[Dict[str, Any]] = Field(
        default=None, description="Custom user-supplied variables. Useful when writing custom extractors, graders, etc."
    )
    rubric: Optional[str] = Field(
        default=None,
        description=(
            "Per-sample rubric text. When set, overrides the grader-level rubric "
            "(e.g. ``prompt`` / ``prompt_path``) for this sample. The rubric is "
            "sent verbatim to the judge after template substitution."
        ),
    )
    rubric_path: Optional[Path] = Field(
        default=None,
        description=(
            "Per-sample rubric file path. Resolved by the dataset loader, which "
            "reads the file contents into ``rubric``. Only one of ``rubric`` or "
            "``rubric_path`` may be set per sample."
        ),
    )

    @model_validator(mode="after")
    def validate_ground_truth_format(self):
        """Validate ground_truth format matches input format."""
        # Reject str input with list ground_truth (doesn't make sense)
        if isinstance(self.ground_truth, list) and not isinstance(self.input, list):
            raise ValueError("ground_truth cannot be a list when input is a string")

        # Ensure lengths match when both are lists
        if isinstance(self.input, list) and isinstance(self.ground_truth, list):
            if len(self.input) != len(self.ground_truth):
                raise ValueError(
                    f"input has {len(self.input)} items but ground_truth has {len(self.ground_truth)} items. "
                    f"For per-turn evaluation, each input must have a corresponding ground_truth."
                )

        return self

    @model_validator(mode="after")
    def validate_rubric_fields(self):
        """At most one of rubric / rubric_path may be set."""
        if self.rubric is not None and self.rubric_path is not None:
            raise ValueError(
                "Sample cannot have both 'rubric' and 'rubric_path' set. "
                "Use one or the other (the loader resolves rubric_path into rubric)."
            )
        return self
