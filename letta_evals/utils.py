import importlib.util
import logging
import re
import sys
from pathlib import Path
from typing import Any, List, Optional

from letta_evals.constants import (
    MODEL_COSTS,
    MODEL_NAME_MAPPING,
    TURN_FAIL_SYMBOL,
    TURN_PASS_SYMBOL,
    TURN_PENDING_SYMBOL,
)

# Pattern to match reasoning effort suffixes
_EFFORT_PATTERN = re.compile(r"-(low|medium|high|xhigh|max)$")
from letta_evals.models import Sample

logger = logging.getLogger(__name__)


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
