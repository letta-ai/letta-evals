import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional

from letta_evals.constants import MODEL_ALIASES, MODEL_COSTS

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
    Normalize model names by resolving aliases and adding provider prefixes.

    Args:
        model_name: Raw model name from results

    Returns:
        Normalized model name that can be found in MODEL_COSTS
    """
    # First, check if it's a direct alias match
    if model_name in MODEL_ALIASES:
        return MODEL_ALIASES[model_name]

    # Then try with provider prefix added
    if model_name.startswith("claude"):
        prefixed = f"anthropic/{model_name}"
        if prefixed in MODEL_COSTS:
            return prefixed
    elif model_name.startswith("gpt"):
        prefixed = f"openai/{model_name}"
        if prefixed in MODEL_COSTS:
            return prefixed

    # Return as-is if no normalization found
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


def extract_token_counts(agent_usage: Optional[List[dict]]) -> tuple[int, int]:
    """
    Extract total token counts from agent_usage data.

    Args:
        agent_usage: List of usage statistics from the agent run

    Returns:
        Tuple of (total_prompt_tokens, total_completion_tokens)
    """
    if not agent_usage:
        return 0, 0

    total_prompt_tokens = 0
    total_completion_tokens = 0

    for usage_record in agent_usage:
        if usage_record.get("message_type") == "usage_statistics":
            total_prompt_tokens += usage_record.get("prompt_tokens", 0)
            total_completion_tokens += usage_record.get("completion_tokens", 0)

    return total_prompt_tokens, total_completion_tokens


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
            prompt_tokens = usage_record.get("prompt_tokens", 0)
            completion_tokens = usage_record.get("completion_tokens", 0)
            total_cost += calculate_cost(model_name, prompt_tokens, completion_tokens)

    return total_cost
