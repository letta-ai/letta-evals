from pathlib import Path
from typing import Callable

from letta_evals.decorators import EXTRACTOR_REGISTRY
from letta_evals.utils import load_object


def get_extractor(name: str, config: dict = None, base_dir: Path = None) -> Callable:
    """Get an extractor function by name or file path.

    Returns a callable that takes (trajectory, context) and returns str.
    The wrapper handles both 2-param and 3-param extractors.
    """
    config = config or {}

    if name in EXTRACTOR_REGISTRY:
        extractor_func = EXTRACTOR_REGISTRY[name]
    elif ":" in name:
        extractor_func = load_object(name, base_dir=base_dir)
        if not (callable(extractor_func) and hasattr(extractor_func, "_is_extractor")):
            raise ValueError(
                f"Loaded object {name} is not a valid @extractor decorated function. "
                f"Please use the @extractor decorator."
            )
    else:
        raise ValueError(f"Unknown extractor: {name}")

    param_count = getattr(extractor_func, "_extractor_param_count", 2)

    def wrapper(trajectory, context):
        if param_count == 3:
            return extractor_func(trajectory, config, context)
        return extractor_func(trajectory, config)

    return wrapper
