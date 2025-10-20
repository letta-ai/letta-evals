from pathlib import Path
from typing import Callable

from letta_evals.decorators import EXTRACTOR_REGISTRY
from letta_evals.utils import load_object


def extractor_requires_agent_state(name: str, base_dir: Path = None) -> bool:
    """Check if an extractor requires agent_state parameter.

    Returns True if the extractor has 3 parameters (trajectory, config, agent_state).
    """
    if name in EXTRACTOR_REGISTRY:
        extractor_func = EXTRACTOR_REGISTRY[name]
        param_count = getattr(extractor_func, "_extractor_param_count", 2)
        return param_count == 3

    if ":" in name:
        obj = load_object(name, base_dir=base_dir)
        if callable(obj) and hasattr(obj, "_is_extractor"):
            param_count = getattr(obj, "_extractor_param_count", 2)
            return param_count == 3
        else:
            raise ValueError(
                f"Loaded object {name} is not a valid @extractor decorated function. "
                f"Please use the @extractor decorator."
            )

    raise ValueError(f"Unknown extractor: {name}")


def get_extractor(name: str, config: dict = None, base_dir: Path = None) -> Callable:
    """Get an extractor function by name or file path.

    Returns a callable that takes (trajectory, agent_state=None) and returns str.
    The wrapper handles both 2-param and 3-param extractors.
    """
    config = config or {}

    if name in EXTRACTOR_REGISTRY:
        extractor_func = EXTRACTOR_REGISTRY[name]
        param_count = getattr(extractor_func, "_extractor_param_count", 2)

        def wrapper(trajectory, agent_state=None):
            if param_count == 3:
                return extractor_func(trajectory, config, agent_state)
            else:
                return extractor_func(trajectory, config)

        return wrapper

    if ":" in name:
        obj = load_object(name, base_dir=base_dir)
        if callable(obj) and hasattr(obj, "_is_extractor"):
            param_count = getattr(obj, "_extractor_param_count", 2)

            def wrapper(trajectory, agent_state=None):
                if param_count == 3:
                    return obj(trajectory, config, agent_state)
                else:
                    return obj(trajectory, config)

            return wrapper
        else:
            raise ValueError(
                f"Loaded object {name} is not a valid @extractor decorated function. "
                f"Please use the @extractor decorator."
            )

    raise ValueError(f"Unknown extractor: {name}")
