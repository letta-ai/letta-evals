from pathlib import Path
from typing import Callable

from letta_evals.decorators import EXTRACTOR_REGISTRY
from letta_evals.utils.module_loader import load_object


def get_extractor(name: str, config: dict = None, base_dir: Path = None) -> Callable:
    """Get an extractor function by name or file path.

    Returns a callable that takes (trajectory, config) and returns str.
    """
    config = config or {}

    if name in EXTRACTOR_REGISTRY:
        extractor_func = EXTRACTOR_REGISTRY[name]

        def wrapper(trajectory):
            return extractor_func(trajectory, config)

        return wrapper

    if ":" in name:
        obj = load_object(name, base_dir=base_dir)
        if callable(obj) and hasattr(obj, "_is_extractor"):

            def wrapper(trajectory):
                return obj(trajectory, config)

            return wrapper
        else:
            raise ValueError(
                f"Loaded object {name} is not a valid @extractor decorated function. "
                f"Please use the @extractor decorator."
            )

    raise ValueError(f"Unknown extractor: {name}")
