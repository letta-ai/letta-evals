import inspect
from functools import wraps
from typing import Callable, Dict

from letta_evals.models import GradeResult

GRADER_REGISTRY: Dict[str, Callable] = {}
EXTRACTOR_REGISTRY: Dict[str, Callable] = {}


def grader(func: Callable = None, *, name: str = None):
    """
    Decorator for grader functions.

    Validates that the function has signature: (Sample, str) -> GradeResult
    Auto-registers the function to the grader registry.

    Usage:
        @grader
        def my_grader(sample: Sample, submission: str) -> GradeResult:
            ...

        @grader(name="custom_name")
        def another_grader(sample: Sample, submission: str) -> GradeResult:
            ...
    """

    def decorator(f: Callable) -> Callable:
        sig = inspect.signature(f)
        params = list(sig.parameters.values())

        if len(params) != 2:
            raise TypeError(
                f"Grader {f.__name__} must have exactly 2 parameters (sample: Sample, submission: str), "
                f"got {len(params)}"
            )

        param_names = [p.name for p in params]
        if param_names != ["sample", "submission"]:
            raise TypeError(
                f"Grader {f.__name__} must have parameters named 'sample' and 'submission', " f"got {param_names}"
            )

        if sig.return_annotation != inspect.Signature.empty:
            if sig.return_annotation != GradeResult:
                raise TypeError(f"Grader {f.__name__} must return GradeResult, " f"got {sig.return_annotation}")

        registry_name = name or f.__name__
        GRADER_REGISTRY[registry_name] = f

        f._is_grader = True

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


def extractor(func: Callable = None, *, name: str = None):
    """
    Decorator for extractor functions.

    Validates that the function has signature: (trajectory: List[List[LettaMessageUnion]], config: dict) -> str
    Auto-registers the function to the extractor registry.

    Usage:
        @extractor
        def my_extractor(trajectory: List[List[LettaMessageUnion]], config: dict) -> str:
            ...

        @extractor(name="custom_name")
        def another_extractor(trajectory: List[List[LettaMessageUnion]], config: dict) -> str:
            ...
    """

    def decorator(f: Callable) -> Callable:
        sig = inspect.signature(f)
        params = list(sig.parameters.values())

        if len(params) != 2:
            raise TypeError(
                f"Extractor {f.__name__} must have exactly 2 parameters (trajectory, config), " f"got {len(params)}"
            )

        param_names = [p.name for p in params]
        if param_names != ["trajectory", "config"]:
            raise TypeError(
                f"Extractor {f.__name__} must have parameters named 'trajectory' and 'config', " f"got {param_names}"
            )

        if sig.return_annotation != inspect.Signature.empty:
            if sig.return_annotation is not str:
                raise TypeError(f"Extractor {f.__name__} must return str, " f"got {sig.return_annotation}")

        registry_name = name or f.__name__
        EXTRACTOR_REGISTRY[registry_name] = f

        f._is_extractor = True

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


def agent_factory(func: Callable) -> Callable:
    """
    Decorator for agent factory functions.

    Validates that the function has signature: async (client: AsyncLetta) -> str

    Usage:
        @agent_factory
        async def create_inventory_agent(client: AsyncLetta) -> str:
            # create agent using client
            return agent_id
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    # check parameter count
    if len(params) != 1:
        raise TypeError(f"Agent factory {func.__name__} must have exactly 1 parameter (client), " f"got {len(params)}")

    # check parameter name
    param_name = params[0].name
    if param_name != "client":
        raise TypeError(f"Agent factory {func.__name__} must have parameter named 'client', " f"got '{param_name}'")

    # check if it's async
    if not inspect.iscoroutinefunction(func):
        raise TypeError(f"Agent factory {func.__name__} must be an async function")

    # check return type annotation if present
    if sig.return_annotation != inspect.Signature.empty:
        if sig.return_annotation is not str:
            raise TypeError(
                f"Agent factory {func.__name__} must return str (agent_id), " f"got {sig.return_annotation}"
            )

    # mark as validated agent factory
    func._is_agent_factory = True

    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    return wrapper
