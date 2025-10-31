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
    Supports both sync and async functions.
    Auto-registers the function to the grader registry.

    Usage:
        @grader
        def my_grader(sample: Sample, submission: str) -> GradeResult:
            ...

        @grader
        async def async_grader(sample: Sample, submission: str) -> GradeResult:
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
                f"Grader {f.__name__} must have parameters named 'sample' and 'submission', got {param_names}"
            )

        if sig.return_annotation != inspect.Signature.empty:
            if sig.return_annotation != GradeResult:
                raise TypeError(f"Grader {f.__name__} must return GradeResult, got {sig.return_annotation}")

        registry_name = name or f.__name__
        GRADER_REGISTRY[registry_name] = f

        f._is_grader = True

        if inspect.iscoroutinefunction(f):

            @wraps(f)
            async def wrapper(*args, **kwargs):
                return await f(*args, **kwargs)
        else:

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

    Validates that the function has signature:
      - (trajectory: List[List[LettaMessageUnion]], config: dict) -> str, OR
      - (trajectory: List[List[LettaMessageUnion]], config: dict, agent_state: Optional[AgentState]) -> str
    Auto-registers the function to the extractor registry.

    Usage:
        @extractor
        def my_extractor(trajectory: List[List[LettaMessageUnion]], config: dict) -> str:
            ...

        @extractor
        def memory_extractor(trajectory: List[List[LettaMessageUnion]], config: dict, agent_state: Optional[AgentState]) -> str:
            ...

        @extractor(name="custom_name")
        def another_extractor(trajectory: List[List[LettaMessageUnion]], config: dict) -> str:
            ...
    """

    def decorator(f: Callable) -> Callable:
        sig = inspect.signature(f)
        params = list(sig.parameters.values())

        if len(params) not in (2, 3):
            raise TypeError(
                f"Extractor {f.__name__} must have 2 or 3 parameters (trajectory, config) or (trajectory, config, agent_state), got {len(params)}"
            )

        param_names = [p.name for p in params]
        if len(params) == 2:
            if param_names != ["trajectory", "config"]:
                raise TypeError(
                    f"Extractor {f.__name__} must have parameters named 'trajectory' and 'config', got {param_names}"
                )
        elif len(params) == 3:
            if param_names != ["trajectory", "config", "agent_state"]:
                raise TypeError(
                    f"Extractor {f.__name__} must have parameters named 'trajectory', 'config', and 'agent_state', got {param_names}"
                )

        if sig.return_annotation != inspect.Signature.empty:
            if sig.return_annotation is not str:
                raise TypeError(f"Extractor {f.__name__} must return str, got {sig.return_annotation}")

        registry_name = name or f.__name__
        EXTRACTOR_REGISTRY[registry_name] = f

        f._is_extractor = True
        f._extractor_param_count = len(params)

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

    Validates that the function has signature: async (client: AsyncLetta, sample: Sample) -> str

    Usage:
        @agent_factory
        async def create_inventory_agent(client: AsyncLetta, sample: Sample) -> str:
            # create agent using client and sample data
            return agent_id
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if len(params) != 2:
        raise TypeError(
            f"Agent factory {func.__name__} must have exactly 2 parameters (client, sample), got {len(params)}"
        )

    param_names = [p.name for p in params]
    if param_names != ["client", "sample"]:
        raise TypeError(
            f"Agent factory {func.__name__} must have parameters named 'client' and 'sample', got {param_names}"
        )

    if not inspect.iscoroutinefunction(func):
        raise TypeError(f"Agent factory {func.__name__} must be an async function")

    if sig.return_annotation != inspect.Signature.empty:
        if sig.return_annotation is not str:
            raise TypeError(f"Agent factory {func.__name__} must return str (agent_id), got {sig.return_annotation}")

    # mark as validated agent factory
    func._is_agent_factory = True

    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    return wrapper


def suite_setup(func: Callable) -> Callable:
    """
    Decorator for suite setup functions.

    Supports three signatures:
    - async () -> None (no parameters)
    - async (client: AsyncLetta) -> None (with client parameter)
    - async (client: AsyncLetta, model_name: str) -> None (with client and model_name parameters)
    Also supports sync versions of all three.

    Usage:
        @suite_setup
        async def prepare_evaluation(client: AsyncLetta) -> None:
            # perform setup operations with client
            await client.tools.add(tool=MyCustomTool())

        @suite_setup
        async def prepare_evaluation_no_client() -> None:
            # perform setup operations without client
            pass

        @suite_setup
        async def prepare_evaluation_with_model(client: AsyncLetta, model_name: str) -> None:
            # perform setup operations with client and model_name
            print(f"Setting up for model: {model_name}")
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if len(params) not in (0, 1, 2):
        raise TypeError(
            f"Suite setup {func.__name__} must have 0, 1, or 2 parameters (client, model_name), got {len(params)}"
        )

    if len(params) == 1:
        param_names = [p.name for p in params]
        if param_names != ["client"]:
            raise TypeError(f"Suite setup {func.__name__} must have parameter named 'client', got {param_names}")
    elif len(params) == 2:
        param_names = [p.name for p in params]
        if param_names != ["client", "model_name"]:
            raise TypeError(
                f"Suite setup {func.__name__} must have parameters named 'client' and 'model_name', got {param_names}"
            )

    if sig.return_annotation != inspect.Signature.empty:
        if sig.return_annotation is not None and sig.return_annotation is not None:
            raise TypeError(f"Suite setup {func.__name__} must return None, got {sig.return_annotation}")

    # mark as validated suite setup and store param count
    func._is_suite_setup = True
    func._suite_setup_param_count = len(params)

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
    else:

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

    return wrapper
