import importlib.util
import sys
from pathlib import Path
from typing import Any


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
