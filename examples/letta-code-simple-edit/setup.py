import shutil
from pathlib import Path

from letta_evals.decorators import suite_setup


@suite_setup
async def prepare_evaluation() -> None:
    """Set up the evaluation environment by copying over the init_sandbox and cleaning up the existing sandbox."""
    script_dir = Path(__file__).parent
    init_sandbox = script_dir / "init_sandbox"
    sandbox = script_dir / "sandbox"

    if sandbox.exists():
        shutil.rmtree(sandbox)
        print(f"Removed existing sandbox directory: {sandbox}")

    shutil.copytree(init_sandbox, sandbox)
    print(f"Copied {init_sandbox} to {sandbox}")
    print("Sandbox directory reset with buggy files")
