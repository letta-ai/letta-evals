import shutil
from pathlib import Path

from letta_client import AsyncLetta

from letta_evals.decorators import suite_setup


@suite_setup
async def prepare_evaluation(client: AsyncLetta, model_name: str) -> None:
    """Set up the evaluation environment by copying over the init_sandbox and cleaning up the existing sandbox."""
    script_dir = Path(__file__).parent
    init_sandbox = script_dir / "init_sandbox"

    model_name = model_name.split("/")[-1]
    sandbox = script_dir / "sandbox" / model_name

    if sandbox.exists():
        shutil.rmtree(sandbox)
        print(f"Removed existing sandbox directory: {sandbox}")

    shutil.copytree(init_sandbox, sandbox)
    print(f"Reset and copied {init_sandbox} to {sandbox}")
