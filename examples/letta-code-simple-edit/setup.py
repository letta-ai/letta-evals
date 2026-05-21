import shutil
from pathlib import Path

from letta_evals.decorators import suite_setup


@suite_setup
async def prepare_evaluation() -> None:
    """Stage pristine buggy task files into ``sandbox/`` before the run.

    The buggy files in ``init_sandbox/`` are copied to ``sandbox/`` (relative
    to this suite directory). Each sample then edits its own copy: when the
    suite runs in a Modal sandbox the whole suite directory is uploaded to
    /mnt/suite per sample, so every sample starts from this pristine copy and
    the agent's edits stay isolated to its container. Run in-process, the
    agent edits sandbox/ directly — re-running setup resets it.
    """
    script_dir = Path(__file__).parent
    init_sandbox = script_dir / "init_sandbox"
    sandbox = script_dir / "sandbox"

    if sandbox.exists():
        shutil.rmtree(sandbox)
        print(f"Removed existing sandbox directory: {sandbox}")

    shutil.copytree(init_sandbox, sandbox)
    print(f"Reset and copied {init_sandbox} to {sandbox}")
