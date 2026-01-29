"""Setup script for skill-test-writing evaluation.

Resets the output sandbox directory per model to ensure clean state.
"""

import shutil
from pathlib import Path

from letta_evals.decorators import suite_setup


@suite_setup
async def prepare_evaluation(client, model_name: str) -> None:
    """Reset the output sandbox directory for each model.
    
    Called once per model before evaluation begins.
    """
    script_dir = Path(__file__).parent
    
    # Extract model name (e.g., "anthropic/claude-sonnet-4-5" -> "claude-sonnet-4-5")
    model_slug = model_name.split("/")[-1]
    
    # Output directory for this model's sandbox
    output_dir = script_dir / "output" / model_slug
    
    # Clean up existing output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Removed existing output directory: {output_dir}")
    
    # Create fresh output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created clean output directory: {output_dir}")
