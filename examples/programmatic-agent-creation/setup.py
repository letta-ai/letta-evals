import sys
from pathlib import Path

from letta_client import AsyncLetta

from letta_evals.decorators import suite_setup

# add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from inventory_tool import TEST_TOOL_NAME, ManageInventoryTool


@suite_setup
async def prepare_evaluation(client: AsyncLetta) -> None:
    """Set up the evaluation environment by creating required tools."""
    tools_page = await client.tools.list(name=TEST_TOOL_NAME)
    if not tools_page.items:
        await client.tools.add(tool=ManageInventoryTool())
        print(f"Created {TEST_TOOL_NAME} tool")
    else:
        print(f"{TEST_TOOL_NAME} tool already exists")
