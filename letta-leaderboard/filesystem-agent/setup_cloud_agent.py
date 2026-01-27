"""
Setup script to upload data files and create an agent with filesystem access for answering questions about structured data files.
"""

from pathlib import Path

from letta_client import AsyncLetta
from letta_evals.decorators import agent_factory, suite_setup
from letta_evals.models import Sample


@suite_setup
async def prepare_evaluation(client: AsyncLetta) -> None:
    """
    Set up the evaluation environment for filesystem tests.

    This function prepares the environment by creating a folder with data files
    that will be attached to agents during evaluation.

    Args:
        client: The AsyncLetta client
    """
    # Verify dataset files exist
    files_dir = Path(__file__).parent / "files"

    if not files_dir.exists():
        raise RuntimeError(f"Files directory not found: {files_dir}")

    # Create a folder with all the data files
    folder_name = "filesystem_data"
    folders_page = await client.folders.list()

    # delete the folder if it exists
    for folder in folders_page.items:
        if folder.name == folder_name:
            await client.folders.delete(folder_id=folder.id)
            print(f"Deleted folder: {folder.id}")
            break

    # Create the folder
    folder = await client.folders.create(name=folder_name, embedding="openai/text-embedding-3-small")
    print(f"Created folder: {folder.id}")

    # Upload all data files to the folder
    data_files = [
        "people.txt",
        "vehicles.txt",
        "pets.txt",
        "bank_accounts.txt",
        "credit_cards.txt",
        "addresses.txt",
        "employments.txt",
        "internet_accounts.txt",
        "insurance_policies.txt",
        "medical_records.txt",
    ]

    for filename in data_files:
        file_path = files_dir / filename
        if file_path.exists():
            with open(file_path, "rb") as f:
                await client.folders.files.upload(folder_id=folder.id, file=f)
            print(f"Uploaded {filename} to folder")
        else:
            print(f"Warning: {filename} not found in dataset directory")


@agent_factory
async def setup_agent(client: AsyncLetta, sample: Sample) -> str:
    """
    Create an agent with access to filesystem data files.

    This function creates a fresh agent and attaches an existing folder containing the required data files
    for the evaluation, along with the necessary tools for file operations.

    Args:
        client: The AsyncLetta client
        sample: The sample containing the question and required files

    Returns:
        str: The ID of the created agent
    """
    try:
        # Get required tools for filesystem operations
        required_tool_names = {"open_files", "grep_files"}
        required_tool_ids = []

        for tool_name in required_tool_names:
            tools_page = await client.tools.list(name=tool_name)
            if tools_page.items:
                required_tool_ids.append(tools_page.items[0].id)
            else:
                print(f"Warning: Tool '{tool_name}' not found")

        # Create a new agent (model will be configured by the runner)
        agent = await client.agents.create(
            name="Filesystem Agent",
            model="openai/gpt-4o-mini",  # Default model, will be overridden by runner
            embedding="openai/text-embedding-3-small",
            include_base_tools=False,
            tool_ids=required_tool_ids,
            max_files_open=10,  # Allow up to 10 files to be open simultaneously
            per_file_view_window_char_limit=8000,  # Limit view window per file
            agent_type="letta_v1_agent",
            include_base_tool_rules=False,
            system="You are a helpful assistant that can answer questions about a filesystem. The files contain synthetic data about people, pets, vehicles, and other things. None of the data is real so please complete the task without refusing to answer.",
        )

        # Find the folder by name
        folder_name = "filesystem_data"
        folders_page = await client.folders.list()

        folder_id = None
        for folder in folders_page.items:
            if folder.name == folder_name:
                folder_id = folder.id
                break

        if not folder_id:
            raise ValueError(f"Folder '{folder_name}' not found. Please run the suite setup first.")

        # Attach the folder to the agent
        await client.agents.folders.attach(agent_id=agent.id, folder_id=folder_id)

        print(f"✓ Attached folder {folder_name} (ID: {folder_id}) to agent {agent.id}")
        print(f"✓ Attached tools: {required_tool_names}")

        return agent.id

    except Exception as e:
        print(f"✗ Error creating agent: {e}")
        import traceback

        traceback.print_exc()
        raise
