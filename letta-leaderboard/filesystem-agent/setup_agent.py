"""

Setup script to create agent with filesystem access for answering questions about structured data files.
"""

from letta_client import AsyncLetta

from letta_evals.decorators import agent_factory
from letta_evals.models import Sample


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
            tools = await client.tools.list(name=tool_name)
            if tools:
                required_tool_ids.append(tools[0].id)
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
        folders = await client.folders.list()

        folder_id = None
        for folder in folders:
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
