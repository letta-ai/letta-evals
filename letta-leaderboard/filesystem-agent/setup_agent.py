"""
Setup script to create agent with filesystem access for answering questions about structured data files.
"""
import uuid
from pathlib import Path
from typing import List

from letta_client import AsyncLetta, CreateBlock
from letta_evals.models import Sample
from letta_evals.decorators import agent_factory


@agent_factory
async def setup_agent(client: AsyncLetta, sample: Sample) -> str:
    """
    Create an agent with access to filesystem data files.

    This function creates a fresh agent with a folder containing the required data files
    for the evaluation.

    Args:
        client: The AsyncLetta client
        sample: The sample containing the question and required files

    Returns:
        str: The ID of the created agent
    """

    # Get required files from sample
    required_files = []
    if sample.agent_args and 'extra' in sample.agent_args and sample.agent_args['extra']:
        required_files = sample.agent_args['extra'].get('required_files', [])

    if not required_files:
        print(f"⚠️ No required files specified for sample: {sample.input}")
        # Default to all common files if none specified
        required_files = ["people.txt", "vehicles.txt", "pets.txt", "bank_accounts.txt",
                         "credit_cards.txt", "addresses.txt", "employments.txt",
                         "internet_accounts.txt", "insurance_policies.txt", "medical_records.txt"]

    try:
        # Get an available embedding config
        embedding_configs = await client.embedding_models.list()
        if not embedding_configs:
            raise ValueError("No embedding configurations available")
        embedding_config = embedding_configs[0]

        # Create a unique folder for this agent's files
        folder_name = f"filesystem_data_{str(uuid.uuid4())[:8]}"
        folder = await client.folders.create(
            name=folder_name,
            embedding_config=embedding_config
        )

        print(f"✓ Created folder {folder.name} with id {folder.id}")

        # Upload required files to the folder
        datasets_dir = Path(__file__).parent / "datasets"
        uploaded_count = 0

        for filename in required_files:
            file_path = datasets_dir / filename
            if not file_path.exists():
                print(f"⚠️ File not found: {file_path}")
                continue

            try:
                # Upload file to folder
                print(f"  ✓ Uploading {filename} to folder")
                with open(file_path, 'rb') as f:
                    file_result = await client.folders.files.upload(
                        folder_id=folder.id,
                        file=f,
                    )

                uploaded_count += 1
                print(f"  ✓ Uploaded {filename} to folder")

            except Exception as e:
                print(f"  ✗ Error uploading {filename}: {e}")

        if uploaded_count == 0:
            raise ValueError("No files could be uploaded to the folder")

        print(f"✓ Successfully uploaded {uploaded_count} files to folder {folder.name}")

        # Create agent first without folder
        agent = await client.agents.create(
            name=f"Filesystem Data Agent {str(uuid.uuid4())[:8]}",
            memory_blocks=[
                CreateBlock(
                    label="persona",
                    value="You are an AI assistant that can access and analyze various data files to answer questions. "
                          "You have access to structured text files containing information about people, their vehicles, "
                          "pets, bank accounts, addresses, employment, and other personal data. "
                          "Use the file tools to open and read the relevant files needed to answer questions accurately. "
                          "When comparing data between entities, make sure to check all relevant files thoroughly. "
                          "The files available to you include: people.txt, vehicles.txt, pets.txt, bank_accounts.txt, "
                          "credit_cards.txt, addresses.txt, employments.txt, internet_accounts.txt, insurance_policies.txt, and medical_records.txt."
                ),
                CreateBlock(
                    label="human",
                    value="A user asking questions about the data in the files."
                )
            ],
            model="openai/gpt-4o-mini",
            embedding="openai/text-embedding-3-small",
            include_base_tools=True
        )

        print(f"✓ Created agent {agent.id}")

        # Attach the folder to the agent
        await client.agents.folders.attach(agent_id=agent.id, folder_id=folder.id)

        print(f"✓ Attached folder {folder.name} to agent {agent.id}")

        return agent.id

    except Exception as e:
        print(f"✗ Error creating agent: {e}")
        import traceback
        traceback.print_exc()
        raise