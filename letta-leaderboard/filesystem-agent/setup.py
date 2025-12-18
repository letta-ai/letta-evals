"""
Setup script for filesystem evaluation environment.
"""

from pathlib import Path

from letta_client import AsyncLetta

from letta_evals.decorators import suite_setup


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
    dataset_dir = Path(__file__).parent / "datasets"

    if not dataset_dir.exists():
        raise RuntimeError(f"Dataset directory not found: {dataset_dir}")

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
        file_path = dataset_dir / filename
        if file_path.exists():
            with open(file_path, "rb") as f:
                await client.folders.files.upload(folder_id=folder.id, file=f)
            print(f"Uploaded {filename} to folder")
        else:
            print(f"Warning: {filename} not found in dataset directory")
