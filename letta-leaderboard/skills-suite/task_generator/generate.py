#!/usr/bin/env python3
"""
Generate a task from a skill and related additional files.

This module uses the Claude API to generate tasks that require the use of a skill and related additional files.
"""

import argparse
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import anthropic
import pandas as pd
from prompts import TASK_GENERATION_PROMPT
from tqdm import tqdm
from utils import build_tree_structure, extract_skill_metadata

# skill -> weight: higher weight = more likely to be selected for task generation
AVAILABLE_SKILLS = {
    "algorithmic-art": 0.25,
    "artifacts-builder": 0.25,
    # "brand-guidelines": 0.0,
    "canvas-design": 1.0,
    "document-skills/docx": 0.5,
    "document-skills/pdf": 1.0,
    "document-skills/pptx": 0.25,
    "document-skills/xlsx": 0.75,
    "internal-comms": 1.0,
    "mcp-builder": 0.5,
    "skill-creator": 0.75,
    "slack-gif-creator": 1.0,
    # "template-skill": 0.0,
    "theme-factory": 0.5,
    "webapp-testing": 0.75,
}


def load_all_skills(skills_path: str) -> str:
    """
    Load all available skills from the skills directory.
    Extract skill metadata from SKILL.md file.

    Args:
        skills_path: Path to the skills directory
    Returns:
        Dictionary with skill directory names as keys and dictionaries containing
        'name' and 'description' extracted from YAML frontmatter as values.
    """
    skills_path = Path(skills_path)
    skills = []
    available_skills = sorted(list(AVAILABLE_SKILLS.keys()))
    for skill_name in available_skills:
        skill_path = skills_path / skill_name / "SKILL.md"
        skill_metadata = extract_skill_metadata(skill_path)
        if skill_metadata:
            skills.append(skill_metadata)

    # print(f"Loading {len(skills)} available skills")
    skills_str = "\n".join(skills)
    return skills_str


def load_skill_directory(skill_path: Path) -> Tuple[str, str]:
    """Load skill from SKILL.md file and list all files in the skill directory."""
    with open(skill_path / "SKILL.md", "r") as f:
        skill_content = f.read().strip()

    files = list(skill_path.glob("**/*"))
    files = [str(file.relative_to(skill_path.parent)) for file in files]
    file_tree = build_tree_structure(files)
    return skill_content, file_tree


def load_tasks_for_skill(
    skill_name: str,
    output_dir: str = "task_generator/output",
) -> str:
    """
    Load previously generated tasks for a given skill.

    Args:
        skill_name: Name of the skill to load tasks for
        output_dir: Directory containing the generated tasks

    Returns:
        String containing all tasks (sorted newest first), or "No tasks found" if no tasks are found
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        return "No tasks found"

    # Convert skill_name to the format used in filenames
    skill_prefix = skill_name.replace("/", "-")
    matching_files = sorted(output_path.glob(f"{skill_prefix}_*.json"), reverse=True)

    if not matching_files:
        return "No tasks found"

    tasks = ""
    for task_file in matching_files:
        with open(task_file, "r") as f:
            task_data = json.load(f)
            tasks += f"- Task: {task_data['task']}\n"
            tasks += f"  - Files: {', '.join(task_data['files'])}\n\n"

    return tasks


def create_task_generation_prompt(skill_name: str, skills_path: str) -> str:
    """Format the task generation prompt."""
    skills_path = Path(skills_path).expanduser()
    skill_path = skills_path / skill_name

    # print(f"Loading skill from {skill_path}")
    skill_content, file_tree = load_skill_directory(skill_path)

    # print(f"Loading previously generated tasks for {skill_name}...")
    previous_tasks = load_tasks_for_skill(skill_name)

    all_skills = load_all_skills(skills_path)

    return (
        TASK_GENERATION_PROMPT.replace("{{skill_name}}", skill_name)
        .replace("{{skill_content}}", skill_content)
        .replace("{{file_tree}}", file_tree)
        .replace("{{previous_tasks}}", previous_tasks)
        .replace("{{all_skills}}", all_skills)
    )


def call_claude_with_prompt(prompt: str) -> Dict[str, Any]:
    """Call Claude API with the task generation prompt and parse JSON response."""

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)

    response = client.beta.messages.create(
        model="claude-sonnet-4-5-20250929", max_tokens=8096, messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.content[0].text if response.content else "No text response"

    # Parse JSON from response - it may be wrapped in markdown code blocks
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]  # Remove ```json
    if response_text.startswith("```"):
        response_text = response_text[3:]  # Remove ```
    if response_text.endswith("```"):
        response_text = response_text[:-3]  # Remove trailing ```
    response_text = response_text.strip()

    try:
        parsed_response = json.loads(response_text)
        return parsed_response
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        print(f"Response text: {response_text}")
        raise


def generate_task(
    skill_name: str, skills_path: str = "~/terminal-bench-skills/.skills", output_dir: str = "task_generator/output"
) -> Path:
    """
    Generate a task from a skill and related additional files.

    Args:
        skill_name: Name of the skill to generate a task for
        skills_path: Path to the directory containing the skills
        output_dir: Directory to save the generated task

    Returns:
        Path to the generated task
    """
    output_dir = Path(output_dir)

    # print(f"Creating task generation prompt for {skill_name}...")
    generation_prompt = create_task_generation_prompt(skill_name, skills_path)

    # print("Calling Claude with task generation prompt...")
    response = call_claude_with_prompt(generation_prompt)

    # print(f"Successfully generated task for {skill_name}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_path = output_dir / f"{skill_name.replace('/', '-')}_{timestamp}.json"
    with open(task_path, "w") as f:
        json.dump(response, f, indent=2)

    return task_path


def generate_task_wrapper(task_num: int, total_tasks: int, skill_name: str) -> Tuple[str, Path]:
    """Wrapper function for parallel task generation."""
    task_path = generate_task(skill_name)
    return skill_name, task_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tasks from skills using the Claude API")
    parser.add_argument("--num-tasks", type=int, default=10, help="Number of tasks to generate")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for task generation")
    args = parser.parse_args()

    # Pre-select skills using weighted random sampling
    skills_to_generate = [
        random.choices(list(AVAILABLE_SKILLS.keys()), weights=list(AVAILABLE_SKILLS.values()), k=1)[0]
        for _ in range(args.num_tasks)
    ]

    # Generate multiple tasks in parallel with progress bar
    generated_tasks = defaultdict(list)

    print(f"Generating {args.num_tasks} tasks with {args.workers} workers...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(generate_task_wrapper, i + 1, args.num_tasks, skill_name): (i, skill_name)
            for i, skill_name in enumerate(skills_to_generate)
        }

        # Collect results as they complete with progress bar
        with tqdm(total=args.num_tasks, desc="Generating tasks", unit="task") as pbar:
            for future in as_completed(futures):
                try:
                    skill_name, task_path = future.result()
                    generated_tasks["skill_name"].append(skill_name)
                    generated_tasks["task_path"].append(task_path)
                    pbar.set_postfix_str(f"Latest: {skill_name}")
                    pbar.update(1)
                except Exception as e:
                    task_num, skill_name = futures[future]
                    tqdm.write(f"Error generating task {task_num + 1} for {skill_name}: {e}")
                    pbar.update(1)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Summary: Generated {len(generated_tasks['skill_name'])} task(s)")
    df = pd.DataFrame(generated_tasks)
    print(df["skill_name"].value_counts())
    print(f"{'=' * 60}")
