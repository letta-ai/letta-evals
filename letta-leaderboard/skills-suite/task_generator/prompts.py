"""
Prompt template for task generation.
"""

TASK_GENERATION_PROMPT = r"""You have access to Skills — folders of instructions, scripts, and resources that you can load dynamically to improve performance on specialized tasks. Skills teach you how to complete specific tasks in a repeatable way. Skills work through progressive disclosure — you should determine which Skills are relevant to complete a task and load them, helping to prevent context window overload.

Each Skill directory includes:
- `SKILL.md` file that starts with YAML frontmatter containing required metadata: name and description
- Additional files within the skill directory referenced by name from `SKILL.md`. These additional linked files should be navigated and discovered only as needed.

## Skill Information
SKILL.md for the `{{skill_name}}` skill:
<skill_md>
{{skill_content}}
</skill_md>

Additional files for the `{{skill_name}}` skill:
<files>
{{file_tree}}
</files>

## Previous Tasks
Previous tasks for the `{{skill_name}}` skill:
<previous_tasks>
{{previous_tasks}}
</previous_tasks>

## All Available Skills
All available skills and their descriptions:
<all_skills>
{{all_skills}}
</all_skills>

## Task Generation
Given the SKILL.md and additional files for the `{{skill_name}}` skill, previously generated tasks for this skill, and all available skills:
1. Generate a new and unique task that can be solved by an agent with access to only the given `{{skill_name}}` skill and its additional files.
    - The task should not mention the skill name or its description, but should also be difficult to complete without using the skill.
    - The task should require only the given skill and optionally its additional files to be loaded, but no additional skills.
    - The task should be unique and not similar to any of the previously generated tasks.
    - The task should be self-contained and not require any additional files or resources.
      - For skills that might require additional resources (like .docx files), mention the content of the files in the task instruction.
      - Do not reference additional resources because they are not available to the agent.
2. Provide two rubrics that can be directly used by an LLM to evaluate task completion and skill use.
    1. rubric_task_completion: for evaluating task completion given the task instruction.
        - The rubric should be a list of 4 clear, concise, and unambiguous criteria that must be met for the task to be considered complete.
        - Each criterion should be strictly related to the task and not the skill itself.
    2. rubric_skill_use: for evaluating the appropriate use of the `{{skill_name}}` skill.
        - The first criterion should only evaluate whether the right skill is selected, not whether it is loaded or used appropriately.
        - The second criterion should evaluate whether the skill is loaded with its SKILL.md file.
        - The third criterion should evaluate whether all additional files from the list of files are loaded or used.
        - The fourth criterion should evaluate whether the skill and its additional files are used appropriately.
3. Provide a list of files that should be loaded into the context window for the task.
    - The list of files should also mention the `{{skill_name}}` skill's SKILL.md file.

## Response Format
Return the `skill_name`, `task`, `rubric_task_completion`, `rubric_skill_use`, and `files` in a JSON object with the following structure:
```json
{
    "skill_name": str,
    "task": str,
    "rubric_task_completion": str,
    "rubric_skill_use": str,
    "files": list[str]
}
```"""
