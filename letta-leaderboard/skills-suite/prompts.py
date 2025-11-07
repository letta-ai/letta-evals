"""
Prompt templates for skills evaluation.
"""

SKILLS_TASK_PROMPT = r"""Complete the given task by:
- Creating artifacts in the provided sandbox path.
  - IMPORTANT: Artifacts should only be created in the sandbox path.
- Using any available skills that are relevant to the task.
  - IMPORTANT: Skills are read-only, DO NOT try to modify or update them. You DO NOT have permission to do so, even with sudo or chmod.
- Don't ask any clarifiying questions, make assumptions when necessary to complete the task.

**Sandbox Path:**
{pwd}/{{task_name}}

**Task:**
{{task}}

**Response Format:**
At the end of the task, return the absolute sandbox path where you created the artifacts, and if you used any skills and their additional files in JSON format with the following keys:
- 'sandbox_path': - sandbox path you used
- 'skills': - list of skills used
- 'skills_files': - list of skills' additional files used

IMPORTANT:
- Skills are read-only, DO NOT try to modify or update them. You DO NOT have permission to do so, even with sudo or chmod.
- Your final response should be only a valid JSON object wrapped in a code block with the above keys."""