#!/usr/bin/env python3
"""Evaluate reference tests using letta-evals AgentJudgeGrader (uses Letta API)."""

import asyncio
import json
from pathlib import Path

from letta_client import AsyncLetta

from letta_evals.graders.agent_judge import AgentJudgeGrader
from letta_evals.models import Sample

JUDGE_PROMPT_PATH = Path(__file__).parent / "judge_prompt.txt"
REFERENCE_TESTS_DIR = Path(__file__).parent / "reference_tests"
REFERENCE_SKILLS_DIR = Path(__file__).parent / "reference_skills"

JUDGE_TOOL_SOURCE = '''
def submit_grade(score: float, rationale: str) -> str:
    """Submit a grade for the evaluation.
    
    Args:
        score: A float between 0.0 and 1.0 indicating the quality score
        rationale: A string explaining the reasoning for the score
        
    Returns:
        Confirmation message
    """
    return f"Grade submitted: {score}"
'''


def load_judge_prompt() -> str:
    with open(JUDGE_PROMPT_PATH) as f:
        return f.read()


def get_skill_context(skill_name: str) -> dict:
    """Get skill context for the judge prompt."""
    skill_dir = REFERENCE_SKILLS_DIR / skill_name
    if not skill_dir.exists():
        return {"skill_name": skill_name, "skill_dir": str(skill_dir), "skill_file_tree": "Not found"}

    files = []
    for p in skill_dir.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            files.append(str(p.relative_to(skill_dir)))

    return {
        "skill_name": skill_name,
        "skill_dir": str(skill_dir),
        "skill_file_tree": "\n".join(sorted(files)[:20]),
    }


def format_test_as_submission(test_path: Path) -> str:
    """Format a test.yaml as a submission for the judge."""
    with open(test_path) as f:
        content = f.read()

    grader_path = test_path.parent / "grader.py"
    grader_content = ""
    if grader_path.exists():
        with open(grader_path) as f:
            grader_content = f.read()

    submission = f"## Test Files Found\n\n### test.yaml\n```\n{content}\n```\n"
    if grader_content:
        submission += f"\n### grader.py\n```python\n{grader_content}\n```\n"

    return submission


def submission_extractor(trajectory, agent_state=None):
    """Extract submission from trajectory (returns the pre-formatted string)."""
    if trajectory and trajectory[0]:
        return trajectory[0][0]
    return ""


async def create_judge_agent(client: AsyncLetta) -> str:
    """Create a fresh judge agent with submit_grade tool."""
    # Create the tool first
    tool = await client.tools.create(
        source_code=JUDGE_TOOL_SOURCE,
    )

    # Create the agent
    agent = await client.agents.create(
        name="eval-reference-tests-judge",
        description="Judge agent for evaluating reference tests",
        system="You are an evaluation judge. Evaluate submissions according to the rubric and call submit_grade with your score and rationale.",
        model="anthropic/claude-sonnet-4-5-20250929",
        embedding="letta/letta-free",
        tool_ids=[tool.id],
        include_base_tools=False,
    )
    return agent.id, tool.id


async def evaluate_single_test(
    grader: AgentJudgeGrader,
    test_path: Path,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Evaluate a single test with rate limiting."""
    rel_path = test_path.relative_to(REFERENCE_TESTS_DIR)
    skill_name = rel_path.parts[0]
    test_name = "/".join(rel_path.parts[1:-1])

    async with semaphore:
        submission = format_test_as_submission(test_path)
        skill_context = get_skill_context(skill_name)

        # Create sample with rubric_vars
        sample = Sample(
            id=0,
            input="",
            rubric_vars=skill_context,
        )

        # Create fake trajectory with our submission
        trajectory = [[submission]]

        try:
            result, _ = await grader.grade(sample, trajectory)
            score = result.score
            print(f"✓ {skill_name}/{test_name}: {score}")

            return {
                "skill": skill_name,
                "test": test_name,
                "score": score,
                "rationale": result.rationale[:200] if result.rationale else "",
            }
        except Exception as e:
            print(f"✗ {skill_name}/{test_name}: {e}")
            return {
                "skill": skill_name,
                "test": test_name,
                "score": "ERROR",
                "error": str(e),
            }


async def main():
    client = AsyncLetta(base_url="https://api.letta.com")
    judge_prompt = load_judge_prompt()

    # Create fresh judge agent
    print("Creating judge agent...")
    agent_id, tool_id = await create_judge_agent(client)
    print(f"Judge agent created: {agent_id}\n")

    try:
        # Create grader using agent_id
        grader = AgentJudgeGrader(
            prompt=judge_prompt,
            client=client,
            agent_id=agent_id,
            judge_tool_name="submit_grade",
            extractor="last_assistant",
            rubric_vars=["skill_name", "skill_dir", "skill_file_tree"],
        )

        # Override extractor to return submission directly
        grader.extractor = submission_extractor

        # Find all reference tests
        test_files = list(REFERENCE_TESTS_DIR.rglob("test.yaml"))
        print(f"Found {len(test_files)} reference tests, running in parallel...\n")

        # Limit concurrency
        semaphore = asyncio.Semaphore(10)

        tasks = [evaluate_single_test(grader, test_path, semaphore) for test_path in sorted(test_files)]
        results = await asyncio.gather(*tasks)

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        by_skill = {}
        for r in results:
            skill = r["skill"]
            if skill not in by_skill:
                by_skill[skill] = []
            by_skill[skill].append(r)

        for skill, tests in sorted(by_skill.items()):
            scores = [t["score"] for t in tests if isinstance(t.get("score"), (int, float))]
            avg = sum(scores) / len(scores) if scores else 0
            print(f"\n{skill}: avg={avg:.2f} ({len(scores)} tests)")
            for t in sorted(
                tests, key=lambda x: x.get("score", 0) if isinstance(x.get("score"), (int, float)) else -1, reverse=True
            ):
                score = t.get("score", "N/A")
                if isinstance(score, float):
                    score = f"{score:.2f}"
                print(f"  {score:>5}  {t['test']}")

        # Save results
        output_path = Path(__file__).parent / "reference_tests_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    finally:
        # Clean up
        print(f"\nCleaning up judge agent {agent_id}...")
        await client.agents.delete(agent_id=agent_id)
        await client.tools.delete(tool_id=tool_id)
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
