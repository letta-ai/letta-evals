"""Generate realistic bloated memory blocks using LLM expansion.

Uses an LLM to generate domain-specific, realistic accumulated knowledge
that simulates organic memory growth over time.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Optional

from agentfile import AgentFile

# Domain configurations for different types of projects
DOMAINS = {
    "ml_research": {
        "description": "ML research project with PyTorch, experiments, and paper writing",
        "seed_topics": [
            "experiment tracking and reproducibility",
            "GPU memory management and CUDA gotchas", 
            "hyperparameter tuning workflow",
            "paper writing conventions (NeurIPS, ICML)",
            "dataset preprocessing patterns",
            "model checkpointing and recovery",
        ],
    },
    "web_backend": {
        "description": "Python web backend with FastAPI, PostgreSQL, Redis",
        "seed_topics": [
            "API design patterns and versioning",
            "database migration workflow",
            "authentication and session handling",
            "caching strategies with Redis",
            "background job processing",
            "logging and monitoring setup",
        ],
    },
    "frontend_react": {
        "description": "React/TypeScript frontend with state management",
        "seed_topics": [
            "component structure and naming",
            "state management patterns (Redux vs Context)",
            "performance optimization (memo, useMemo)",
            "testing with React Testing Library",
            "styling approach (CSS modules vs styled-components)",
            "API integration and error handling",
        ],
    },
    "mobile_app": {
        "description": "React Native mobile app with offline support",
        "seed_topics": [
            "navigation patterns and deep linking",
            "offline-first data sync strategy",
            "push notification handling",
            "app store submission gotchas",
            "performance on low-end devices",
            "native module integration",
        ],
    },
    "devops_infra": {
        "description": "Infrastructure and DevOps with Kubernetes, Terraform",
        "seed_topics": [
            "Kubernetes deployment patterns",
            "Terraform state management",
            "CI/CD pipeline configuration",
            "secrets management approach",
            "monitoring and alerting setup",
            "disaster recovery procedures",
        ],
    },
    "data_pipeline": {
        "description": "Data engineering with Airflow, Spark, dbt",
        "seed_topics": [
            "DAG design and dependencies",
            "data quality checks and validation",
            "incremental vs full refresh patterns",
            "schema evolution handling",
            "backfill procedures",
            "cost optimization for cloud compute",
        ],
    },
}

EXPANSION_PROMPT = """You are simulating an AI coding assistant's memory that has accumulated organically over 6 months of working on a {domain_description}.

Generate a SINGLE memory block containing accumulated knowledge about the following topics. The content should feel like it was added incrementally over time - NOT a clean, organized document.

Topics to cover:
{topics}

Requirements:
1. Mix different formatting styles (some ## headers, some ### headers, some just bold text)
2. Include specific gotchas, warnings, and learned lessons
3. Add code snippets where relevant
4. Include some redundancy - similar points stated differently in different sections
5. Have inconsistent organization - related topics should NOT be grouped perfectly
6. Include specific tool versions, file paths, command examples
7. Add "CRITICAL" or "IMPORTANT" warnings for some items
8. Mix bullet styles (-, *, •)
9. Total length should be 4000-8000 characters

Output ONLY the memory block content, no preamble or explanation."""

USER_PREF_PROMPT = """Generate realistic user preferences that an AI coding assistant would learn over time. Include:

1. Communication style preferences (terse vs detailed, emoji usage, etc.)
2. Tool preferences (package managers, formatters, test runners)
3. Workflow shortcuts and conventions
4. Things the user gets annoyed by
5. Specific patterns the user likes/dislikes in code

Make it feel organically accumulated with:
- Inconsistent formatting
- Some redundancy
- Mix of ## and ### headers
- Specific examples and commands
- 2000-4000 characters total

Output ONLY the memory block content."""


async def expand_with_llm(
    prompt: str,
    model: str = "openai/gpt-4o-mini",
) -> str:
    """Use LLM to expand a prompt into realistic content."""
    from letta_client import AsyncLetta
    
    client = AsyncLetta(base_url="https://api.letta.com")
    
    # Create a temporary agent for generation
    agent = await client.agents.create(
        name="bloat-generator-temp",
        model=model,
        embedding="openai/text-embedding-3-small",
        include_base_tools=False,
    )
    
    try:
        response = await client.agents.messages.create(
            agent_id=agent.id,
            messages=[{"role": "user", "content": prompt}],
        )
        
        # Extract text from response
        for msg in response.messages:
            if hasattr(msg, 'content') and msg.content:
                return msg.content
        
        return ""
    finally:
        await client.agents.delete(agent_id=agent.id)


async def generate_domain_bloat(
    domain: str,
    seed: Optional[int] = None,
) -> tuple[str, str]:
    """Generate bloated memory for a specific domain.
    
    Returns:
        Tuple of (persona_content, project_content)
    """
    if seed is not None:
        random.seed(seed)
    
    config = DOMAINS[domain]
    
    # Shuffle and select topics
    topics = config["seed_topics"].copy()
    random.shuffle(topics)
    selected = topics[:random.randint(4, 6)]
    
    # Generate project knowledge
    project_prompt = EXPANSION_PROMPT.format(
        domain_description=config["description"],
        topics="\n".join(f"- {t}" for t in selected),
    )
    
    project_content = await expand_with_llm(project_prompt)
    
    # Generate user preferences
    persona_content = await expand_with_llm(USER_PREF_PROMPT)
    
    return persona_content, project_content


def create_llm_bloated_agent(
    name: str,
    persona_content: str,
    project_content: str,
) -> AgentFile:
    """Create agent with LLM-generated bloated blocks."""
    af = AgentFile.create_empty(name=name)
    
    af.data["blocks"] = [
        {
            "label": "persona",
            "value": persona_content,
            "description": "Accumulated preferences and behaviors",
            "limit": 20000,
            "id": "block-0",
        },
        {
            "label": "project", 
            "value": project_content,
            "description": "Project-specific knowledge",
            "limit": 20000,
            "id": "block-1",
        },
        {
            "label": "human",
            "value": "[TODO: Fill with user info]",
            "description": "Information about the user",
            "limit": 5000,
            "id": "block-2",
        },
    ]
    
    return af


async def main():
    """Generate LLM-expanded bloated agents."""
    output_dir = Path("defrag_test_cases/llm_generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = []
    idx = 0
    
    print("Generating LLM-expanded bloated agents...")
    print("(This may take a few minutes due to API calls)\n")
    
    for domain in DOMAINS:
        print(f"Domain: {domain}")
        
        # Generate 2 agents per domain
        for i in range(2):
            try:
                persona, project = await generate_domain_bloat(
                    domain=domain,
                    seed=42 + idx,
                )
                
                af = create_llm_bloated_agent(
                    name=f"llm-{domain}-{i:02d}",
                    persona_content=persona,
                    project_content=project,
                )
                
                path = output_dir / f"llm_{idx:02d}_{domain}.af"
                af.save(path)
                
                total_size = len(persona) + len(project)
                persona_topics = len([l for l in persona.split("\n") if l.startswith("## ") or l.startswith("### ")])
                project_topics = len([l for l in project.split("\n") if l.startswith("## ") or l.startswith("### ")])
                
                manifest.append({
                    "file": path.name,
                    "domain": domain,
                    "total_size": total_size,
                    "persona_size": len(persona),
                    "project_size": len(project),
                    "persona_topics": persona_topics,
                    "project_topics": project_topics,
                    "type": "llm_generated",
                })
                
                print(f"  {path.name}: {total_size} chars ({persona_topics}+{project_topics} topics)")
                idx += 1
                
            except Exception as e:
                print(f"  Error generating {domain}-{i}: {e}")
    
    # Save manifest
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nGenerated {len(manifest)} LLM-expanded agents in {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
