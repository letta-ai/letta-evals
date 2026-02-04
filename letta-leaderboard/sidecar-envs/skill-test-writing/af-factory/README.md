# af-factory

Generate test cases for memory defragmentation evaluation.

## Contents

- `agentfile.py` - Wrapper for .af file manipulation
- `generate_bloated_blocks.py` - Template-based synthetic bloat generation
- `generate_llm_bloat.py` - LLM-based domain-specific bloat generation
- `defrag_test_cases/` - Generated test cases (44 agents)
- `extracted_contexts/raw/` - 1000 exported agentfiles (source for mining)

## Test Cases

| Type | Count | Description |
|------|-------|-------------|
| Real | 12 | Extracted from actual agents with organic bloat |
| Synthetic | 25 | Template-based (9 topic categories) |
| LLM-generated | 7 | Domain-specific (ML, web, React, mobile) |
| **Total** | **44** | |

## Usage

```bash
# Generate synthetic bloated agents
uv run python generate_bloated_blocks.py

# Generate LLM-expanded agents (requires API)
uv run python generate_llm_bloat.py

# Inspect an agent
uv run python agentfile.py info defrag_test_cases/00_Big_Chungus_the_2nd.af
```

## Topic Categories (Synthetic)

1. git_workflow
2. coding_style
3. project_knowledge
4. user_preferences
5. framework_gotchas
6. debugging
7. security
8. performance
9. deployment

## Domains (LLM-generated)

1. ml_research - PyTorch, experiments, paper writing
2. web_backend - FastAPI, PostgreSQL, Redis
3. frontend_react - React/TypeScript, state management
4. mobile_app - React Native, offline support
5. devops_infra - Kubernetes, Terraform
6. data_pipeline - Airflow, Spark, dbt
