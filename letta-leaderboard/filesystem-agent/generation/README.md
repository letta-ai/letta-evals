# Filesystem Benchmark Generation

We use an AI agent (with `anthropic/claude-sonnet-4-20250514`) to generate difficult questions for this benchmark. The agent: 
1. Explores a [SQLite database](data/letta_file_bench.db)
2. Finds unique identifiers and relationships
3. Creates questions that require multiple file lookups
4. Verifies answers through SQL execution

## Question Types

The generator produces 8 question types, each with a dedicated prompt template in `prompts/`:

| Type | % of Eval | Files | Description |
|------|-----------|-------|-------------|
| `multi_hop_chain` | 20% | 3-4 | Follow references across files |
| `aggregation` | 15% | 3-5 | Sum balances, count records |
| `set_intersection` | 15% | 4-5 | Person matching X AND Y AND Z |
| `negation` | 10% | 3-4 | Who does NOT own X |
| `comparison_tiebreak` | 15% | 4-5 | Highest Y, if tied oldest |
| `multi_entity_comparison` | 10% | 4-5 | Between person A and B |
| `cross_file_counting` | 10% | 4-5 | Total financial products |
| `temporal_reasoning` | 5% | 3-4 | Date comparisons |

## Generate New Questions

```bash
# Generate 100 questions using the configured type distribution
python3 question_generator.py \
    --num-questions 100 \
    --model claude-opus-4-5-20251101 \
    --new-run

# Generate with parallel workers (one per question type)
python3 question_generator.py \
    --num-questions 100 \
    --parallel 15 \
    --new-run

# Generate only a specific question type
python3 question_generator.py \
    --num-questions 10 \
    --question-type aggregation \
    --new-run

# Append to the latest run
python3 question_generator.py \
    --num-questions 10 \
    --question-type negation
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-questions` | 10 | Number of questions to generate |
| `--model` | `claude-sonnet-4-20250514` | Model to use for generation |
| `--question-type` | None | Generate only this type (default: use distribution from config) |
| `--parallel` | 1 | Number of parallel workers (each handles one question type) |
| `--db-path` | `data/letta_file_bench.db` | Path to SQLite database |
| `--output-dir` | `data/generated_questions` | Output directory for generated questions |
| `--new-run` | False | Create a new timestamped run directory |
| `--append` | True | Append to the latest existing run |

## Guardrails

The `register_question` tool enforces quality checks before accepting a question:

- Minimum 3 files required
- Minimum 3 SQL queries in the reasoning chain
- Answer must be a concrete value (rejects "None", "does not own", etc.)
- `verification_query` must return exactly 1 row (proves answer uniqueness)
- `question_type` must be one of the 8 valid types
