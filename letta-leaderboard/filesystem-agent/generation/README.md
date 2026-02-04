# Filesystem Benchmark Generation

We use an AI agent to generate difficult questions that test an agent's ability to **chain file operations**, **trace entity relationships**, and **manage multi-step information retrieval**.

The agent:
1. Explores a [SQLite database](data/letta_file_bench.db)
2. Finds unique identifiers and relationships
3. Creates questions that require multiple file lookups
4. Verifies answers through SQL execution

## Design Principle: TRUE Sequential Dependencies

Questions must have **sequential dependencies** — step N's output is step N+1's query input. The agent CANNOT parallelize queries.

**Good (sequential):**
```
"What pet does the coworker of the owner of plate 'XYZ-123' own?"
- Step 1: vehicles.txt → find owner of plate → pers-042
- Step 2: employments.txt → find employer of pers-042 → "Acme Corp"
- Step 3: employments.txt → find OTHER employee at Acme Corp → pers-087
- Step 4: pets.txt → find pers-087's pet
```
You cannot write step 3 until step 2 completes. "Coworker" is an **indirect relationship**.

**Bad (parallelizable):**
```
"Who has a Mastercard, owns a dog, and lives in Texas?"
- All 3 conditions can be grepped independently and intersected
```

## Question Types

The generator produces 8 question types, each with a dedicated prompt template in `prompts/`:

| Type | % of Eval | Files | Description | Opus Accuracy |
|------|-----------|-------|-------------|---------------|
| `multi_entity_comparison` | **30%** | 5-6 | Two parallel chains that converge for comparison | ~50% |
| `multi_hop_chain` | **25%** | 4-5 | Two parallel chains with comparison | ~40% |
| `aggregation` | 10% | 4-5 | Target found through chain, then aggregate | ~100% |
| `set_intersection` | 10% | 4-5 | Set defined by chain (same city as X), not parallel greps | ~100% |
| `comparison_tiebreak` | 10% | 4-5 | Group defined by chain, then compare with tiebreaker | ~100% |
| `negation` | 5% | 4-5 | Group defined by chain, then check absence | ~100% |
| `cross_file_counting` | 5% | 4-5 | Target found through chain, then count across files | ~100% |
| `temporal_reasoning` | 5% | 4-5 | Dates to compare found through chain | ~100% |

**~55% of questions use the two-parallel-chains pattern** (`multi_entity_comparison` + `multi_hop_chain`), which is the only pattern that consistently challenges strong models.

All types require:
- 4-5 files with true dependencies
- At least one **indirect relationship** (coworker, same city, same bank)
- Sequential steps where step N+1 cannot be written until step N completes

## Generate New Questions

```bash
# Generate 100 questions using the configured type distribution
python3 question_generator.py \
    --num-questions 100 \
    --new-run

# Generate with parallel workers (one per question type)
python3 question_generator.py \
    --num-questions 20 \
    --parallel 8 \
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
| `--model` | from config | Model to use (default: claude-opus-4-5-20251101) |
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
- Answer must be short (<100 chars) and not contain question text
- `verification_query` is **required** and must:
  - Compute the answer **end-to-end** using nested subqueries
  - NOT contain hardcoded person IDs (`pers-XXXX`)
  - NOT contain CASE statements with hardcoded answers
  - Return exactly 1 row that matches the provided answer
- `question_type` must be one of the 8 valid types

### Retry Logic

If a question fails to generate (API error, validation failure), the generator retries up to `max_retries_per_question` times (default: 3) before moving on.

## Validate Generated Questions

After generation, **always validate before testing**:

```bash
python validate_questions.py data/generated_questions/run_XXXX/agent_generated_questions.jsonl
```

The validation script checks:
- No forbidden terms (SSN, neighbor)
- Answer quality (length, no question text, no negatives)
- Verification query quality (no hardcoded IDs, end-to-end)
- GT correctness (runs verification query against DB)

**Exit codes:**
- `0` = Clean, ready for testing
- `1` = Issues found, review before testing

## Target Difficulty

Questions should achieve ~60-70% accuracy among top models. The key test:
- **Can all conditions be grepped in parallel?** → Too easy
- **Does step N+1 require step N's output?** → Good
- **Does it require deriving an indirect relationship?** → Good
- **Does it require completing TWO independent chains and comparing?** → Best
