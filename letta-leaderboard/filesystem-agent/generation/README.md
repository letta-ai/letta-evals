# Filesystem Benchmark

The filesystem benchmark evaluates the ability of models to answer questions related to multiple files using filesystem tools: `grep` (search text) and `open_files` (read text files).

We use an AI agent (with `anthropic/claude-sonnet-4-20250514`) to generate difficult questions for this benchmark. The agent: 
1. Explores a [SQLite database](data/letta_file_bench.db)
2. Finds unique identifiers and relationships
3. Creates questions that require multiple file lookups
4. Verifies answers through SQL execution

## Generate New Questions

```bash
python3 -m question_generator.py \
    --num-questions 25 \
    --model claude-sonnet-4-20250514 \
    --db-path data/letta_file_bench.db \
    --output-dir data/generated_questions
```

### Parameters for Question Generation

- `--num-questions`: Number of questions to generate (default: 25)
- `--model`: Model to use for question generation
- `--db-path`: Path to the SQLite database file
- `--output-dir`: Directory to save generated questions