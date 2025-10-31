# Letta Code Bug Fixing Example

This example demonstrates evaluating Letta Code's ability to find and fix bugs in Python files automatically.

## What This Example Shows

- Using the `letta_code` target to test the Letta Code CLI agent
- Custom async grader that executes Python files and validates output
- Suite setup scripts that prepare the testing environment
- Using `extra_vars` in datasets to pass additional context to graders
- Testing autonomous debugging capabilities

## How It Works

### 1. Setup Phase (`setup.py:prepare_evaluation`)

Before each evaluation run, the setup script resets the sandbox:

```python
@suite_setup
async def prepare_evaluation() -> None:
    # removes existing sandbox directory
    # copies fresh buggy files from init_sandbox/
```

This ensures each test starts with the original buggy code.

**Note:** The `@suite_setup` decorator supports three signatures:
- `() -> None` - No parameters (shown above)
- `(client: AsyncLetta) -> None` - With client access
- `(client: AsyncLetta, model_name: str) -> None` - With client and model name (runs once per model when testing multiple models)

### 2. Evaluation Phase

For each sample, Letta Code:
1. Receives a prompt: "Find and fix the bug in sandbox/task_X.py"
2. Reads the file, identifies the bug
3. Fixes the bug using its Edit tool
4. Optionally runs the file to verify the fix

### 3. Grading Phase (`custom_python_grader.py`)

The custom async grader:
1. Retrieves the file path from `sample.extra_vars`
2. Executes the (hopefully fixed) Python file using `asyncio.create_subprocess_exec`
3. Captures stdout and compares it to the expected output
4. Returns score 1.0 if output matches, 0.0 otherwise

## The Buggy Files

### `task_1.py` - Syntax Error
```python
def calculate_sum(numbers)  # missing colon
```
**Expected output after fix:** `The sum is: 15`

### `task_2.py` - IndexError
```python
return arr[len(arr)]  # should be arr[-1] or arr[len(arr)-1]
```
**Expected output after fix:** `The last element is: 50`

### `task_3.py` - ZeroDivisionError
```python
def calculate_average(numbers):
    return sum(numbers) / len(numbers)  # crashes on empty list

calculate_average([])  # needs guard clause
```
**Expected output after fix:** `Average: 30.0` (only the second test case)

## Running This Example

### Prerequisites

1. **Install Letta Code CLI:**
```bash
npm install -g @letta-ai/letta-code
```

2. **Set up environment variables:**
```bash
export LETTA_API_KEY=your-api-key
export LETTA_PROJECT_ID=your-project-id
```

### Run the evaluation:
```bash
cd examples/letta-code-simple-edit
letta-evals run suite.yaml
```

## Configuration Details

### `suite.yaml`

```yaml
name: letta-code-bug-fix-test
dataset: dataset.jsonl
setup_script: setup.py:prepare_evaluation
target:
  kind: letta_code
  base_url: https://api.letta.com/
  working_dir: sandbox
  timeout: 300
  max_retries: 3
graders:
  bug_fix_check:
    kind: tool
    function: custom_python_grader.py:python_output_grader
    extractor: last_assistant
```

**Key fields:**
- `kind: letta_code` - Uses the Letta Code CLI instead of SDK
- `working_dir: sandbox` - Sets the working directory for letta CLI execution
- `timeout: 300` - 5 minute timeout per sample
- `max_retries: 3` - Retries failed CLI invocations
- `setup_script` - Runs before evaluation to reset the sandbox

### `dataset.jsonl`

```jsonl
{"input": "Find and fix the bug in sandbox/task_1.py", "ground_truth": "The sum is: 15", "extra_vars": {"file_path": "sandbox/task_1.py"}}
{"input": "Find and fix the bug in sandbox/task_2.py", "ground_truth": "The last element is: 50", "extra_vars": {"file_path": "sandbox/task_2.py"}}
{"input": "Find and fix the bug in sandbox/task_3.py", "ground_truth": "Average: 30.0", "extra_vars": {"file_path": "sandbox/task_3.py"}}
```

**Key points:**
- Generic prompts that don't specify the type of bug
- `extra_vars` passes file path to the custom grader
- `ground_truth` is the expected output when the file runs correctly

### `custom_python_grader.py`

```python
@grader
async def python_output_grader(sample: Sample, submission: str) -> GradeResult:
    # get file path from sample.extra_vars
    file_path = sample.extra_vars["file_path"]

    # run the python file asynchronously
    process = await asyncio.create_subprocess_exec(
        "python3", str(full_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)

    # compare output to ground truth
    if output == expected:
        return GradeResult(score=1.0, rationale=f"Output matches expected: {output}")
    else:
        return GradeResult(score=0.0, rationale=f"Output mismatch. Expected: '{expected}', Got: '{output}'")
```

**Key features:**
- Async grader using `asyncio.create_subprocess_exec`
- Reads file path from `sample.extra_vars`
- Executes the file and captures stdout/stderr
- Compares actual output to expected ground truth

## Key Takeaways

1. **`letta_code` target**: Tests the autonomous Letta Code CLI agent instead of SDK-based agents
2. **Custom async graders**: Can perform complex operations like running subprocesses
3. **Suite setup scripts**: Prepare the environment before evaluation (resetting files, creating directories, etc.)
4. **`extra_vars`**: Pass additional context to custom graders beyond input/ground_truth
5. **Working directory control**: The `working_dir` field sets where letta CLI executes

## Expected Results

All three samples should pass if Letta Code successfully:
1. Identifies the bug type (syntax, runtime, logic error)
2. Applies the correct fix
3. Produces code that runs without errors
4. Generates the expected output

The evaluation will show:
- Whether each file was fixed correctly
- The actual vs expected output for failures
- Overall pass rate (should be 100% for a successful run)
