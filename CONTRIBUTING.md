# Contributing to Letta Evals Kit

Thanks for your interest in contributing! We welcome issues, bug fixes, features, docs, and examples. By participating, you agree to abide by our Code of Conduct (see CODE_OF_CONDUCT.md).

## Development Setup

- Requirements:
  - Python 3.11+
  - Git, make (optional)
- Create and activate a virtualenv, then install deps:

```bash
# Recommended: using uv
pip install uv
uv sync --extra dev

# Alternatively with pip
pip install -e .[dev]

# Optional: set up pre-commit hooks
pre-commit install
```

## Linting & Formatting

```bash
ruff check .
ruff format .
```

- The project uses Ruff with a max line length of 120 and targets Python 3.11.
- Keep CLI output readable and prefer existing `rich` helpers.

## Testing

```bash
pytest        # run all tests
pytest -k expr -vv  # filter/verbose
```

- Async tests use `pytest-asyncio`.
- Focus on graders’ scoring logic, runner concurrency, and CLI options.

## Running the CLI Locally

```bash
# validate a suite
letta-evals validate examples/simple-rubric-grader/suite.yaml

# run a suite
letta-evals run examples/simple-rubric-grader/suite.yaml
```

## Commit Messages (Conventional Commits)

We use Conventional Commits to drive automated releases:

- feat: add something user-visible
- fix: bug fix
- docs:, chore:, refactor:, test:, perf:, build:, ci:
- BREAKING CHANGE: in the footer (or `!` after the type) for breaking changes

Examples:

- feat: add multi-model evaluation support
- fix(cli): correct exit code on failed gates
- refactor(runner): extract streaming writer
- feat!: change default concurrency to 15

Pre-1.0 releases: Breaking changes bump the MINOR version (configured via Release Please).

## Release Process (Automated)

We use Release Please to manage versions and CHANGELOG.

- Merge PRs with Conventional Commit messages
- Release Please opens/updates a single “release PR” (e.g., chore(main): release x.y.z)
- Review and merge the release PR
- On merge, a GitHub Release is created
- Our Publish workflow builds and publishes:
  - Normal releases → PyPI
  - Marked prereleases → TestPyPI

Manual runs: You can also run the “Publish” workflow from GitHub Actions and choose `pypi` or `testpypi` target.

## Pull Request Guidelines

- Keep PRs focused and small when possible
- Add/adjust tests for behavior changes
- Update README/docs and examples for user-facing changes
- Ensure `ruff check` and `pytest` pass (CI enforces this)
- Link related issues and include a clear description/screenshots for visualization changes

## Security

Please report security issues privately. See SECURITY.md for details.

## License

By contributing, you agree that your contributions will be licensed under the repository’s license.
