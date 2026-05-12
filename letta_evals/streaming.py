"""Incremental writer + reader for evaluation results.

On-disk layout (always partitioned by model — there is always at least one
model identifier, defaulting to ``"default"`` when the suite does not pin
specific models):

    num_runs = 1:
        output/
        ├── suite.json
        ├── <model>.jsonl
        └── summary.json

    num_runs > 1:
        output/
        ├── suite.json
        ├── <model>/
        │   ├── run_1.jsonl
        │   ├── run_2.jsonl
        │   └── summary.json
        └── summary.json

Each per-sample line in a ``*.jsonl`` is just the serialized ``SampleResult``
JSON (no wrapper). ``suite.json`` and ``summary.json`` are single JSON files.
"""

import json
from pathlib import Path
from typing import Dict, List

import anyio

from letta_evals.models import (
    ModelRun,
    ModelSummary,
    RunnerResult,
    Sample,
    SampleResult,
    Summary,
    SuiteSpec,
)


class StreamingWriter:
    """Writes evaluation outputs incrementally to the on-disk layout.

    Parameters
    ----------
    output_path : Path
        Output directory root.
    suite_spec : SuiteSpec
        The suite configuration; written to ``suite.json`` on initialize.
    samples : List[Sample]
        The dataset, written to ``suite.json`` on initialize.
    models : List[str]
        Ordered list of model identifiers. Must be non-empty.
    num_runs : int
        Number of runs. Determines whether to use ``run_N.jsonl`` filenames
        and whether per-model summaries are written.
    """

    def __init__(
        self,
        output_path: Path,
        suite_spec: SuiteSpec,
        samples: List[Sample],
        models: List[str],
        num_runs: int = 1,
    ):
        if not models:
            raise ValueError("StreamingWriter requires at least one model id")
        self.output_path = Path(output_path)
        self.suite_spec = suite_spec
        self.samples = samples
        self.models = list(models)
        self.num_runs = max(1, int(num_runs))
        self.output_path.mkdir(parents=True, exist_ok=True)

    @property
    def multi_run(self) -> bool:
        return self.num_runs > 1

    # ── path resolution ──

    def results_path(self, model: str, run: int = 1) -> Path:
        """Return the per-(model, run) results.jsonl path.

        Creates parent directories as needed.
        """
        segment = _safe_segment(model)
        if self.multi_run:
            model_dir = self.output_path / segment
            model_dir.mkdir(parents=True, exist_ok=True)
            return model_dir / f"run_{run}.jsonl"
        return self.output_path / f"{segment}.jsonl"

    def model_summary_path(self, model: str) -> Path:
        """Return the per-model summary.json path (multi-run only)."""
        return self.output_path / _safe_segment(model) / "summary.json"

    def summary_path(self) -> Path:
        return self.output_path / "summary.json"

    def suite_path(self) -> Path:
        return self.output_path / "suite.json"

    # ── I/O ──

    async def initialize(self) -> None:
        """Write suite.json (config + dataset) once.

        Also truncates any existing per-(model, run) results.jsonl files so we
        start each run from a clean slate.
        """
        suite_obj = {
            "suite": self.suite_spec.name,
            "config": json.loads(self.suite_spec.model_dump_json(exclude={"base_dir"})),
            "samples": [json.loads(s.model_dump_json()) for s in self.samples],
        }

        def _write() -> None:
            with open(self.suite_path(), "w", encoding="utf-8") as f:
                json.dump(suite_obj, f, indent=2, default=str)
            # truncate per-(model, run) result files
            targets: List[Path] = []
            for model in self.models:
                if self.multi_run:
                    for run in range(1, self.num_runs + 1):
                        targets.append(self.results_path(model, run))
                else:
                    targets.append(self.results_path(model))
            for path in targets:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")

        await anyio.to_thread.run_sync(_write)

    async def append_result(
        self,
        result: SampleResult,
        model: str,
        run: int = 1,
    ) -> None:
        """Append a per-sample record to the appropriate jsonl file."""
        path = self.results_path(model=model, run=run)
        line = result.model_dump_json() + "\n"

        def _append() -> None:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)

        await anyio.to_thread.run_sync(_append)

    async def write_model_summary(self, summary: ModelSummary) -> None:
        """Write per-model summary.json (multi-run only). Lives in ``<model>/``."""
        path = self.model_summary_path(summary.model)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = summary.model_dump_json(indent=2, exclude_none=True)

        def _write() -> None:
            path.write_text(payload, encoding="utf-8")

        await anyio.to_thread.run_sync(_write)

    async def write_summary(self, summary: Summary) -> None:
        """Write top-level summary.json.

        ``runs`` is omitted from each ModelSummary in the top-level file (it
        only belongs in the per-model summary.json).
        """
        payload_obj = json.loads(summary.model_dump_json(exclude_none=True))
        for m in payload_obj.get("models", []):
            m.pop("runs", None)
        payload = json.dumps(payload_obj, indent=2)
        path = self.summary_path()

        def _write() -> None:
            path.write_text(payload, encoding="utf-8")

        await anyio.to_thread.run_sync(_write)


class StreamingReader:
    """Reads a streaming results directory back into a RunnerResult."""

    @staticmethod
    async def to_runner_result(path: Path) -> RunnerResult:
        """Reconstruct a RunnerResult from an on-disk output directory."""
        path = Path(path)

        def _read() -> RunnerResult:
            suite_path = path / "suite.json"
            summary_path = path / "summary.json"
            if not suite_path.exists():
                raise FileNotFoundError(f"suite.json missing in {path}")
            if not summary_path.exists():
                raise FileNotFoundError(f"summary.json missing in {path}")

            with open(suite_path, "r", encoding="utf-8") as f:
                suite_obj = json.load(f)
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_obj = json.load(f)

            suite_spec = SuiteSpec(**suite_obj["config"])
            samples = [Sample(**s) for s in suite_obj["samples"]]
            summary = Summary(**summary_obj)

            runs: Dict[str, ModelRun] = {}
            for model_summary in summary.models:
                segment = _safe_segment(model_summary.model)
                model_dir = path / segment
                model_jsonl = path / f"{segment}.jsonl"

                if model_dir.is_dir():
                    # Multi-run layout
                    per_run_results: List[List[SampleResult]] = [
                        _read_jsonl(rf) for rf in sorted(model_dir.glob("run_*.jsonl"))
                    ]
                    detailed_summary_path = model_dir / "summary.json"
                    if detailed_summary_path.exists():
                        with open(detailed_summary_path, "r", encoding="utf-8") as f:
                            detailed_summary = ModelSummary(**json.load(f))
                    else:
                        detailed_summary = model_summary
                    runs[model_summary.model] = ModelRun(
                        model=model_summary.model,
                        results=per_run_results[-1] if per_run_results else [],
                        runs=per_run_results if per_run_results else None,
                        summary=detailed_summary,
                    )
                elif model_jsonl.exists():
                    # Single-run layout
                    runs[model_summary.model] = ModelRun(
                        model=model_summary.model,
                        results=_read_jsonl(model_jsonl),
                        runs=None,
                        summary=model_summary,
                    )
                else:
                    raise FileNotFoundError(
                        f"No results file found for model {model_summary.model!r} in {path} "
                        f"(expected {model_jsonl} or {model_dir}/run_*.jsonl)"
                    )

            return RunnerResult(
                suite_spec=suite_spec,
                samples=samples,
                runs=runs,
                summary=summary,
                gates_passed=summary.gates_passed,
            )

        return await anyio.to_thread.run_sync(_read)


def _read_jsonl(path: Path) -> List[SampleResult]:
    out: List[SampleResult] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            out.append(SampleResult(**json.loads(line)))
    return out


def _safe_segment(name: str) -> str:
    """Sanitize a model identifier for use as a file/directory name.

    Replaces path separators and a few problematic characters with ``-``. Model
    names like ``openai/gpt-4`` become ``openai-gpt-4``.
    """
    return name.replace("/", "-").replace("\\", "-").replace(":", "-")
