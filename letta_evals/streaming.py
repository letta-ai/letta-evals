import json
from pathlib import Path
from typing import List, Optional, Tuple

import anyio

from letta_evals.models import Metrics, ModelMetrics, RunnerResult, SampleResult


class StreamingWriter:
    """Writes evaluation outputs incrementally to a JSONL file.

    Record schema per line:
      - header:  {"type": "header", "suite": str, "config": {...}}
      - result:  {"type": "result", "result": SampleResult}
      - summary: {"type": "summary", "metrics": Metrics, "gates_passed": bool}
    """

    def __init__(self, output_path: Path, suite_name: str, config: dict):
        self.output_path = Path(output_path)
        self.suite_name = suite_name
        self.config = config

    async def initialize(self) -> None:
        # truncate and write header
        await self._write_line({"type": "header", "suite": self.suite_name, "config": self.config}, truncate=True)

    async def append_result(self, result: SampleResult) -> None:
        result_obj = json.loads(result.model_dump_json())
        await self._write_line({"type": "result", "result": result_obj})

    async def write_metrics(self, metrics: Metrics, gates_passed: bool) -> None:
        metrics_obj = json.loads(metrics.model_dump_json())
        await self._write_line({"type": "summary", "metrics": metrics_obj, "gates_passed": gates_passed})

    async def _write_line(self, obj: dict, truncate: bool = False) -> None:
        mode = "w" if truncate else "a"
        line = json.dumps(obj, ensure_ascii=False) + "\n"

        def _write() -> None:
            with open(self.output_path, mode, encoding="utf-8") as f:
                f.write(line)

        await anyio.to_thread.run_sync(_write)


class StreamingReader:
    """Reads a JSONL streaming results file back into a RunnerResult."""

    @staticmethod
    async def to_runner_result(path: Path) -> RunnerResult:
        def _read() -> Tuple[Optional[str], Optional[dict], List[SampleResult], Optional[Metrics], bool]:
            _suite: Optional[str] = None
            _config: Optional[dict] = None
            _results: List[SampleResult] = []
            _metrics: Optional[Metrics] = None
            _gates: bool = False

            with open(path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    rec_type = rec.get("type")
                    if rec_type == "header":
                        _suite = rec.get("suite")
                        _config = rec.get("config")
                    elif rec_type == "result":
                        _results.append(SampleResult(**rec["result"]))
                    elif rec_type == "summary":
                        _metrics = Metrics(**rec["metrics"])
                        _gates = bool(rec.get("gates_passed", False))
            return _suite, _config, _results, _metrics, _gates

        suite, config, results, metrics, gates_passed = await anyio.to_thread.run_sync(_read)

        # derive defaults if header/summary are missing
        if suite is None:
            suite = "unknown"
        if config is None:
            config = {}
        if metrics is None:
            metrics = StreamingReader._calculate_metrics(results)

        return RunnerResult(suite=suite, config=config, results=results, metrics=metrics, gates_passed=gates_passed)

    @staticmethod
    def _calculate_metrics(results: List[SampleResult]) -> Metrics:
        total = len(results)
        if total == 0:
            return Metrics(total=0, avg_score=0.0)

        scores = [r.grade.score for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # per-model breakdown if model_name present
        model_buckets = {}
        for r in results:
            key = r.model_name or "default"
            model_buckets.setdefault(key, []).append(r)

        per_model_list: List[ModelMetrics] = []
        for model_name, bucket in model_buckets.items():
            model_scores = [r.grade.score for r in bucket]
            model_avg = sum(model_scores) / len(model_scores) if model_scores else 0.0
            # pass/fail counts can't be determined without gate spec; store zeros
            per_model_list.append(
                ModelMetrics(
                    model_name=model_name,
                    total=len(bucket),
                    avg_score=model_avg,
                    passed_samples=0,
                    failed_samples=0,
                )
            )

        return Metrics(total=total, avg_score=avg_score, per_model=per_model_list or None)
