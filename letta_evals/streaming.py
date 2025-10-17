import json
from pathlib import Path
from typing import List, Optional, Tuple

import anyio

from letta_evals.models import Metrics, RunnerResult, SampleResult


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
        self.output_path.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        # truncate and write header
        header_obj = {"type": "header", "suite": self.suite_name, "config": self.config}
        with open(self.output_path / "header.json", "w", encoding="utf-8") as f:
            json.dump(header_obj, f, indent=2)

    async def append_result(self, result: SampleResult) -> None:
        result_obj = json.loads(result.model_dump_json())
        await self._write_line({"type": "result", "result": result_obj})

    async def write_metrics(self, metrics: Metrics, gates_passed: bool) -> None:
        metrics_obj = json.loads(metrics.model_dump_json())
        summary_obj = {"type": "summary", "metrics": metrics_obj, "gates_passed": gates_passed}
        with open(self.output_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_obj, f, indent=2)

    async def _write_line(self, obj: dict, truncate: bool = False) -> None:
        mode = "w" if truncate else "a"
        line = json.dumps(obj, ensure_ascii=False) + "\n"

        def _write() -> None:
            with open(self.output_path / "results.jsonl", mode, encoding="utf-8") as f:
                f.write(line)

        await anyio.to_thread.run_sync(_write)


class StreamingReader:
    """Reads a JSONL streaming results directory back into a RunnerResult."""

    @staticmethod
    async def to_runner_result(path: Path) -> RunnerResult:
        def _read() -> Tuple[Optional[str], Optional[dict], List[SampleResult], Optional[Metrics], bool]:
            _suite: Optional[str] = None
            _config: Optional[dict] = None
            _results: List[SampleResult] = []
            _metrics: Optional[Metrics] = None
            _gates: bool = False

            with open(path / "header.json", "r", encoding="utf-8") as f:
                header_obj = json.load(f)
                _suite = header_obj.get("suite")
                _config = header_obj.get("config")

            with open(path / "summary.json", "r", encoding="utf-8") as f:
                summary_obj = json.load(f)
                _metrics = Metrics(**summary_obj.get("metrics"))
                _gates = bool(summary_obj.get("gates_passed", False))

            with open(path / "results.jsonl", "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    _results.append(SampleResult(**rec["result"]))
            return _suite, _config, _results, _metrics, _gates

        suite, config, results, metrics, gates_passed = await anyio.to_thread.run_sync(_read)

        if suite is None or config is None:
            raise ValueError("Results JSONL missing header record")
        if metrics is None:
            raise ValueError("Results JSONL missing summary record")

        return RunnerResult(suite=suite, config=config, results=results, metrics=metrics, gates_passed=gates_passed)

    # No fallback metrics: summary is required in JSONL results.
