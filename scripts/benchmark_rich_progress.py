#!/usr/bin/env python3
"""Synthetic benchmark for Rich progress rendering under event pressure.

Usage:
  python3 scripts/benchmark_rich_progress.py --samples 300 --workers 30 --duration 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

import anyio
from rich.console import Console

# Allow running directly from a source checkout without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from letta_evals.visualization.rich_progress import DisplayMode, EvalProgress  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark rich progress render behavior")
    parser.add_argument("--samples", type=int, default=200, help="Number of synthetic samples")
    parser.add_argument("--workers", type=int, default=20, help="Concurrent event workers")
    parser.add_argument("--duration", type=float, default=10.0, help="Benchmark duration in seconds")
    parser.add_argument("--refresh-fps", type=float, default=4.0, help="UI render FPS cap")
    parser.add_argument("--models", type=int, default=2, help="Number of synthetic model names")
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="Render sample table (off by default for very high-throughput runs)",
    )
    return parser.parse_args()


async def _event_worker(
    progress: EvalProgress,
    total_samples: int,
    model_names: List[str],
    worker_id: int,
) -> None:
    rng = random.Random(worker_id)

    while True:
        sample_id = rng.randrange(total_samples)
        model_name = model_names[worker_id % len(model_names)]
        agent_id = f"agent-{sample_id % 10}"

        roll = rng.random()
        if roll < 0.10:
            await progress.sample_started(sample_id, agent_id=agent_id, model_name=model_name)
        elif roll < 0.45:
            total_messages = rng.randint(1, 8)
            message_num = rng.randint(1, total_messages)
            await progress.message_sending(
                sample_id,
                message_num,
                total_messages,
                agent_id=agent_id,
                model_name=model_name,
            )
        elif roll < 0.75:
            total_turns = rng.randint(2, 12)
            turn_num = rng.randrange(total_turns)
            await progress.turn_graded(
                sample_id=sample_id,
                turn_num=turn_num,
                total_turns=total_turns,
                turn_score=rng.random(),
                grader_key="quality",
                agent_id=agent_id,
                model_name=model_name,
            )
        else:
            await progress.grading_started(sample_id, agent_id=agent_id, model_name=model_name)

        # Yield frequently to maximize interleaving pressure.
        await anyio.sleep(0)


async def run_benchmark(args: argparse.Namespace) -> None:
    model_names = [f"model-{idx + 1}" for idx in range(max(1, args.models))]
    total_evaluations = args.samples * len(model_names)
    console = Console()
    progress = EvalProgress(
        suite_name="rich-progress-benchmark",
        total_samples=total_evaluations,
        target_kind="benchmark",
        grader_kind="synthetic",
        max_concurrent=args.workers,
        display_mode=DisplayMode.STANDARD,
        console=console,
        update_freq=args.refresh_fps,
        show_samples=args.show_samples,
        cached_mode=False,
        metric_labels={"quality": "Quality"},
    )

    await progress.start()
    try:
        async with anyio.create_task_group() as tg:
            for worker_id in range(args.workers):
                tg.start_soon(
                    _event_worker,
                    progress,
                    args.samples,
                    model_names,
                    worker_id,
                )
            await anyio.sleep(args.duration)
            tg.cancel_scope.cancel()

        # Mark each synthetic sample/model as completed once so counters remain bounded.
        for sample_id in range(args.samples):
            for model_name in model_names:
                await progress.sample_completed(
                    sample_id=sample_id,
                    agent_id=f"agent-{sample_id % 10}",
                    score=0.5,
                    model_name=model_name,
                    metric_scores={"quality": 0.5},
                    rationale="benchmark",
                    metric_rationales={"quality": "benchmark"},
                )
    finally:
        progress.stop()

    snapshot = progress.get_perf_snapshot()
    print(json.dumps(snapshot, indent=2, sort_keys=True))


if __name__ == "__main__":
    anyio.run(run_benchmark, parse_args())
