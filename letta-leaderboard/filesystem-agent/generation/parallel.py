"""Parallel question generation with progress bars."""

import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

from display import Colors


class ParallelMixin:
    """Mixin providing parallel generation for QuestionGeneratorAgent.

    Expects the host class to have:
        - self.quiet: bool
        - self.config: Dict[str, Any]
        - self.model: str
        - self.output_path: Path
        - self.total_tokens: Dict[str, int]
        - self.question_type_distribution: Dict[str, float]
        - self._print_separator()
        - self.get_existing_questions()
        - self.generate_single_question()
    """

    def generate_questions_parallel(
        self, num_questions: int = 10, num_workers: int = 4, question_type: Optional[str] = None
    ):
        """Generate questions with parallel workers, one worker per question type.

        Each worker handles all questions for a single type sequentially,
        so it can see its own previous questions and avoid duplicates.
        Different types run in parallel since they won't collide.
        """
        # Build per-type counts
        type_counts = self._build_type_counts(num_questions, question_type)

        print(f"\n{Colors.HEADER}Starting PARALLEL question generation with {self.model}{Colors.ENDC}")
        print(f"Target: {num_questions} questions across {len(type_counts)} type workers")
        print(f"Output: {self.output_path}")
        for tname, tcount in type_counts.items():
            print(f"  {tname}: {tcount} questions")
        self._print_separator("=")

        # Suppress verbose output in parallel mode
        self.quiet = True

        # Thread-safe progress state
        progress_lock = threading.Lock()
        progress = {"success": 0, "failed": 0}
        type_progress = {
            qtype: {"done": 0, "ok": 0, "fail": 0, "total": count}
            for qtype, count in type_counts.items()
        }

        max_name_len = max(len(name) for name in type_counts)

        def _render_progress():
            """Render all progress bars to stdout."""
            lines = []
            for qtype in type_counts:
                tp = type_progress[qtype]
                total = tp["total"]
                ok = tp["ok"]
                fail = tp["fail"]
                done = tp["done"]

                bar_width = 20
                filled = int(bar_width * done / total) if total > 0 else 0
                bar = "█" * filled + "░" * (bar_width - filled)

                if done == 0:
                    color = Colors.DIM
                elif fail > 0:
                    color = Colors.YELLOW
                elif done == total:
                    color = Colors.GREEN
                else:
                    color = Colors.CYAN

                name_padded = qtype.ljust(max_name_len)
                fail_str = f" ({fail} failed)" if fail > 0 else ""
                lines.append(f"  {color}{name_padded}  {bar}  {ok}/{total}{fail_str}{Colors.ENDC}")

            total_done = progress["success"] + progress["failed"]
            lines.append(f"\n  Total: {total_done}/{num_questions} ({progress['success']} ok, {progress['failed']} failed)")

            output = "\n".join(lines)
            print(f"\033[{len(lines) + 1}A\033[J{output}", flush=True)

        # Print initial empty progress bars
        num_lines = len(type_counts) + 2
        print("\n" * num_lines, flush=True)
        _render_progress()

        def _generate_type_batch(qtype: str, count: int) -> Dict[str, int]:
            """Generate `count` questions of `qtype` sequentially."""
            batch_results = {"success": 0, "failed": 0}
            max_iterations = self.config.get("max_iterations_per_question", 20)

            for i in range(count):
                existing_questions = self.get_existing_questions()
                success, _ = self.generate_single_question(
                    i + 1, count, existing_questions,
                    max_iterations=max_iterations, question_type=qtype,
                )
                with progress_lock:
                    if success:
                        batch_results["success"] += 1
                        progress["success"] += 1
                        type_progress[qtype]["ok"] += 1
                    else:
                        batch_results["failed"] += 1
                        progress["failed"] += 1
                        type_progress[qtype]["fail"] += 1
                    type_progress[qtype]["done"] += 1
                    _render_progress()
            return batch_results

        # Launch one worker per type, capped at num_workers
        with ThreadPoolExecutor(max_workers=min(num_workers, len(type_counts))) as executor:
            futures = {}
            for qtype, count in type_counts.items():
                future = executor.submit(_generate_type_batch, qtype, count)
                futures[future] = qtype

            for future in as_completed(futures):
                qtype = futures[future]
                try:
                    future.result()
                except Exception as e:
                    with progress_lock:
                        remaining = type_counts[qtype] - type_progress[qtype]["done"]
                        progress["failed"] += remaining
                        type_progress[qtype]["fail"] += remaining
                        type_progress[qtype]["done"] = type_counts[qtype]
                        _render_progress()

        self.quiet = False

        self._print_separator("=")
        print(f"\n{Colors.HEADER}Parallel generation complete!{Colors.ENDC}")
        print(f"  Succeeded: {progress['success']}/{num_questions}")
        print(f"  Failed: {progress['failed']}/{num_questions}")
        print(
            f"  {Colors.DIM}Total tokens: {self.total_tokens['input']:,} in, {self.total_tokens['output']:,} out{Colors.ENDC}"
        )
        self._print_separator("=")

    def _build_type_counts(self, num_questions: int, question_type: Optional[str] = None) -> Dict[str, int]:
        """Build per-type question counts from distribution config."""
        if question_type:
            return {question_type: num_questions}

        type_counts = {}
        for type_name, pct in self.question_type_distribution.items():
            count = round(num_questions * pct)
            if count > 0:
                type_counts[type_name] = count

        # Adjust to hit exact total
        total = sum(type_counts.values())
        if total < num_questions:
            most_common = max(self.question_type_distribution, key=self.question_type_distribution.get)
            type_counts[most_common] = type_counts.get(most_common, 0) + (num_questions - total)
        elif total > num_questions:
            most_common = max(type_counts, key=type_counts.get)
            type_counts[most_common] -= (total - num_questions)

        return type_counts
