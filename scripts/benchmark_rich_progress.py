#!/usr/bin/env python3
"""Synthetic benchmark for Rich progress rendering under event pressure.

Usage:
  python3 scripts/benchmark_rich_progress.py --samples 300 --workers 30 --duration 10
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from a source checkout without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    from letta_evals.visualization.benchmark import main

    main()
