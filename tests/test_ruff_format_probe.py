"""Temporary CI probe: print the exact Ruff formatting diff, then remove this file."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PATHS = [
    "src/ai_pr_orchestrator/v3/foreman.py",
    "tests/integration/test_v3_dispositions.py",
    "tests/unit/test_v3_foreman.py",
    "tests/unit/test_v3_git_ops.py",
]


def test_print_exact_ruff_format_diff() -> None:
    ruff = str(Path(sys.executable).with_name("ruff"))
    subprocess.run([ruff, "format", *PATHS], check=True)
    diff = subprocess.check_output(["git", "diff", "--", *PATHS], text=True)
    raise AssertionError("RUFF_FORMAT_DIFF_BEGIN\n" + diff + "\nRUFF_FORMAT_DIFF_END")
