"""
Output directory management.

Provides a single `make_run_dir` implementation for the timestamped
output directories created by every RAPTOR workflow.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from core.config import RaptorConfig


def make_run_dir(prefix: str, suffix: Optional[str] = None) -> Path:
    """Create and return a timestamped output directory under RAPTOR_OUT_DIR.

    The directory name follows the convention used across all workflows::

        {prefix}_{suffix}_{timestamp}   # when suffix is given
        {prefix}_{timestamp}            # when suffix is omitted

    Args:
        prefix: Short identifier for the workflow (e.g. ``"scan"``, ``"raptor"``).
        suffix: Optional extra label such as a repository name.

    Returns:
        The created :class:`pathlib.Path`.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{prefix}_{suffix}_{timestamp}" if suffix else f"{prefix}_{timestamp}"
    out_dir = RaptorConfig.get_out_dir() / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
