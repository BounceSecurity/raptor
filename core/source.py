"""
Source code context reader.

Provides a single implementation of "read N lines around line M" used by
multiple packages (codeql, llm_analysis).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union


def read_code_context(
    file_path: Union[str, Path],
    line: int,
    context_lines: int = 5,
) -> str:
    """Return source lines around *line* with a highlight marker on the target line.

    Args:
        file_path: Path to the source file (already resolved by the caller).
        line: 1-indexed line number to highlight.
        context_lines: Number of lines before and after *line* to include.

    Returns:
        Formatted snippet, e.g.::

                1 | import os
            >>> 2 | x = input()
                3 | print(x)

        Returns an empty string when the file cannot be read.
    """
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        start = max(0, line - context_lines - 1)
        end = min(len(lines), line + context_lines)

        context = []
        for i in range(start, end):
            marker = ">>>" if i == line - 1 else "   "
            context.append(f"{marker} {i + 1:4d} | {lines[i].rstrip()}")

        return "\n".join(context)
    except Exception:
        return ""
