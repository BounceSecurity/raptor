"""Safe JSON file loading.

Consolidates the repeated try/except (json.JSONDecodeError, OSError) pattern
that appeared in orchestrator.py, bootstrap.py (×3), and reporting.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_json_file(
    path: "str | Path",
    *,
    logger: Any = None,
) -> Optional[Dict[str, Any]]:
    """Read *path* and return its contents parsed as a JSON object.

    Returns ``None`` on any read or parse error.  The caller is responsible
    for deciding what to do with a ``None`` result (skip, return early, etc.).

    Args:
        path: File path to read.
        logger: Optional logger; when provided, a ``warning`` is emitted on
            failure including the exception message.

    Returns:
        Parsed dict, or ``None`` if the file could not be read or parsed.
    """
    try:
        return json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError) as e:
        if logger is not None:
            logger.warning(f"Failed to read {path}: {e}")
        return None
