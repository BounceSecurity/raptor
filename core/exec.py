#!/usr/bin/env python3
"""Command execution utilities.

Provides safe, consistent command execution across RAPTOR packages.
All functions use list-based arguments (never shell=True) for security.
"""

import os
import subprocess
from pathlib import Path
from typing import IO, List, Optional, Tuple, Union

from core.config import RaptorConfig
from core.logging import get_logger

logger = get_logger()


def run(
    cmd,
    cwd=None,
    timeout=RaptorConfig.DEFAULT_TIMEOUT,
    env=None,
    stdin: Optional[IO] = None,
    input: Optional[str] = None,
) -> Tuple[int, str, str]:
    """Execute a command and return (returncode, stdout, stderr).

    Args:
        cmd: Command and arguments as a list (never use shell=True).
        cwd: Working directory for the subprocess.
        timeout: Seconds before the subprocess is killed.
        env: Environment variables dict; defaults to a copy of os.environ.
        stdin: Open file-like object to pass as stdin (e.g. a crash file for
            a fuzzer replay).  Mutually exclusive with *input*.
        input: String to write to stdin.  Mutually exclusive with *stdin*.

    Returns:
        Tuple of (returncode, stdout, stderr) all as strings.
    """
    if stdin is not None and input is not None:
        raise ValueError("stdin and input are mutually exclusive")

    cwd_str = str(cwd) if isinstance(cwd, Path) else cwd

    kwargs = dict(
        cwd=cwd_str,
        env=env or os.environ.copy(),
        text=True,
        timeout=timeout,
    )

    if stdin is not None:
        kwargs["stdin"] = stdin
        kwargs["capture_output"] = True
    elif input is not None:
        kwargs["input"] = input
        kwargs["capture_output"] = True
    else:
        kwargs["capture_output"] = True

    p = subprocess.run(cmd, **kwargs)
    return p.returncode, p.stdout, p.stderr
