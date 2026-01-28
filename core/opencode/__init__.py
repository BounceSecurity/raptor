#!/usr/bin/env python3
"""
OpenCode Integration Module

Provides codebase intelligence capabilities through OpenCode server integration.
This is a reusable core module that can be used by any RAPTOR package.
"""

from .exceptions import (
    OpenCodeError,
    OpenCodeUnavailableError,
    OpenCodeConnectionError,
)
from .server_manager import OpenCodeServerManager
from .client_sdk import OpenCodeClient
from .analyzer import (
    analyze_codebase,
    CodebaseAnalysisResult,
)

__all__ = [
    "OpenCodeError",
    "OpenCodeUnavailableError",
    "OpenCodeConnectionError",
    "OpenCodeServerManager",
    "OpenCodeClient",
    "analyze_codebase",
    "CodebaseAnalysisResult",
]
