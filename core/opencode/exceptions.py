#!/usr/bin/env python3
"""
OpenCode Exception Classes
"""


class OpenCodeError(Exception):
    """Base exception for OpenCode operations."""
    pass


class OpenCodeUnavailableError(OpenCodeError):
    """
    Raised when OpenCode is required but not available.
    
    This can happen when:
    - OpenCode server is not running
    - OpenCode Python client is not installed
    - Configuration is disabled
    """
    pass


class OpenCodeConnectionError(OpenCodeError):
    """
    Raised when connection to OpenCode server fails.
    
    This can happen when:
    - Network issues
    - Server not responding
    - Authentication failures
    """
    pass
