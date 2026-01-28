# OpenCode Integration with LiteLLM in RAPTOR

## Executive Summary

This document outlines a plan for integrating OpenCode with LiteLLM in RAPTOR through a **single new package** that provides sophisticated codebase interaction capabilities. This approach keeps the integration clean and isolated, without modifying existing packages.

**Key Points:**
- **New Package**: `packages/opencode_analysis/` - Standalone package using OpenCode + LiteLLM
- **Simple Interface**: Accepts repository path and prompt, uses OpenCode for codebase intelligence
- **RAPTOR-managed Server**: RAPTOR can automatically manage OpenCode server lifecycle
- **No Breaking Changes**: Existing packages remain unchanged
- **Clean Architecture**: Follows RAPTOR's modular package design principles

## Table of Contents

1. [Installation Requirements](#installation-requirements)
2. [New Package Design](#new-package-design)
3. [Current State Analysis](#current-state-analysis)
4. [OpenCode Capabilities](#opencode-capabilities)
5. [Integration Architecture](#integration-architecture)
6. [Core Architecture Integration](#core-architecture-integration)
7. [User Interaction Patterns](#user-interaction-patterns)
8. [Implementation Plan](#implementation-plan)
9. [Configuration](#configuration)
10. [Testing Strategy](#testing-strategy)
11. [Security Considerations](#security-considerations)
12. [Success Metrics](#success-metrics)
13. [Risks and Mitigations](#risks-and-mitigations)

---

## Installation Requirements

### Overview

OpenCode integration requires **two separate components**:

1. **OpenCode Server** - A separate service that must be installed and running (like Ollama)
2. **Python Client Library** - Python package installed via pip (like `litellm`)

**Important**: The new `packages/opencode_analysis/` package **requires** OpenCode to function. It will fail with clear error messages if OpenCode is not installed or unavailable.

**Recommended Setup**: RAPTOR automatically manages the OpenCode server lifecycle (see [RAPTOR-Managed Server](#raptor-managed-server) below). This makes setup much simpler - just install the Python package and RAPTOR handles the rest!

### Component 1: OpenCode Server (Required)

**What it is**: A separate server application that runs independently, similar to:
- **Ollama** (local LLM server) - RAPTOR connects to `http://localhost:11434`
- **CodeQL** (static analysis engine) - RAPTOR calls `codeql` command-line tool
- **Semgrep** (scanner) - RAPTOR calls `semgrep` command-line tool

**RAPTOR-Managed Server**:

RAPTOR automatically manages the OpenCode server lifecycle, making setup much simpler:

**How it works**:
- RAPTOR checks if OpenCode server is already running
- If not running, RAPTOR automatically spawns the server process
- Server runs for the duration of the RAPTOR session
- RAPTOR handles cleanup on exit (graceful shutdown)
- Port conflicts are handled automatically (tries next available port)

**Benefits**:
- ✅ No manual server management required
- ✅ Works out of the box after `pip install`
- ✅ Automatic cleanup on exit
- ✅ Handles port conflicts automatically
- ✅ Still works if server is already running externally

**Implementation** (in `core/opencode/server_manager.py`):

```python
class OpenCodeServerManager:
    """Manages OpenCode server lifecycle."""
    
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url
        self.process: Optional[subprocess.Popen] = None
        self.managed = False  # True if we started it
    
    def ensure_running(self) -> bool:
        """
        Ensure OpenCode server is running.
        
        Returns True if server is available (either already running or we started it).
        """
        # Check if server is already running
        if self._check_server_health():
            logger.debug("OpenCode server already running")
            return True
        
        # Try to start server
        if self._start_server():
            self.managed = True
            logger.info("Started OpenCode server (RAPTOR-managed)")
            return True
        
        return False
    
    def _check_server_health(self) -> bool:
        """Check if server is responding."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _start_server(self) -> bool:
        """Start OpenCode server as subprocess."""
        # Find opencode-server binary
        server_binary = self._find_server_binary()
        if not server_binary:
            return False
        
        # Start server in background
        try:
            self.process = subprocess.Popen(
                [server_binary, "--port", str(self._get_port())],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True  # Detach from parent
            )
            
            # Wait for server to be ready (with timeout)
            for _ in range(30):  # 30 seconds max
                time.sleep(1)
                if self._check_server_health():
                    return True
            
            # Server didn't start in time
            self.process.kill()
            self.process = None
            return False
        except Exception as e:
            logger.error(f"Failed to start OpenCode server: {e}")
            return False
    
    def _find_server_binary(self) -> Optional[Path]:
        """Find opencode-server binary."""
        # Check common locations
        possible_paths = [
            shutil.which("opencode-server"),
            Path.home() / ".local/bin/opencode-server",
            Path("/usr/local/bin/opencode-server"),
            Path("/usr/bin/opencode-server"),
        ]
        
        for path in possible_paths:
            if path and Path(path).exists():
                return Path(path)
        
        return None
    
    def shutdown(self):
        """Shutdown managed server."""
        if self.process and self.managed:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            logger.info("Shutdown OpenCode server (RAPTOR-managed)")
```

**Usage in OpenCodeClient**:

```python
class OpenCodeClient:
    def __init__(self, repo_path: Path, server_url: Optional[str] = None):
        """
        Initialize OpenCode client.
        
        Args:
            repo_path: Repository path
            server_url: Server URL (default: from config)
        
        Note: Server is automatically managed by RAPTOR.
        """
        self.server_manager = OpenCodeServerManager(server_url or RaptorConfig.OPENCODE_SERVER_URL)
        
        # Automatically ensure server is running
        if not self.server_manager.ensure_running():
            raise OpenCodeUnavailableError(
                "Cannot start OpenCode server. "
                "Please install opencode-server binary. "
                "See docs/OPENCODE_INTEGRATION.md for installation instructions."
            )
        
        # Register cleanup on exit
        if self.server_manager.managed:
            import atexit
            atexit.register(self.server_manager.shutdown)
```

**Configuration**:

```python
# core/config.py
class RaptorConfig:
    # ... existing config ...
    
    # OpenCode server management (always auto-managed)
    OPENCODE_SERVER_BINARY = os.getenv("OPENCODE_SERVER_BINARY")  # Custom path if needed
    OPENCODE_STARTUP_TIMEOUT = 30  # seconds to wait for server startup
```

**Environment Variables**:
```bash
# Custom server binary path (if not in PATH)
export OPENCODE_SERVER_BINARY="/custom/path/to/opencode-server"

# Custom server URL (if using different port)
export OPENCODE_SERVER_URL="http://localhost:9000"
```

**Note**: 
- RAPTOR automatically manages the server lifecycle (start, health checks, shutdown)
- If a server is already running at the configured URL, RAPTOR will use it
- Server is automatically shut down when RAPTOR exits

**Future Option**: External server management (manual server control) may be added in the future for production deployments requiring persistent servers or custom configurations.

### Component 2: Python Client Library (Required)

**What it is**: Python package that provides the client interface to communicate with OpenCode server.

**Installation**:
Add to `requirements.txt`:
```txt
opencode>=0.1.0  # OpenCode Python client library
```

Then install:
```bash
pip install -r requirements.txt
```

**What it provides**:
- `OpenCodeClient` class in `core/opencode/client.py`
- API to communicate with OpenCode server
- LSP connection management
- Tool execution interface

**Note**: This is just a Python library (like `litellm`, `requests`). It does not include the OpenCode server itself.

### Component 3: LSP Servers (Optional, Auto-managed)

**What they are**: Language Server Protocol servers for code intelligence (definitions, references, etc.)

**Installation**:
- **Option 1**: OpenCode server can auto-install/manage LSP servers
- **Option 2**: Install manually (e.g., `pyright` for Python, `typescript-language-server` for TypeScript)
- **Option 3**: Use system-installed LSP servers

**Supported Languages** (configurable):
- Python (pyright, pylsp, jedi)
- JavaScript/TypeScript (typescript-language-server)
- Java (eclipse.jdt.ls)
- Go (gopls)
- Rust (rust-analyzer)
- C/C++ (clangd)
- And more...

**Configuration**:
```bash
# Specify which languages to enable LSP for
export OPENCODE_LSP_LANGUAGES="python,javascript,typescript"
```

**Note**: LSP servers are optional. OpenCode works without them (file operations only), but LSP enables semantic code intelligence features.

### Complete Installation Steps

```bash
# Step 1: Install Python client (includes OpenCode client)
cd raptor
pip install -r requirements.txt

# Step 2: Run RAPTOR - it will auto-install and start OpenCode server!
python3 packages/opencode_analysis/agent.py \
    --repo /path/to/code \
    --prompt "Analyze this codebase"

# That's it! RAPTOR will:
# - Auto-download OpenCode server binary if not found
# - Auto-start the server when needed
# - Auto-shut down when RAPTOR exits
```

**Auto-Installation Details**:
- RAPTOR automatically downloads the OpenCode server binary on first use
- **Runs in place**: Binary is cached in `out/opencode_bin/` directory (no system installation)
- **Portable**: No system-wide installation required - runs from cache directory
- Platform-specific binaries are downloaded (Linux, macOS, Windows)
- Handles both standalone binaries and archives (auto-extracts if needed)
- To disable auto-install: `export OPENCODE_AUTO_INSTALL=false`
- Manual installation: Download from https://github.com/anomalyco/opencode/releases

### Comparison with Other RAPTOR Dependencies

| Component | Type | Installation | Management | Similar To |
|-----------|------|--------------|------------|------------|
| **OpenCode Server** | Separate service | User installs binary | **Auto-managed by RAPTOR** | Unlike Ollama/CodeQL, RAPTOR manages lifecycle |
| **opencode (Python)** | Python library | `pip install opencode` | N/A | `litellm`, `requests` |
| **LSP Servers** | Language tools | Auto-managed or manual | Auto or Manual | System tools (gcc, python) |

**Key Difference**: Unlike Ollama/CodeQL which require manual server management, OpenCode server is **automatically managed by RAPTOR** - no manual server startup needed!

---

## New Package Design

### Package: `packages/opencode_analysis/`

**Purpose**: Thin CLI wrapper around core OpenCode analysis functionality.

**Design Principles**:
- **Thin Wrapper**: All logic is in `core/opencode/` for reuse by other packages
- **Simple CLI**: Just handles argument parsing and calls core functions
- **Reusable Core**: Other packages can use `core.opencode.analyze_codebase()` directly
- **No Package Logic**: All analysis logic lives in core module

### Package Structure

```
packages/opencode_analysis/
├── __init__.py
└── agent.py              # Thin CLI wrapper (calls core/opencode/analyzer.py)

core/opencode/            # Reusable core module
├── __init__.py
├── server_manager.py     # Server lifecycle management
├── client.py            # OpenCode client API
├── analyzer.py           # Core analysis logic (reusable by any package)
└── exceptions.py         # Error classes
```

### CLI Interface

```bash
# Basic usage
python3 packages/opencode_analysis/agent.py \
    --repo /path/to/code \
    --prompt "Analyze this codebase for security vulnerabilities"

# With output directory
python3 packages/opencode_analysis/agent.py \
    --repo /path/to/code \
    --prompt "Find all SQL injection vulnerabilities" \
    --out /path/to/output

# With custom model
python3 packages/opencode_analysis/agent.py \
    --repo /path/to/code \
    --prompt "Review authentication mechanisms" \
    --model claude-opus-4.5
```

### Package API

**Main Entry Point** (`agent.py`):

```python
#!/usr/bin/env python3
"""
OpenCode Analysis Agent

Uses OpenCode for sophisticated codebase interactions combined with LLM analysis.
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for core imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.config import RaptorConfig
from core.logging import get_logger
from core.llm.client import LLMClient
from core.llm.config import LLMConfig
from packages.opencode_analysis.analyzer import OpenCodeAnalyzer

logger = get_logger()


def main():
    parser = argparse.ArgumentParser(
        description="OpenCode-powered codebase analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python3 packages/opencode_analysis/agent.py \\
    --repo /path/to/code \\
    --prompt "Find security vulnerabilities"

  # Custom output
  python3 packages/opencode_analysis/agent.py \\
    --repo /path/to/code \\
    --prompt "Analyze authentication" \\
    --out ./results
        """
    )
    
    parser.add_argument("--repo", required=True, 
                       help="Path to repository to analyze")
    parser.add_argument("--prompt", required=True,
                       help="Analysis prompt/question")
    parser.add_argument("--out", 
                       help="Output directory (default: auto-generated)")
    parser.add_argument("--model",
                       help="LLM model to use (default: from config)")
    parser.add_argument("--opencode-server",
                       help="OpenCode server URL (default: http://localhost:8080)")
    
    args = parser.parse_args()
    
    # Validate repo path
    repo_path = Path(args.repo).resolve()
    if not repo_path.exists():
        print(f"ERROR: Repository path does not exist: {repo_path}")
        sys.exit(1)
    
    # Initialize LLM client
    llm_config = LLMConfig()
    if args.model:
        # Override model if specified
        llm_config.primary_model.model_name = args.model
    
    llm_client = LLMClient(llm_config)
    
    # Create analyzer
    analyzer = OpenCodeAnalyzer(
        repo_path=repo_path,
        llm_client=llm_client,
        opencode_server_url=args.opencode_server
    )
    
    # Run analysis
    try:
        result = analyzer.analyze(args.prompt, output_dir=args.out)
        
        # Print results
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nResult saved to: {result.output_dir}")
        print(f"Files analyzed: {result.files_analyzed}")
        print(f"LLM calls: {result.llm_calls}")
        print(f"Cost: ${result.cost:.4f}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**Core Module** (`core/opencode/analyzer.py` - reusable by any package):

```python
#!/usr/bin/env python3
"""
OpenCode Analyzer

Core analysis logic combining OpenCode codebase intelligence with LLM analysis.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

from core.logging import get_logger
from core.llm.client import LLMClient
from core.opencode import OpenCodeClient, OpenCodeUnavailableError

logger = get_logger()


@dataclass
class AnalysisResult:
    """Result of codebase analysis."""
    output_dir: Path
    prompt: str
    response: str
    files_analyzed: int
    llm_calls: int
    cost: float
    opencode_operations: List[Dict[str, Any]]


def analyze_codebase(
    repo_path: Path,
    prompt: str,
    llm_client: LLMClient,
    output_dir: Optional[Path] = None,
    opencode_server_url: Optional[str] = None
) -> CodebaseAnalysisResult:
    """
    Analyze a codebase using OpenCode + LLM.
    
    This is the core reusable function that any package can use.
    
    Args:
        repo_path: Path to repository to analyze
        prompt: Analysis prompt/question
        llm_client: LLM client for analysis
        output_dir: Output directory (auto-generated if not provided)
        opencode_server_url: Optional OpenCode server URL
    
    Returns:
        CodebaseAnalysisResult with analysis output and metadata
    
    Raises:
        OpenCodeUnavailableError: If OpenCode is required but unavailable
    """
    # Implementation in core/opencode/analyzer.py
    # This function handles:
    # - OpenCode client initialization
    # - Codebase context gathering
    # - LLM analysis with codebase tools
    # - Result saving and metadata
```
```

---

## Installation Requirements

### Troubleshooting

**Server Not Found** (for required packages):
1. Check if OpenCode server is running: `curl http://localhost:8080/health`
2. Verify server URL: `echo $OPENCODE_SERVER_URL`
3. Check firewall/network settings
4. **Required packages will fail** with `OpenCodeUnavailableError` - install and start OpenCode server

**LSP Features Not Working**:
1. Check if LSP is enabled: `export OPENCODE_ENABLE_LSP="true"`
2. Verify LSP servers are installed for your languages
3. Check OpenCode server logs for LSP errors
4. Some features may be unavailable if LSP servers are not configured

**Python Import Errors**:
```bash
pip install opencode>=0.1.0
# Or
pip install -r requirements.txt
```

---

## Current State Analysis

### LiteLLM Usage in RAPTOR

**Location**: `core/llm/`

**Current Architecture**:
- `client.py`: `LLMClient` class with fallback, caching, cost tracking
- `providers.py`: `LiteLLMProvider` wrapping litellm + instructor for structured outputs
- `config.py`: Model configuration with automatic provider detection

**Key Features**:
- Multi-provider support (Anthropic, OpenAI, Gemini, Ollama)
- Automatic fallback between models
- Cost tracking and budget limits
- Response caching
- Structured output via Instructor + Pydantic
- Task-specific model selection

**Current Limitations**:
- Packages manually read files using `open()` and `Path.read_text()`
- No semantic code understanding (definitions, references, call hierarchy)
- Limited code search capabilities (basic grep patterns)
- No LSP integration for code intelligence
- Manual context extraction (fixed line ranges around vulnerabilities)

### Package Codebase Interaction Patterns

**Current Approach** (from codebase review):

**Current Approach** (existing packages - unchanged):
- `packages/llm_analysis/`: Manual file reading with fixed context windows
- `packages/codeql/`: Manual code extraction for dataflow validation
- `packages/static-analysis/`: Rule-based pattern matching

**New Package Approach** (`packages/opencode_analysis/`):
- Uses OpenCode for semantic code understanding
- LSP-based code navigation (definitions, references, call hierarchy)
- Intelligent context extraction (function/class boundaries)
- Tool-based LLM interactions for codebase exploration

**Common Patterns**:
- Direct file I/O: `open(file_path, "r")` and `readlines()`
- Fixed context windows (50 lines before/after)
- Manual path resolution and URI parsing
- No understanding of code structure, imports, or call graphs

---

## OpenCode Capabilities

### Built-in Tools

1. **File Operations**:
   - `read`: Read file contents with optional line ranges
   - `write`: Write file contents
   - `edit`: Apply structured edits (insert, replace, delete)
   - `glob`: Pattern-based file discovery
   - `list`: Directory listing

2. **Code Search**:
   - `grep`: Pattern-based search across codebase
   - Semantic search (via LSP integration)

3. **Code Intelligence (LSP)**:
   - `definitions`: Find symbol definitions
   - `references`: Find all references to a symbol
   - `call_hierarchy`: Understand function call relationships
   - `hover`: Get symbol information and documentation
   - `completion`: Code completion suggestions
   - `diagnostics`: Real-time error detection

4. **Execution**:
   - `bash`: Execute shell commands (for builds, tests, etc.)

5. **Extensibility**:
   - Custom tool registration
   - MCP (Model Context Protocol) server support
   - Permission-based access control

### Architecture

OpenCode uses a client-server model:
- **Server**: Manages codebase state, LSP connections, tool execution
- **Client**: LLM interface that invokes tools via structured function calls

---

## Integration Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                    RAPTOR Packages Layer                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  packages/opencode_analysis/ (NEW)                   │  │
│  │  - Standalone package using OpenCode + LiteLLM      │  │
│  │  - Takes repo path + prompt                         │  │
│  │  - Uses OpenCodeClient via LLMClient                │  │
│  └──────────────────────────────────────────────────────┘  │
│  (Other packages unchanged - llm_analysis, codeql, etc.)   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core Layer (Reusable)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  core/opencode/ (NEW)                                 │  │
│  │  ├── __init__.py                                      │  │
│  │  ├── client.py (OpenCodeClient)                       │  │
│  │  ├── lsp_manager.py (LSP lifecycle)                  │  │
│  │  └── tools.py (Tool definitions)                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                       │                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  core/llm/ (EXISTING - Enhanced)                     │  │
│  │  ├── client.py (adds create_opencode_client())       │  │
│  │  ├── providers.py (adds OpenCodeToolProvider)        │  │
│  │  └── config.py (adds OpenCode config)                │  │
│  └──────────────────────────────────────────────────────┘  │
│                       │                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  core/config.py (EXISTING - Enhanced)                │  │
│  │  - Adds OpenCode configuration options                │  │
│  │  - Server URL, LSP settings, permissions             │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    LiteLLM Layer (Existing)                 │
│  - Multi-provider abstraction                                │
│  - Cost tracking & budget management                          │
│  - Fallback logic                                            │
│  - Response caching                                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              OpenCode Server (External)                     │
│  - LSP server management (per language)               │
│  - File system operations                                    │
│  - Tool execution                                            │
│  - Codebase state management                                 │
└─────────────────────────────────────────────────────────────┘
```

### Core Architecture Integration

Following RAPTOR's modular architecture principles:

1. **Core Layer (`core/opencode/`)**: New reusable module following `core/` patterns
   - Shared utilities for all packages
   - No package-specific logic
   - Clean abstraction over OpenCode server

2. **LLM Integration (`core/llm/`)**: Enhanced existing module
   - Extends `LLMClient` with OpenCode factory method
   - Adds `OpenCodeToolProvider` for tool-based LLM interactions
   - Maintains backward compatibility

3. **Configuration (`core/config.py`)**: Enhanced existing config
   - Adds OpenCode settings to `RaptorConfig`
   - Environment variable support
   - Consistent with existing config patterns

---

## Core Architecture Integration

### Reusable Core Module Design

OpenCode integration follows RAPTOR's established patterns for reusable core infrastructure, similar to how `core/sarif/`, `core/logging/`, and `core/config.py` are structured.

**Key Principles**:
1. **Shared Infrastructure**: All packages can use OpenCode without duplicating code
2. **Clean Abstraction**: Packages import `from core.opencode import OpenCodeClient`
3. **No Package Dependencies**: Core modules only import from other core modules
4. **Consistent Patterns**: Follows same structure as existing core modules

### Directory Structure

Following RAPTOR's modular architecture, OpenCode integration fits into the core layer:

```
core/
├── __init__.py
├── config.py (enhanced with OpenCode settings)
├── logging.py
├── progress.py
├── sarif/
│   └── parser.py
├── llm/ (existing - enhanced)
│   ├── __init__.py
│   ├── client.py (adds create_opencode_client())
│   ├── providers.py (adds OpenCodeToolProvider)
│   └── config.py (adds OpenCode config)
└── opencode/ (NEW - reusable core module)
    ├── __init__.py
    ├── server_manager.py (Automatic server lifecycle management)
    ├── client.py (OpenCodeClient)
    ├── lsp_manager.py (LSP lifecycle management)
    └── tools.py (Tool definitions and utilities)

packages/
└── opencode_analysis/ (NEW - standalone package)
    ├── __init__.py
    ├── agent.py (CLI entry point)
    ├── analyzer.py (Core analysis logic)
    └── README.md
```

### Design Principles

1. **Reusable Core**: `core/opencode/` follows same patterns as `core/sarif/`, `core/logging/`
2. **No Package Dependencies**: Core modules only import from other core modules
3. **Clean Abstraction**: Packages import `from core.opencode import OpenCodeClient`
4. **Backward Compatible**: Existing code continues to work without OpenCode
5. **Opt-in**: Packages explicitly enable OpenCode features

### Component Design

#### 1. OpenCode Server Manager (`core/opencode/server_manager.py`)

**Purpose**: Automatically manage OpenCode server lifecycle

**Key Features**:
- Check if server is already running
- Spawn server process if needed
- Handle port conflicts
- Graceful shutdown on exit
- Health check verification

**Key Methods**:
```python
class OpenCodeServerManager:
    def ensure_running(self) -> bool:
        """Ensure server is running (start if needed)."""
        
    def _check_server_health(self) -> bool:
        """Check if server is responding."""
        
    def _start_server(self) -> bool:
        """Start server as subprocess."""
        
    def shutdown(self):
        """Shutdown managed server."""
        
    def _find_server_binary(self) -> Optional[Path]:
        """Find opencode-server binary in PATH or common locations."""
```

#### 2. OpenCode Client (`core/opencode/client.py`)

**Purpose**: High-level API for codebase interactions (reusable across all packages)

**Key Methods**:
```python
class OpenCodeClient:
    def __init__(self, repo_path: Path, server_url: Optional[str] = None):
        """Initialize OpenCode client for a repository."""
        
    def read_file(self, file_path: str, start_line: Optional[int] = None, 
                  end_line: Optional[int] = None) -> str:
        """Read file with optional line range."""
        
    def find_definitions(self, symbol: str, file_path: str, line: int) -> List[Location]:
        """Find symbol definitions using LSP."""
        
    def find_references(self, symbol: str, file_path: str, line: int) -> List[Location]:
        """Find all references to a symbol."""
        
    def get_call_hierarchy(self, symbol: str, file_path: str, line: int) -> CallHierarchy:
        """Get function call hierarchy."""
        
    def semantic_search(self, query: str, file_pattern: Optional[str] = None) -> List[SearchResult]:
        """Semantic code search across codebase."""
        
    def grep(self, pattern: str, file_pattern: Optional[str] = None) -> List[Match]:
        """Pattern-based search."""
        
    def get_context_around(self, file_path: str, line: int, 
                          context_lines: int = 50) -> str:
        """Get code context with intelligent boundaries (function/class)."""
        
    def get_imports(self, file_path: str) -> List[str]:
        """Get imports for a file."""
        
    def execute_bash(self, command: str, cwd: Optional[Path] = None) -> CommandResult:
        """Execute shell command safely."""
```

#### 3. LSP Manager (`core/opencode/lsp_manager.py`)

**Purpose**: Manage LSP server lifecycle per repository

**Key Features**:
- Auto-detect languages in repository
- Start/stop LSP servers per language
- Connection pooling and lifecycle management
- Graceful degradation if LSP unavailable

```python
class LSPManager:
    """Manages LSP server connections for a repository."""
    
    def __init__(self, repo_path: Path, languages: Optional[List[str]] = None):
        """Initialize LSP manager for repository."""
        
    def detect_languages(self) -> List[str]:
        """Auto-detect languages in repository."""
        
    def start_lsp_servers(self, languages: List[str]) -> Dict[str, LSPConnection]:
        """Start LSP servers for specified languages."""
        
    def get_lsp_connection(self, language: str) -> Optional[LSPConnection]:
        """Get LSP connection for language."""
        
    def shutdown(self):
        """Shutdown all LSP servers."""
```

#### 4. OpenCode Tool Provider (`core/llm/providers.py` - additions)

**Purpose**: Bridge between LiteLLM and OpenCode tools

**Key Features**:
- Extends `LiteLLMProvider` interface
- Registers OpenCode tools as LiteLLM function calling tools
- Handles tool call execution
- Injects codebase context into LLM prompts

```python
class OpenCodeToolProvider(LiteLLMProvider):
    """LLM provider that integrates OpenCode tools."""
    
    def __init__(self, config: ModelConfig, opencode_client: OpenCodeClient):
        super().__init__(config)
        self.opencode_client = opencode_client
        self._register_tools()
        
    def _register_tools(self):
        """Register OpenCode tools as LiteLLM function calling tools."""
        # Convert OpenCode tools to OpenAI function calling format
        # Register with litellm
        
    def generate_with_tools(self, prompt: str, system_prompt: Optional[str] = None,
                           **kwargs) -> LLMResponse:
        """Generate with automatic tool calling."""
        # Use litellm's function calling with OpenCode tools
```

#### 5. Enhanced LLM Client (`core/llm/client.py` - modifications)

**New Methods**:
```python
class LLMClient:
    def create_opencode_client(self, repo_path: Path, required: bool = False) -> 'OpenCodeClient':
        """
        Create OpenCode client for a repository.
        
        Args:
            repo_path: Path to repository
            required: If True, raise OpenCodeUnavailableError if unavailable.
                     If False, return None (for optional usage).
        
        Returns:
            OpenCodeClient instance
            
        Raises:
            OpenCodeUnavailableError: If required=True and OpenCode is unavailable
        """
        if not self.config.enable_opencode:
            if required:
                raise OpenCodeUnavailableError(
                    "OpenCode is required but disabled. "
                    "Set enable_opencode=True in LLMConfig or set OPENCODE_ENABLED=true"
                )
            return None
        
        try:
            from core.opencode import OpenCodeClient
            client = OpenCodeClient(repo_path)
            # Verify connection
            client.verify_connection()
            return client
        except OpenCodeConnectionError as e:
            if required:
                raise OpenCodeUnavailableError(
                    f"OpenCode is required but server is unavailable: {e}. "
                    "Please install and start OpenCode server. "
                    "See docs/OPENCODE_INTEGRATION.md for installation instructions."
                ) from e
            logger.warning(f"OpenCode unavailable: {e}")
            return None
        except ImportError as e:
            if required:
                raise OpenCodeUnavailableError(
                    f"OpenCode Python client not installed: {e}. "
                    "Install with: pip install opencode>=0.1.0"
                ) from e
            logger.warning(f"OpenCode client not available: {e}")
            return None
        
    def generate_with_codebase(self, prompt: str, repo_path: Path,
                              system_prompt: Optional[str] = None,
                              **kwargs) -> LLMResponse:
        """
        Generate with automatic codebase context and tool access.
        
        Creates OpenCode client, registers tools, generates response.
        Falls back to regular generation if OpenCode unavailable.
        """
        # For generate_with_codebase, OpenCode is required
        opencode = self.create_opencode_client(repo_path, required=True)
        # Use OpenCodeToolProvider for tool-based generation
        provider = OpenCodeToolProvider(self.config.primary_model, opencode)
        return provider.generate_with_tools(prompt, system_prompt, **kwargs)
```

---

## User Interaction Patterns

### CLI Interface

Following RAPTOR's CLI patterns, OpenCode is controlled via command-line arguments:

#### Unified Launcher (`raptor.py`)

```bash
# OpenCode auto-enabled if available (default behavior)
python3 raptor.py agentic --repo /path/to/code

# Explicitly enable OpenCode
python3 raptor.py agentic --repo /path/to/code --opencode

# Disable OpenCode (use manual operations)
python3 raptor.py agentic --repo /path/to/code --no-opencode

# Custom OpenCode server
python3 raptor.py agentic --repo /path/to/code \
    --opencode-server http://localhost:9000

# Specify LSP languages
python3 raptor.py codeql --repo /path/to/code \
    --opencode-languages python,javascript,typescript
```

#### New Package Usage

```bash
# OpenCode analysis package
python3 packages/opencode_analysis/agent.py \
    --repo /path/to/code \
    --prompt "Analyze this codebase for security vulnerabilities"

# With custom model
python3 packages/opencode_analysis/agent.py \
    --repo /path/to/code \
    --prompt "Find SQL injection vulnerabilities" \
    --model claude-opus-4.5
```

### Configuration

#### Environment Variables

Following RAPTOR's config patterns (like `OLLAMA_HOST`, `RAPTOR_OUT_DIR`):

```bash
# OpenCode server URL
export OPENCODE_SERVER_URL="http://localhost:8080"

# Enable/disable LSP
export OPENCODE_ENABLE_LSP="true"

# Comma-separated list of languages for LSP
export OPENCODE_LSP_LANGUAGES="python,javascript,typescript"

# Disable OpenCode entirely
export OPENCODE_ENABLED="false"
```

#### Core Configuration (`core/config.py`)

OpenCode settings added to `RaptorConfig` class:

```python
class RaptorConfig:
    # ... existing config ...
    
    # OpenCode Configuration (following existing patterns)
    OPENCODE_SERVER_URL = os.getenv("OPENCODE_SERVER_URL", "http://localhost:8080")
    OPENCODE_ENABLE_LSP = os.getenv("OPENCODE_ENABLE_LSP", "true").lower() == "true"
    OPENCODE_LSP_LANGUAGES = [
        "python", "javascript", "typescript", "java", 
        "go", "rust", "cpp", "c", "ruby", "php"
    ]
    OPENCODE_CACHE_ENABLED = True
    OPENCODE_CACHE_DIR = BASE_OUT_DIR / "opencode_cache"
    OPENCODE_TIMEOUT = 30  # seconds for LSP operations
    OPENCODE_MAX_WORKERS = 2  # parallel LSP server connections
```

#### LLM Config Enhancement (`core/llm/config.py`)

Add to `LLMConfig` dataclass:

```python
@dataclass
class LLMConfig:
    # ... existing fields ...
    
    # OpenCode integration
    enable_opencode: bool = True
    opencode_tool_permissions: Dict[str, str] = field(default_factory=lambda: {
        "read": "allow",
        "write": "deny",  # Security: prevent accidental writes
        "bash": "approval",  # Require approval for execution
        "grep": "allow",
        "definitions": "allow",
        "references": "allow",
        "call_hierarchy": "allow",
        "semantic_search": "allow",
    })
```

#### CLI Arguments

Add to entry point scripts following RAPTOR's CLI patterns:

**`raptor.py`** (unified launcher):
```python
# Add new mode for OpenCode analysis
def mode_opencode(args: list) -> int:
    """Run OpenCode-powered analysis."""
    script_root = Path(__file__).parent
    opencode_script = script_root / "packages/opencode_analysis/agent.py"
    
    if not opencode_script.exists():
        print(f"✗ OpenCode analysis script not found: {opencode_script}")
        return 1
    
    print("\n[*] Running OpenCode-powered analysis...\n")
    return run_script(opencode_script, args)
```

**`packages/opencode_analysis/agent.py`**:
```python
parser.add_argument("--repo", required=True,
                   help="Path to repository to analyze")
parser.add_argument("--prompt", required=True,
                   help="Analysis prompt/question")
parser.add_argument("--out",
                   help="Output directory (default: auto-generated)")
parser.add_argument("--model",
                   help="LLM model to use (default: from config)")
parser.add_argument("--opencode-server",
                   help="OpenCode server URL (default: http://localhost:8080)")
```

### Programmatic Usage

#### Using the New Package

**CLI Usage**:
```bash
# Basic analysis
python3 packages/opencode_analysis/agent.py \
    --repo /path/to/code \
    --prompt "Find all SQL injection vulnerabilities"

# With custom output
python3 packages/opencode_analysis/agent.py \
    --repo /path/to/code \
    --prompt "Analyze authentication mechanisms" \
    --out ./results
```

**Programmatic Usage**:
```python
from core.llm.client import LLMClient
from core.llm.config import LLMConfig
from packages.opencode_analysis.analyzer import OpenCodeAnalyzer
from pathlib import Path

# Initialize
llm_client = LLMClient(LLMConfig())
analyzer = OpenCodeAnalyzer(
    repo_path=Path("/path/to/code"),
    llm_client=llm_client
)

# Run analysis
result = analyzer.analyze(
    prompt="Find security vulnerabilities in authentication code"
)

print(f"Analysis complete: {result.output_dir}")
print(f"Cost: ${result.cost:.4f}")
```

#### Direct Core Module Access

Packages can import OpenCode directly from core:

```python
from core.opencode import OpenCodeClient
from core.config import RaptorConfig

# Create client for repository
opencode = OpenCodeClient(repo_path)

# Use codebase intelligence
definitions = opencode.find_definitions("vulnerable_function", "src/main.py", 42)
references = opencode.find_references("vulnerable_function", "src/main.py", 42)
context = opencode.get_context_around("src/main.py", 42, context_lines=50)
```

#### Tool-Based LLM Interactions

For advanced use cases, use OpenCode tools with LLM function calling:

```python
# Generate with automatic tool access
response = llm_client.generate_with_codebase(
    prompt="Analyze this vulnerability and find all call sites",
    repo_path=repo_path,
    system_prompt="You are a security analyst..."
)

# LLM can automatically call OpenCode tools:
# - find_references()
# - get_call_hierarchy()
# - semantic_search()
# - read_file()
```

### Error Handling

The `opencode_analysis` package requires OpenCode and will raise clear errors if unavailable:

```python
from core.opencode import OpenCodeUnavailableError, OpenCodeConnectionError

try:
    analyzer = OpenCodeAnalyzer(repo_path, llm_client)
    result = analyzer.analyze(prompt)
except OpenCodeUnavailableError as e:
    # Clear error message with installation instructions
    print(f"ERROR: {e}")
    print("\nInstallation steps:")
    print("1. Install OpenCode server binary")
    print("2. RAPTOR will auto-start server, or start manually")
    print("3. Verify: curl http://localhost:8080/health")
    sys.exit(1)
except OpenCodeConnectionError as e:
    # Connection-specific error
    print(f"ERROR: Cannot connect to OpenCode server: {e}")
    print("Verify server is running: curl http://localhost:8080/health")
    sys.exit(1)
```

### Error Types

- **`OpenCodeUnavailableError`**: OpenCode is required but not available
  - Server not running
  - Python client not installed
  - Configuration disabled
  
- **`OpenCodeConnectionError`**: Cannot connect to OpenCode server
  - Network issues
  - Server not responding
  - Authentication failures

---

## Implementation Plan

### Phase 1: Foundation (Week 1-2)

**Goal**: Basic OpenCode integration without LSP, following RAPTOR core patterns

1. **Create Core Module Structure**:
   - Create `core/opencode/` directory following `core/sarif/` pattern
   - Add `__init__.py` with exports
   - Implement `core/opencode/server_manager.py` for automatic server management
   - Implement `core/opencode/client.py` with basic file operations

2. **Install Python Client Library**:
   - Add to `requirements.txt`: `opencode>=0.1.0` (or appropriate version)
   - Document that OpenCode server must be installed separately

3. **Documentation for Users**:
   - Update `DEPENDENCIES.md` with OpenCode entry
   - Clarify: Python library vs. separate server component
   - Provide installation steps for OpenCode server

4. **Core Configuration**:
   - Add OpenCode settings to `core/config.py` (`RaptorConfig`)
   - Follow existing config patterns (environment variables, defaults)
   - Default to `http://localhost:8080` (similar to Ollama pattern)

5. **LLM Client Integration**:
   - Add `create_opencode_client()` to `core/llm/client.py`
   - Implement `required` parameter (raises error if required=True and unavailable)
   - Add connection health check and verification
   - Add `OpenCodeUnavailableError` and `OpenCodeConnectionError` exception classes
   - Add to `LLMConfig` dataclass

6. **Testing**:
   - Unit tests in `core/tests/test_opencode.py`
   - Integration test with sample repository
   - Test error handling when server unavailable (required=True)
   - Test optional behavior (required=False)
   - Test connection failure handling

**Deliverables**:
- `core/opencode/server_manager.py` (automatic server lifecycle management)
- `core/opencode/client.py` (basic file operations)
- `core/opencode/__init__.py`
- Enhanced `core/config.py` and `core/llm/client.py`
- Updated `DEPENDENCIES.md`
- Basic tests (including server management tests)
- Updated `requirements.txt` (Python client only)

### Phase 2: LSP Integration (Week 3-4)

**Goal**: Add semantic code understanding

1. **LSP Server Management**:
   - Auto-detect languages in repository
   - Start/stop LSP servers per language
   - Connection pooling and lifecycle management

2. **Code Intelligence Methods**:
   - `find_definitions()`: Symbol definition lookup
   - `find_references()`: Reference finding
   - `get_call_hierarchy()`: Function call graphs
   - `hover()`: Symbol information

3. **Enhanced Context Extraction**:
   - `get_context_around()`: Intelligent context (function/class boundaries)
   - `get_related_code()`: Find related code via LSP

4. **Testing**:
   - Test with multi-language repositories
   - Validate LSP server lifecycle
   - Test semantic search accuracy

**Deliverables**:
- LSP integration in `OpenCodeClient`
- Enhanced context extraction methods
- Multi-language test suite

### Phase 3: Tool Provider Integration (Week 5-6)

**Goal**: Integrate OpenCode tools with LiteLLM function calling

1. **OpenCode Tool Provider**:
   - Implement `OpenCodeToolProvider` class
   - Convert OpenCode tools to OpenAI function calling format
   - Register tools with LiteLLM

2. **Automatic Tool Calling**:
   - Enable function calling in LiteLLM requests
   - Handle tool execution and response injection
   - Support multi-turn tool usage

3. **Enhanced LLM Client**:
   - Add `generate_with_codebase()` method
   - Automatic tool registration per repository
   - Context-aware prompt generation

4. **Testing**:
   - Test tool calling with various LLM providers
   - Validate tool response handling
   - Test multi-turn conversations with tools

**Deliverables**:
- `OpenCodeToolProvider` implementation
- Enhanced `LLMClient` methods
- Tool calling integration tests

### Phase 4: Package Integration (Week 7-8)

**Goal**: Update packages to use OpenCode capabilities following RAPTOR patterns

1. **LLM Analysis Package** (`packages/llm_analysis/`) - **REQUIRES OpenCode**:
   - Update `VulnerabilityContext.read_vulnerable_code()` to use OpenCode (required=True)
   - Remove manual fallback - OpenCode is required for this package
   - Use `find_references()` for dataflow analysis
   - Use `get_call_hierarchy()` for impact analysis
   - Enhanced context extraction with semantic boundaries
   - Raise clear error if OpenCode unavailable with installation instructions

2. **CodeQL Package** (`packages/codeql/autonomous_analyzer.py`) - **REQUIRES OpenCode**:
   - Update `autonomous_analyzer.py` to use OpenCode for code reading (required=True)
   - Remove manual fallback - OpenCode is required for autonomous analysis
   - Leverage LSP for better dataflow path validation
   - Use `find_definitions()` for taint source/sink analysis
   - Update `raptor_codeql.py` CLI to require OpenCode for autonomous mode
   - Raise clear error if OpenCode unavailable

3. **Static Analysis Package** (`packages/static-analysis/`) - **Optional OpenCode**:
   - Use OpenCode grep for pattern-based searches (optional, required=False)
   - Leverage semantic search for finding similar code patterns (if available)
   - Update `scanner.py` to accept `--opencode` flag (optional enhancement)
   - Maintains backward compatibility - works without OpenCode

4. **Entry Point Updates**:
   - Update `raptor.py` to pass OpenCode flags to packages
   - Update `raptor_agentic.py` with OpenCode CLI arguments
   - Update `raptor_codeql.py` with OpenCode options

5. **Testing**:
   - End-to-end tests for each updated package
   - Test with `--opencode` enabled and disabled
   - Performance benchmarks (with/without OpenCode)
   - Accuracy validation

**Deliverables**:
- Updated package implementations (with backward compatibility)
- Enhanced CLI interfaces
- Comprehensive test suite
- Performance benchmarks
- User documentation updates

### Phase 5: Advanced Features (Week 9-10)

**Goal**: Advanced capabilities and optimization

1. **Caching Layer**:
   - Cache LSP responses (definitions, references)
   - Cache file reads with change detection
   - Optimize repeated queries

2. **Permission System**:
   - Implement tool permission controls
   - Configurable access levels per package
   - Audit logging for tool usage

3. **Error Handling**:
   - Graceful degradation if OpenCode unavailable
   - Fallback to manual file reading
   - LSP server failure recovery

4. **Documentation**:
   - Usage examples for each package
   - API documentation
   - Migration guide from manual file operations

**Deliverables**:
- Caching implementation
- Permission system
- Comprehensive documentation
- Migration guide

---

## New Package Benefits

### `packages/opencode_analysis/`

**Purpose**: Standalone package that demonstrates OpenCode + LiteLLM integration

**Key Features**:
- **Simple Interface**: Just provide repo path and prompt
- **OpenCode Intelligence**: Uses LSP for semantic code understanding
- **LLM Analysis**: Uses LiteLLM for multi-provider LLM access
- **Tool-Based Interactions**: LLM can call OpenCode tools automatically
- **Self-Contained**: No dependencies on other RAPTOR packages

**Capabilities**:
- Intelligent codebase navigation (definitions, references, call hierarchy)
- Semantic code search
- Context-aware analysis (function/class boundaries)
- Multi-file analysis with code relationships
- Tool-based LLM interactions (LLM can explore codebase via tools)

**Example Use Cases**:
```bash
# Security analysis
python3 packages/opencode_analysis/agent.py \
    --repo /path/to/code \
    --prompt "Find all SQL injection vulnerabilities"

# Code review
python3 packages/opencode_analysis/agent.py \
    --repo /path/to/code \
    --prompt "Review authentication mechanisms for security issues"

# Architecture analysis
python3 packages/opencode_analysis/agent.py \
    --repo /path/to/code \
    --prompt "Analyze the codebase architecture and identify design patterns"

# Dependency analysis
python3 packages/opencode_analysis/agent.py \
    --repo /path/to/code \
    --prompt "Find all places where user input flows to database queries"
```

**Advantages of Standalone Package**:
- ✅ No modifications to existing packages
- ✅ Clean separation of concerns
- ✅ Easy to test and maintain
- ✅ Can evolve independently
- ✅ Demonstrates OpenCode capabilities clearly
- ✅ Future packages can follow this pattern

---

## Configuration

### Core Configuration (`core/config.py`)

Add to `RaptorConfig` class following existing patterns:

```python
class RaptorConfig:
    # ... existing config ...
    
    # OpenCode Configuration
    OPENCODE_SERVER_URL = os.getenv("OPENCODE_SERVER_URL", "http://localhost:8080")
    OPENCODE_ENABLE_LSP = os.getenv("OPENCODE_ENABLE_LSP", "true").lower() == "true"
    OPENCODE_LSP_LANGUAGES = [
        "python", "javascript", "typescript", "java", 
        "go", "rust", "cpp", "c", "ruby", "php"
    ]
    OPENCODE_CACHE_ENABLED = True
    OPENCODE_CACHE_DIR = BASE_OUT_DIR / "opencode_cache"
    OPENCODE_TIMEOUT = 30  # seconds for LSP operations
    OPENCODE_MAX_WORKERS = 2  # parallel LSP server connections
    OPENCODE_SERVER_BINARY = os.getenv("OPENCODE_SERVER_BINARY")  # Custom path if needed
    OPENCODE_STARTUP_TIMEOUT = 30  # seconds to wait for server startup
```

### LLM Config Enhancement (`core/llm/config.py`)

Add to `LLMConfig` dataclass:

```python
@dataclass
class LLMConfig:
    # ... existing fields ...
    
    # OpenCode integration
    enable_opencode: bool = True
    opencode_tool_permissions: Dict[str, str] = field(default_factory=lambda: {
        "read": "allow",
        "write": "deny",  # Security: prevent accidental writes
        "bash": "approval",  # Require approval for execution
        "grep": "allow",
        "definitions": "allow",
        "references": "allow",
        "call_hierarchy": "allow",
        "semantic_search": "allow",
    })
```

### Environment Variables

Users can configure via environment variables (consistent with existing patterns):

```bash
# OpenCode server URL
export OPENCODE_SERVER_URL="http://localhost:8080"

# Enable/disable LSP
export OPENCODE_ENABLE_LSP="true"

# Comma-separated list of languages for LSP
export OPENCODE_LSP_LANGUAGES="python,javascript,typescript"

# Disable OpenCode entirely
export OPENCODE_ENABLED="false"
```

---

## Migration Strategy

### Package Requirements

**New Package**:
- `packages/opencode_analysis/` - **REQUIRES OpenCode**
  - Uses OpenCode for all codebase interactions
  - Uses LiteLLM for LLM analysis
  - Fails with clear error if OpenCode unavailable

**Existing Packages**:
- All existing packages remain unchanged
- No modifications to `llm_analysis`, `codeql`, `static-analysis`, etc.
- Clean separation - new package demonstrates OpenCode capabilities

### Error Handling Strategy

The `opencode_analysis` package:
- Uses `create_opencode_client(repo_path, required=True)`
- Raises `OpenCodeUnavailableError` with clear installation instructions
- No fallback - package fails fast with helpful error message
- Error includes installation steps and verification commands

### Migration Steps

1. **Phase 1**: Core infrastructure (no package changes)
   - Install OpenCode Python client
   - Create `core/opencode/` module
   - Add configuration
   - Add error classes (`OpenCodeUnavailableError`, `OpenCodeConnectionError`)
   - Add server management (`OpenCodeServerManager`)
   - Existing packages remain unchanged

2. **Phase 2**: LLM integration
   - Add `create_opencode_client()` to `LLMClient` with `required` parameter
   - Add `OpenCodeToolProvider` for tool-based LLM interactions
   - Add `generate_with_codebase()` method
   - No breaking changes to existing code

3. **Phase 3**: New package creation
   - Create `packages/opencode_analysis/` directory structure
   - Implement `OpenCodeAnalyzer` class
   - Implement CLI (`agent.py`)
   - Add to `raptor.py` launcher
   - **No changes to existing packages**

4. **Phase 4**: Testing and documentation
   - Comprehensive tests for new package
   - Integration tests with various repositories
   - Documentation and usage examples
   - Update main RAPTOR docs

5. **Phase 5**: Future enhancements (optional)
   - Additional analysis modes
   - More sophisticated prompt templates
   - Integration patterns for other packages (if desired)

---

## Testing Strategy

### Unit Tests

1. **OpenCode Client**:
   - File operations
   - LSP methods
   - Error handling

2. **Tool Provider**:
   - Tool registration
   - Function calling conversion
   - Response handling

### Integration Tests

1. **End-to-End**:
   - Full workflow with OpenCode
   - Multi-language repositories
   - LSP server lifecycle

2. **Package Tests**:
   - New `opencode_analysis` package
   - Comparison with manual analysis methods
   - Performance benchmarks

### Performance Tests

1. **LSP Response Times**:
   - Measure LSP query latency
   - Cache effectiveness
   - Multi-language performance

2. **Tool Calling Overhead**:
   - Compare with/without tool calling
   - Multi-turn conversation performance

---

## Security Considerations

1. **Tool Permissions**:
   - Default deny for write operations
   - Approval required for bash execution
   - Audit logging for all tool usage

2. **LSP Server Security**:
   - Isolated execution environment
   - Resource limits
   - Network restrictions

3. **Code Access**:
   - Repository path validation
   - No access outside repository
   - Sandboxed execution

---

## Success Metrics

1. **Code Understanding**:
   - Improved context extraction accuracy
   - Better dataflow path validation
   - Reduced false positives

2. **Performance**:
   - LSP query latency < 100ms (cached)
   - Tool calling overhead < 20%
   - Overall analysis time improvement

3. **Developer Experience**:
   - Easier codebase navigation
   - Better code intelligence
   - Reduced manual code reading

---

## Risks and Mitigations

### Risks

1. **LSP Server Reliability**:
   - **Risk**: LSP servers may crash or be unavailable
   - **Mitigation**: Retry logic, connection health checks, clear error messages for required packages

2. **Performance Overhead**:
   - **Risk**: LSP queries add latency
   - **Mitigation**: Aggressive caching, async queries, parallel execution

3. **Complexity**:
   - **Risk**: Additional complexity in codebase
   - **Mitigation**: Clean abstraction, comprehensive tests, documentation

4. **Dependency Management**:
   - **Risk**: OpenCode and LSP servers require maintenance
   - **Mitigation**: Version pinning, dependency updates, fallback options

---

## Conclusion

This integration plan provides a comprehensive roadmap for integrating OpenCode with LiteLLM in RAPTOR while maintaining the framework's core architectural principles. The phased approach allows for gradual adoption while maintaining backward compatibility. The benefits include:

- **Semantic Code Understanding**: LSP integration provides real code intelligence
- **Better Context Extraction**: Intelligent boundaries instead of fixed line ranges
- **Enhanced Analysis**: Reference tracking, call hierarchies, semantic search
- **Tool-Based Interactions**: LLMs can interact with codebases directly via tools
- **Maintained Flexibility**: LiteLLM continues to provide multi-provider support
- **Reusable Core**: OpenCode becomes part of shared infrastructure
- **User-Friendly**: Transparent integration with clear opt-in/opt-out controls

The integration maintains RAPTOR's modular architecture while significantly enhancing codebase interaction capabilities, following established patterns for extensibility and user experience.

**Important Note**: Packages that require OpenCode (`llm_analysis`, `codeql/autonomous_analyzer`) will fail with clear error messages if OpenCode is not installed. This ensures users get the full benefit of semantic code intelligence rather than silently falling back to less capable manual methods.
