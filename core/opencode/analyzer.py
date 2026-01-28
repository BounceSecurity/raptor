#!/usr/bin/env python3
"""
OpenCode Codebase Analyzer

Core analysis functionality combining OpenCode codebase intelligence with LLM analysis.
This is a reusable core module that can be used by any package.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

from core.config import RaptorConfig
from core.logging import get_logger
from core.llm.client import LLMClient
from .client_sdk import OpenCodeClient
from .exceptions import OpenCodeUnavailableError

logger = get_logger()


@dataclass
class CodebaseAnalysisResult:
    """Result of codebase analysis."""
    output_dir: Path
    prompt: str
    response: str
    files_analyzed: int
    llm_calls: int
    cost: float
    opencode_operations: List[Dict[str, Any]]
    languages: List[str]


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
        RuntimeError: For other analysis errors
    """
    repo_path = Path(repo_path).resolve()
    
    # Create output directory
    if not output_dir:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = RaptorConfig.get_out_dir() / f"opencode_analysis_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting codebase analysis: {prompt}")
    
    # Initialize OpenCode client (required)
    try:
        opencode = llm_client.create_opencode_client(repo_path, required=True)
    except OpenCodeUnavailableError as e:
        raise OpenCodeUnavailableError(
            f"OpenCode is required for codebase analysis: {e}. "
            "Please install and start OpenCode server. "
            "See docs/OPENCODE_INTEGRATION.md for installation instructions."
        ) from e
    
    # Gather codebase context
    opencode_operations = []
    try:
        logger.info("Listing files in repository (this may take a while for large repos)...")
        files = opencode.list_files()
        logger.info(f"Found {len(files)} files in repository")
        opencode_operations.append({"operation": "list_files", "count": len(files)})
        
        logger.info("Detecting programming languages...")
        # Use the file list we already have instead of scanning again
        languages = opencode.detect_languages_from_files(files)
        logger.info(f"Detected languages: {', '.join(languages)}")
        opencode_operations.append({"operation": "detect_languages", "languages": languages})
        
        codebase_context = {
            "files_count": len(files),
            "languages": languages,
            "structure": _get_repository_structure(files[:100])  # Limit to first 100 files
        }
    except Exception as e:
        logger.warning(f"Failed to gather full codebase context: {e}")
        codebase_context = {
            "files_count": 0,
            "languages": [],
            "structure": {}
        }
    
    # Send prompt directly to OpenCode via HTTP SDK
    # OpenCode will handle the analysis using its built-in tools and LLM
    logger.info("Sending analysis prompt to OpenCode via HTTP SDK...")
    try:
        opencode_response = opencode.send_message(prompt)
        
        # Log the OpenCode response in debug mode
        logger.debug(f"OpenCode raw response ({len(opencode_response)} chars):\n{opencode_response}")
        
        # Use LLM to potentially enhance/format the response if needed
        # For now, use OpenCode's response directly
        response_content = opencode_response
    except Exception as e:
        logger.error(f"Failed to get response from OpenCode: {e}")
        # Fallback to LLM-only analysis
        system_prompt = _build_system_prompt(repo_path, codebase_context)
        response = llm_client.generate(prompt, system_prompt)
        response_content = response.content
        opencode_response = ""
    
    # Create a mock LLMResponse for compatibility
    class MockResponse:
        def __init__(self, content):
            self.content = content
            self.model = "opencode"
            self.tokens_used = 0
            self.cost = 0.0
    
    response = MockResponse(response_content)
    
    # Save results
    result_file = output_dir / "analysis_result.txt"
    result_file.write_text(response.content)
    
    # Save metadata
    metadata = {
        "prompt": prompt,
        "repo_path": str(repo_path),
        "model": response.model,
        "tokens_used": response.tokens_used,
        "cost": response.cost,
        "files_analyzed": codebase_context.get("files_count", 0),
        "languages": codebase_context.get("languages", []),
        "opencode_operations": len(opencode_operations),
    }
    
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    
    return CodebaseAnalysisResult(
        output_dir=output_dir,
        prompt=prompt,
        response=response.content,
        files_analyzed=codebase_context.get("files_count", 0),
        llm_calls=1,
        cost=response.cost,
        opencode_operations=opencode_operations,
        languages=codebase_context.get("languages", [])
    )


def _get_repository_structure(files: List[str]) -> Dict[str, Any]:
    """Get repository structure from file list."""
    structure = {}
    for file_path in files:
        parts = Path(file_path).parts
        if len(parts) > 1:
            dir_name = parts[0]
            if dir_name not in structure:
                structure[dir_name] = []
            structure[dir_name].append("/".join(parts[1:]))
        else:
            if "root" not in structure:
                structure["root"] = []
            structure["root"].append(file_path)
    
    return structure


def _build_system_prompt(repo_path: Path, context: Dict[str, Any]) -> str:
    """Build system prompt with codebase context."""
    languages_str = ", ".join(context.get("languages", [])) if context.get("languages") else "Unknown"
    files_count = context.get("files_count", 0)
    
    return f"""You are a security analyst analyzing a codebase.

Codebase Context:
- Repository: {repo_path}
- Files: {files_count} files
- Languages: {languages_str}

You have access to OpenCode tools for codebase interaction:
- read_file: Read file contents
- find_definitions: Find symbol definitions
- find_references: Find all references to a symbol
- get_call_hierarchy: Understand function call relationships
- semantic_search: Search code semantically
- grep: Pattern-based search

Use these tools to understand the codebase and provide thorough analysis."""
