#!/usr/bin/env python3
"""
OpenCode Analysis Agent

Thin CLI wrapper around core OpenCode analysis functionality.
All analysis logic is in core/opencode/ for reuse by other packages.
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for core imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.logging import get_logger
from core.llm.client import LLMClient
from core.llm.config import LLMConfig
from core.opencode import analyze_codebase, OpenCodeUnavailableError

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

  # Custom model
  python3 packages/opencode_analysis/agent.py \\
    --repo /path/to/code \\
    --prompt "Review code quality" \\
    --model claude-3-5-sonnet-20241022
        """
    )
    
    parser.add_argument("--repo", required=True, type=Path,
                       help="Path to repository to analyze")
    parser.add_argument("--prompt", required=True,
                       help="Analysis prompt/question")
    parser.add_argument("--out", type=Path,
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
    
    if not repo_path.is_dir():
        print(f"ERROR: Repository path is not a directory: {repo_path}")
        sys.exit(1)
    
    # Initialize LLM client
    llm_config = LLMConfig()
    if args.model:
        # Override model if specified
        llm_config.primary_model.model_name = args.model
    
    try:
        llm_client = LLMClient(llm_config)
    except Exception as e:
        print(f"ERROR: Failed to initialize LLM client: {e}")
        sys.exit(1)
    
    # Run analysis using core module
    try:
        result = analyze_codebase(
            repo_path=repo_path,
            prompt=args.prompt,
            llm_client=llm_client,
            output_dir=args.out,
            opencode_server_url=args.opencode_server
        )
        
        # Print results
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nResult saved to: {result.output_dir}")
        print(f"Files analyzed: {result.files_analyzed}")
        print(f"Languages: {', '.join(result.languages) if result.languages else 'Unknown'}")
        print(f"LLM calls: {result.llm_calls}")
        print(f"Cost: ${result.cost:.4f}")
        print(f"\nAnalysis output:")
        print("-" * 70)
        print(result.response)
        print("-" * 70)
        
    except OpenCodeUnavailableError as e:
        print(f"ERROR: {e}")
        print("\nInstallation steps:")
        print("1. RAPTOR will auto-install OpenCode server on first use")
        print("2. Or set OPENCODE_AUTO_INSTALL=false and install manually")
        print("3. See docs/OPENCODE_INTEGRATION.md for details")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
