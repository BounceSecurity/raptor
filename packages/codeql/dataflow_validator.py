#!/usr/bin/env python3
"""
CodeQL Dataflow Validator

Validates CodeQL dataflow findings using LLM analysis to determine
if dataflow paths are truly exploitable beyond theoretical detection.
"""

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.logging import get_logger
from core.source import read_code_context
from core.sarif.parser import extract_dataflow_path as _extract_dataflow_path

logger = get_logger()


@dataclass
class DataflowStep:
    """A single step in a dataflow path."""
    file_path: str
    line: int
    column: int
    snippet: str
    label: str  # e.g., "source", "step", "sink"


@dataclass
class DataflowPath:
    """Complete dataflow path from source to sink."""
    source: DataflowStep
    sink: DataflowStep
    intermediate_steps: List[DataflowStep]
    sanitizers: List[str]
    rule_id: str
    message: str


@dataclass
class DataflowValidation:
    """Result of dataflow validation."""
    is_exploitable: bool
    confidence: float  # 0.0-1.0
    sanitizers_effective: bool
    bypass_possible: bool
    bypass_strategy: Optional[str]
    attack_complexity: str  # "low", "medium", "high"
    reasoning: str
    barriers: List[str]
    prerequisites: List[str]


# Dict schema for LLM structured generation (consistent with other callers)
DATAFLOW_VALIDATION_SCHEMA = {
    "is_exploitable": "boolean",
    "confidence": "float (0.0-1.0)",
    "sanitizers_effective": "boolean",
    "bypass_possible": "boolean",
    "bypass_strategy": "string - strategy to bypass sanitizers, or empty if none",
    "attack_complexity": "string (low/medium/high)",
    "reasoning": "string",
    "barriers": "list of strings",
    "prerequisites": "list of strings",
}


class DataflowValidator:
    """
    Validate CodeQL dataflow findings using LLM analysis.

    Goes beyond CodeQL's static detection to determine:
    - Are sanitizers truly effective?
    - Are there hidden barriers?
    - Is the path reachable in practice?
    - What's the real attack complexity?
    """

    def __init__(self, llm_client):
        """
        Initialize dataflow validator.

        Args:
            llm_client: LLM client from packages/llm_analysis/llm/client.py
        """
        self.llm = llm_client
        self.logger = get_logger()

    def extract_dataflow_from_sarif(self, result: Dict) -> Optional[DataflowPath]:
        """Extract dataflow path from SARIF result, returning a typed DataflowPath.

        Delegates SARIF parsing to core.sarif.parser.extract_dataflow_path and
        converts the generic dict result into the typed DataflowPath/DataflowStep
        dataclasses used by this validator.  The sanitizer-detection heuristic
        (labels containing "sanitiz" or "validat") is applied here because it is
        specific to CodeQL validation logic.
        """
        try:
            raw = _extract_dataflow_path(result.get("codeFlows", []))
            if raw is None:
                return None

            def _to_step(d: Dict) -> DataflowStep:
                return DataflowStep(
                    file_path=d["file"],
                    line=d["line"],
                    column=d["column"],
                    snippet=d["snippet"],
                    label=d["label"],
                )

            source = _to_step(raw["source"])
            sink = _to_step(raw["sink"])
            intermediate = [_to_step(s) for s in raw["steps"]]

            sanitizers = [
                s.label for s in intermediate
                if "sanitiz" in s.label.lower() or "validat" in s.label.lower()
            ]

            return DataflowPath(
                source=source,
                sink=sink,
                intermediate_steps=intermediate,
                sanitizers=sanitizers,
                rule_id=result.get("ruleId", ""),
                message=result.get("message", {}).get("text", ""),
            )

        except Exception as e:
            self.logger.warning(f"Failed to extract dataflow path: {e}")
            return None

    def read_source_context(self, file_path: str, line: int, context_lines: int = 10) -> str:
        """Read source code context around a location."""
        return read_code_context(file_path, line, context_lines)

    def validate_dataflow_path(
        self,
        dataflow: DataflowPath,
        repo_path: Path
    ) -> DataflowValidation:
        """
        Validate dataflow path exploitability using LLM.

        Args:
            dataflow: DataflowPath object
            repo_path: Repository root path

        Returns:
            DataflowValidation result
        """
        self.logger.info(f"Validating dataflow path: {dataflow.rule_id}")

        # Read source context for key locations
        source_context = self.read_source_context(
            str(repo_path / dataflow.source.file_path),
            dataflow.source.line
        )

        sink_context = self.read_source_context(
            str(repo_path / dataflow.sink.file_path),
            dataflow.sink.line
        )

        # Build dataflow path summary
        path_summary = []
        path_summary.append(f"SOURCE: {dataflow.source.label}")
        path_summary.append(f"  {dataflow.source.file_path}:{dataflow.source.line}")

        for i, step in enumerate(dataflow.intermediate_steps, 1):
            path_summary.append(f"STEP {i}: {step.label}")
            path_summary.append(f"  {step.file_path}:{step.line}")

        path_summary.append(f"SINK: {dataflow.sink.label}")
        path_summary.append(f"  {dataflow.sink.file_path}:{dataflow.sink.line}")

        # Create validation prompt
        prompt = f"""You are a security researcher analyzing a potential vulnerability detected by CodeQL.

VULNERABILITY: {dataflow.message}
RULE: {dataflow.rule_id}

DATAFLOW PATH:
{chr(10).join(path_summary)}

SOURCE LOCATION:
File: {dataflow.source.file_path}
Line: {dataflow.source.line}

{source_context}

SINK LOCATION:
File: {dataflow.sink.file_path}
Line: {dataflow.sink.line}

{sink_context}

SANITIZERS DETECTED: {', '.join(dataflow.sanitizers) if dataflow.sanitizers else 'None'}

Analyze this dataflow path and determine:

1. **Exploitability**: Can an attacker actually control data flowing from source to sink?
2. **Sanitization**: Are there effective sanitizers in the path? Can they be bypassed?
3. **Reachability**: Is this path reachable in real execution scenarios?
4. **Attack Complexity**: How difficult is exploitation?
5. **Bypass Strategy**: If there are barriers, how can they be bypassed?
6. **Prerequisites**: What conditions must be met for successful exploitation?

Respond in JSON format:
{{
    "is_exploitable": boolean,
    "confidence": float (0.0-1.0),
    "sanitizers_effective": boolean,
    "bypass_possible": boolean,
    "bypass_strategy": string or null,
    "attack_complexity": "low" | "medium" | "high",
    "reasoning": string,
    "barriers": [list of strings],
    "prerequisites": [list of strings]
}}
"""

        try:
            # Use LLM to analyze
            response_dict, _ = self.llm.generate_structured(
                prompt=prompt,
                schema=DATAFLOW_VALIDATION_SCHEMA,
                system_prompt="You are an expert security researcher analyzing dataflow vulnerabilities."
            )

            # Parse response
            validation = DataflowValidation(**response_dict)

            self.logger.info(
                f"Dataflow validation: exploitable={validation.is_exploitable}, "
                f"confidence={validation.confidence:.2f}"
            )

            return validation

        except Exception as e:
            self.logger.error(f"Dataflow validation failed: {e}")

            # Return conservative default
            return DataflowValidation(
                is_exploitable=False,
                confidence=0.0,
                sanitizers_effective=True,
                bypass_possible=False,
                bypass_strategy=None,
                attack_complexity="high",
                reasoning=f"Validation failed: {str(e)}",
                barriers=["Analysis failed"],
                prerequisites=[]
            )

    def validate_finding(
        self,
        sarif_result: Dict,
        repo_path: Path
    ) -> Optional[DataflowValidation]:
        """
        Validate a SARIF finding if it contains dataflow.

        Args:
            sarif_result: SARIF result object
            repo_path: Repository root path

        Returns:
            DataflowValidation or None if not a dataflow finding
        """
        # Extract dataflow path
        dataflow = self.extract_dataflow_from_sarif(sarif_result)

        if not dataflow:
            self.logger.debug("Not a dataflow finding, skipping validation")
            return None

        # Validate the path
        return self.validate_dataflow_path(dataflow, repo_path)


def main():
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate CodeQL dataflow findings")
    parser.add_argument("--sarif", required=True, help="SARIF file")
    parser.add_argument("--repo", required=True, help="Repository path")
    parser.add_argument("--finding-index", type=int, default=0, help="Finding index to validate")
    args = parser.parse_args()

    # Load SARIF
    from core.sarif.parser import load_sarif
    sarif = load_sarif(Path(args.sarif))
    if not sarif:
        return

    runs = sarif.get("runs", [])
    if not runs:
        print("No runs in SARIF file")
        return
    results = runs[0].get("results", [])
    if args.finding_index >= len(results):
        print(f"Finding index {args.finding_index} out of range (0-{len(results)-1})")
        return

    finding = results[args.finding_index]

    # Initialize validator (would need LLM client in real usage)
    # validator = DataflowValidator(llm_client)
    # validation = validator.validate_finding(finding, Path(args.repo))

    print(f"Dataflow validation would analyze finding:")
    print(f"  Rule: {finding.get('ruleId')}")
    print(f"  Message: {finding.get('message', {}).get('text')}")
    print(f"  Has dataflow: {bool(finding.get('codeFlows'))}")


if __name__ == "__main__":
    main()
