"""Tests for DataflowValidator.extract_dataflow_from_sarif.

These tests verify that the typed-dataclass output produced by the codeql
validator matches the behaviour of the old inline implementation that was
replaced with a delegation to core.sarif.parser.extract_dataflow_path.

Edge cases cover the implementation differences:
  - Core returns a dict with key "file"; DataflowStep uses field "file_path"
  - Core puts intermediate steps in "steps"; DataflowPath uses "intermediate_steps"
  - Sanitizer detection ("sanitiz"/"validat" in label) is applied after conversion
  - Exactly-2-location flows → intermediate_steps=[], not missing
"""

import sys
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from packages.codeql.dataflow_validator import (
    DataflowValidator,
    DataflowPath,
    DataflowStep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loc(file: str = "src/foo.py", line: int = 10, col: int = 5,
              snippet: str = "x = input()", label: str = "") -> Dict[str, Any]:
    return {
        "location": {
            "physicalLocation": {
                "artifactLocation": {"uri": file},
                "region": {
                    "startLine": line,
                    "startColumn": col,
                    "snippet": {"text": snippet},
                },
            },
            "message": {"text": label},
        }
    }


def _make_sarif(locations, rule_id: str = "py/sql-injection",
                message: str = "Taint flows") -> Dict[str, Any]:
    return {
        "ruleId": rule_id,
        "message": {"text": message},
        "codeFlows": [{"threadFlows": [{"locations": locations}]}],
    }


def _validator():
    return DataflowValidator(llm_client=MagicMock())


# ---------------------------------------------------------------------------
# Guard conditions — must return None
# ---------------------------------------------------------------------------

class TestGuardConditions:

    def test_no_code_flows_returns_none(self):
        result = _validator().extract_dataflow_from_sarif({"codeFlows": []})
        assert result is None

    def test_missing_code_flows_key_returns_none(self):
        result = _validator().extract_dataflow_from_sarif({})
        assert result is None

    def test_empty_thread_flows_returns_none(self):
        sarif = {"codeFlows": [{"threadFlows": []}]}
        result = _validator().extract_dataflow_from_sarif(sarif)
        assert result is None

    def test_single_location_returns_none(self):
        """Need at least source + sink (2 locations)."""
        sarif = _make_sarif([_make_loc()])
        result = _validator().extract_dataflow_from_sarif(sarif)
        assert result is None

    def test_empty_locations_returns_none(self):
        sarif = _make_sarif([])
        result = _validator().extract_dataflow_from_sarif(sarif)
        assert result is None


# ---------------------------------------------------------------------------
# Minimal 2-location flow (source + sink, no intermediate steps)
# ---------------------------------------------------------------------------

class TestTwoLocationFlow:

    def setup_method(self):
        locs = [
            _make_loc(file="src/input.py", line=1, label="user input"),
            _make_loc(file="src/db.py", line=99, label="sql sink"),
        ]
        self.result = _validator().extract_dataflow_from_sarif(_make_sarif(locs))

    def test_returns_dataflow_path(self):
        assert isinstance(self.result, DataflowPath)

    def test_source_is_first_location(self):
        assert self.result.source.file_path == "src/input.py"
        assert self.result.source.line == 1

    def test_sink_is_last_location(self):
        assert self.result.sink.file_path == "src/db.py"
        assert self.result.sink.line == 99

    def test_no_intermediate_steps(self):
        """Exactly 2 locations → empty intermediate_steps, not None."""
        assert self.result.intermediate_steps == []

    def test_no_sanitizers_detected(self):
        assert self.result.sanitizers == []


# ---------------------------------------------------------------------------
# Field name translation: core dict "file" → DataflowStep.file_path
# ---------------------------------------------------------------------------

class TestFieldTranslation:
    """Core returns {"file": ...} but DataflowStep stores it as file_path.
    This was the main structural difference between the two implementations."""

    def test_file_key_mapped_to_file_path(self, tmp_path):
        locs = [
            _make_loc(file="a/source.py"),
            _make_loc(file="b/sink.py"),
        ]
        result = _validator().extract_dataflow_from_sarif(_make_sarif(locs))
        assert result.source.file_path == "a/source.py"
        assert result.sink.file_path == "b/sink.py"

    def test_line_and_column_preserved(self):
        locs = [
            _make_loc(line=7, col=3),
            _make_loc(line=42, col=11),
        ]
        result = _validator().extract_dataflow_from_sarif(_make_sarif(locs))
        assert result.source.line == 7
        assert result.source.column == 3
        assert result.sink.line == 42
        assert result.sink.column == 11

    def test_snippet_preserved(self):
        locs = [
            _make_loc(snippet="req.GET['q']"),
            _make_loc(snippet="cursor.execute(query)"),
        ]
        result = _validator().extract_dataflow_from_sarif(_make_sarif(locs))
        assert result.source.snippet == "req.GET['q']"
        assert result.sink.snippet == "cursor.execute(query)"

    def test_label_preserved(self):
        locs = [
            _make_loc(label="tainted source"),
            _make_loc(label="dangerous sink"),
        ]
        result = _validator().extract_dataflow_from_sarif(_make_sarif(locs))
        assert result.source.label == "tainted source"
        assert result.sink.label == "dangerous sink"


# ---------------------------------------------------------------------------
# Intermediate steps: core "steps" list → DataflowPath.intermediate_steps
# ---------------------------------------------------------------------------

class TestIntermediateSteps:

    def test_three_location_flow_has_one_intermediate(self):
        locs = [
            _make_loc(file="a.py", line=1, label="src"),
            _make_loc(file="b.py", line=5, label="pass-through"),
            _make_loc(file="c.py", line=9, label="sink"),
        ]
        result = _validator().extract_dataflow_from_sarif(_make_sarif(locs))
        assert len(result.intermediate_steps) == 1
        assert result.intermediate_steps[0].file_path == "b.py"
        assert result.intermediate_steps[0].line == 5

    def test_five_location_flow_has_three_intermediates(self):
        locs = [_make_loc(file=f"f{i}.py", line=i) for i in range(5)]
        result = _validator().extract_dataflow_from_sarif(_make_sarif(locs))
        assert len(result.intermediate_steps) == 3
        # Source and sink are not in intermediate
        intermediate_files = [s.file_path for s in result.intermediate_steps]
        assert "f0.py" not in intermediate_files
        assert "f4.py" not in intermediate_files

    def test_source_and_sink_not_in_intermediate(self):
        locs = [
            _make_loc(file="source.py", line=1),
            _make_loc(file="middle.py", line=5),
            _make_loc(file="sink.py", line=9),
        ]
        result = _validator().extract_dataflow_from_sarif(_make_sarif(locs))
        intermediate_files = [s.file_path for s in result.intermediate_steps]
        assert "source.py" not in intermediate_files
        assert "sink.py" not in intermediate_files


# ---------------------------------------------------------------------------
# Sanitizer detection (codeql-specific logic applied after conversion)
# ---------------------------------------------------------------------------

class TestSanitizerDetection:

    def test_sanitize_label_detected(self):
        locs = [
            _make_loc(label="user input"),
            _make_loc(label="sanitize(value)"),
            _make_loc(label="sql sink"),
        ]
        result = _validator().extract_dataflow_from_sarif(_make_sarif(locs))
        assert len(result.sanitizers) == 1
        assert "sanitize(value)" in result.sanitizers

    def test_validate_label_detected(self):
        locs = [
            _make_loc(label="source"),
            _make_loc(label="validate_input(x)"),
            _make_loc(label="sink"),
        ]
        result = _validator().extract_dataflow_from_sarif(_make_sarif(locs))
        assert "validate_input(x)" in result.sanitizers

    def test_sanitizer_case_insensitive(self):
        locs = [
            _make_loc(label="source"),
            _make_loc(label="SANITIZER_CHECK"),
            _make_loc(label="sink"),
        ]
        result = _validator().extract_dataflow_from_sarif(_make_sarif(locs))
        assert len(result.sanitizers) == 1

    def test_non_sanitizer_intermediate_not_detected(self):
        locs = [
            _make_loc(label="source"),
            _make_loc(label="assign x = y"),   # no sanitiz/validat substring
            _make_loc(label="sink"),
        ]
        result = _validator().extract_dataflow_from_sarif(_make_sarif(locs))
        assert result.sanitizers == []

    def test_sanitizer_on_source_or_sink_not_detected(self):
        """Sanitizer detection only applies to intermediate steps."""
        locs = [
            _make_loc(label="sanitize source"),   # source — should NOT count
            _make_loc(label="validate sink"),     # sink — should NOT count
        ]
        result = _validator().extract_dataflow_from_sarif(_make_sarif(locs))
        assert result.sanitizers == []

    def test_multiple_sanitizers_all_detected(self):
        locs = [
            _make_loc(label="source"),
            _make_loc(label="sanitize_html(x)"),
            _make_loc(label="validate_length(x)"),
            _make_loc(label="sink"),
        ]
        result = _validator().extract_dataflow_from_sarif(_make_sarif(locs))
        assert len(result.sanitizers) == 2


# ---------------------------------------------------------------------------
# Top-level metadata: rule_id and message
# ---------------------------------------------------------------------------

class TestMetadata:

    def test_rule_id_preserved(self):
        locs = [_make_loc(), _make_loc()]
        sarif = _make_sarif(locs, rule_id="java/sqli")
        result = _validator().extract_dataflow_from_sarif(sarif)
        assert result.rule_id == "java/sqli"

    def test_message_text_preserved(self):
        locs = [_make_loc(), _make_loc()]
        sarif = _make_sarif(locs, message="Taint flows to SQL sink")
        result = _validator().extract_dataflow_from_sarif(sarif)
        assert result.message == "Taint flows to SQL sink"

    def test_missing_rule_id_defaults_to_empty(self):
        sarif = _make_sarif([_make_loc(), _make_loc()])
        del sarif["ruleId"]
        result = _validator().extract_dataflow_from_sarif(sarif)
        assert result.rule_id == ""

    def test_missing_message_defaults_to_empty(self):
        sarif = _make_sarif([_make_loc(), _make_loc()])
        del sarif["message"]
        result = _validator().extract_dataflow_from_sarif(sarif)
        assert result.message == ""


# ---------------------------------------------------------------------------
# Missing / partial SARIF fields — must not crash, must default gracefully
# ---------------------------------------------------------------------------

class TestMissingFields:

    def test_missing_snippet_defaults_to_empty_string(self):
        loc = {"location": {
            "physicalLocation": {
                "artifactLocation": {"uri": "f.py"},
                "region": {"startLine": 1, "startColumn": 1},
                # no "snippet" key
            },
            "message": {"text": ""},
        }}
        sarif = {"codeFlows": [{"threadFlows": [{"locations": [loc, loc]}]}]}
        result = _validator().extract_dataflow_from_sarif(sarif)
        assert result.source.snippet == ""

    def test_missing_uri_defaults_to_empty_string(self):
        loc = {"location": {
            "physicalLocation": {
                "artifactLocation": {},   # no "uri"
                "region": {"startLine": 5, "startColumn": 1},
            },
            "message": {"text": ""},
        }}
        sarif = {"codeFlows": [{"threadFlows": [{"locations": [loc, loc]}]}]}
        result = _validator().extract_dataflow_from_sarif(sarif)
        assert result.source.file_path == ""

    def test_missing_line_defaults_to_zero(self):
        loc = {"location": {
            "physicalLocation": {
                "artifactLocation": {"uri": "f.py"},
                "region": {},   # no startLine
            },
            "message": {"text": ""},
        }}
        sarif = {"codeFlows": [{"threadFlows": [{"locations": [loc, loc]}]}]}
        result = _validator().extract_dataflow_from_sarif(sarif)
        assert result.source.line == 0
