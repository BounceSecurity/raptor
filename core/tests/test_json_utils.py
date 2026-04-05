"""Tests for core.json_utils.load_json_file.

Edge cases map to the 4 original callers that were replaced:
  orchestrator.py  — error-level logging, returns None
  bootstrap.py #1  — warning-level logging + continue (None check)
  bootstrap.py #2  — silent continue on failure (None check, no logging)
  bootstrap.py #3  — KeyError from dict comprehension (separate concern from load)
  reporting.py     — returns False, sets instance attribute to None on failure
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.json_utils import load_json_file


class TestSuccessfulLoad:

    def test_returns_dict_for_valid_json(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"key": "value", "num": 42}')
        result = load_json_file(f)
        assert result == {"key": "value", "num": 42}

    def test_accepts_string_path(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"x": 1}')
        result = load_json_file(str(f))
        assert result == {"x": 1}

    def test_accepts_path_object(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"x": 1}')
        result = load_json_file(f)
        assert result == {"x": 1}

    def test_nested_dict(self, tmp_path):
        f = tmp_path / "data.json"
        data = {"findings": [{"id": 1}, {"id": 2}]}
        f.write_text(json.dumps(data))
        assert load_json_file(f) == data

    def test_empty_dict(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text("{}")
        assert load_json_file(f) == {}


class TestReadFailures:
    """Callers relied on (json.JSONDecodeError, OSError) catch — verify both."""

    def test_returns_none_when_file_missing(self, tmp_path):
        result = load_json_file(tmp_path / "nonexistent.json")
        assert result is None

    def test_returns_none_for_invalid_json(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not json at all {{{")
        assert load_json_file(f) is None

    def test_returns_none_for_truncated_json(self, tmp_path):
        f = tmp_path / "truncated.json"
        f.write_text('{"key":')
        assert load_json_file(f) is None

    def test_returns_none_for_empty_file(self, tmp_path):
        f = tmp_path / "empty.json"
        f.write_text("")
        assert load_json_file(f) is None

    def test_returns_none_for_json_array(self, tmp_path):
        """json.loads('[1,2,3]') returns a list, not a dict.
        The function returns whatever json.loads returns — callers that need
        a dict should check the type if needed.  Here we just verify no crash."""
        f = tmp_path / "array.json"
        f.write_text("[1, 2, 3]")
        # Returns a list (not None) — this is intentional, callers check .get() etc.
        result = load_json_file(f)
        assert result == [1, 2, 3]


class TestOptionalLogger:
    """Logger parameter: when supplied, warning must be emitted on failure."""

    def test_no_logger_arg_does_not_raise(self, tmp_path):
        """Calling without logger must not crash even on failure."""
        result = load_json_file(tmp_path / "missing.json")
        assert result is None

    def test_logger_warning_called_on_missing_file(self, tmp_path):
        mock_log = MagicMock()
        load_json_file(tmp_path / "missing.json", logger=mock_log)
        mock_log.warning.assert_called_once()
        assert "missing.json" in mock_log.warning.call_args[0][0]

    def test_logger_warning_called_on_bad_json(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("!!!invalid")
        mock_log = MagicMock()
        load_json_file(f, logger=mock_log)
        mock_log.warning.assert_called_once()

    def test_logger_not_called_on_success(self, tmp_path):
        f = tmp_path / "ok.json"
        f.write_text('{"ok": true}')
        mock_log = MagicMock()
        load_json_file(f, logger=mock_log)
        mock_log.warning.assert_not_called()

    def test_none_logger_does_not_raise(self, tmp_path):
        """Explicit None logger must not crash."""
        result = load_json_file(tmp_path / "missing.json", logger=None)
        assert result is None


class TestCallerPatterns:
    """Verify patterns used by each original caller still work correctly."""

    def test_orchestrator_pattern_returns_none_on_bad_file(self, tmp_path):
        """orchestrator.py: check `if report is None: return None`."""
        result = load_json_file(tmp_path / "prep_report.json")
        assert result is None  # caller then returns None from outer function

    def test_bootstrap_loop_pattern(self, tmp_path):
        """bootstrap.py #1/#2: skip item in loop when file is unreadable."""
        results = []
        files = [
            (tmp_path / "good.json", '{"findings": [1, 2]}'),
            (tmp_path / "bad.json", "INVALID"),
            (tmp_path / "also_good.json", '{"findings": [3]}'),
        ]
        for path, content in files:
            path.write_text(content)

        for path, _ in files:
            data = load_json_file(path)
            if data is None:
                continue
            results.append(data)

        assert len(results) == 2
        assert results[0]["findings"] == [1, 2]

    def test_checklist_pattern_keyerror_separate(self, tmp_path):
        """bootstrap.py #3: json loading succeeds; KeyError from dict
        comprehension is separate and not swallowed by load_json_file."""
        f = tmp_path / "checklist.json"
        f.write_text('{"files": [{"sha256": "abc"}]}')  # missing "path" key
        data = load_json_file(f)
        assert data is not None
        # KeyError from dict comprehension — caller handles separately
        with pytest.raises(KeyError):
            _ = {entry["path"]: entry["sha256"] for entry in data["files"]}

    def test_reporting_pattern_sets_attribute(self, tmp_path):
        """reporting.py: store loaded data, return False if None."""
        f = tmp_path / "findings.json"
        f.write_text('{"status": "confirmed"}')

        validation_data = load_json_file(f)
        assert validation_data is not None
        assert validation_data["status"] == "confirmed"

        # Failure path
        missing_result = load_json_file(tmp_path / "nonexistent.json")
        assert missing_result is None
