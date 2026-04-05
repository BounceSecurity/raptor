"""Tests for core.source.read_code_context."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.source import read_code_context


class TestReadCodeContext:
    def test_highlights_target_line(self, tmp_path):
        f = tmp_path / "src.py"
        f.write_text("line1\nline2\nline3\n")
        result = read_code_context(str(f), line=2, context_lines=1)
        assert ">>>" in result
        # Only the target line should have the marker
        lines = result.splitlines()
        marked = [l for l in lines if ">>>" in l]
        assert len(marked) == 1
        assert "line2" in marked[0]

    def test_includes_context_lines(self, tmp_path):
        f = tmp_path / "src.py"
        f.write_text("\n".join(f"L{i}" for i in range(1, 11)) + "\n")
        result = read_code_context(str(f), line=5, context_lines=2)
        assert "L3" in result
        assert "L4" in result
        assert "L5" in result
        assert "L6" in result
        assert "L7" in result
        # Lines outside the window should not appear
        assert "L1" not in result
        assert "L9" not in result

    def test_clamps_to_file_start(self, tmp_path):
        f = tmp_path / "src.py"
        f.write_text("first\nsecond\nthird\n")
        result = read_code_context(str(f), line=1, context_lines=10)
        assert "first" in result
        assert "second" in result

    def test_clamps_to_file_end(self, tmp_path):
        f = tmp_path / "src.py"
        f.write_text("a\nb\nc\n")
        result = read_code_context(str(f), line=3, context_lines=10)
        assert "a" in result
        assert "c" in result

    def test_returns_empty_string_on_missing_file(self):
        result = read_code_context("/nonexistent/path/file.py", line=1)
        assert result == ""

    def test_accepts_path_object(self, tmp_path):
        f = tmp_path / "src.py"
        f.write_text("hello\n")
        result = read_code_context(f, line=1)
        assert "hello" in result

    def test_line_numbers_in_output(self, tmp_path):
        f = tmp_path / "src.py"
        f.write_text("x\ny\nz\n")
        result = read_code_context(str(f), line=2, context_lines=1)
        # All output lines should contain a line number
        for line in result.splitlines():
            assert "|" in line
