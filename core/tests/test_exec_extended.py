"""Tests for the stdin/input extensions added to core.exec.run.

Each test group corresponds to a real caller:
  - binary_analysis/debugger.py  passes stdin=<file object>
  - llm_analysis/cc_dispatch.py  passes input=<string>
  - codeql/database_manager.py   plain run (no stdin/input)
  - static-analysis/codeql/env   plain run; combines stdout+stderr

Edge cases covered:
  - Mutual exclusion of stdin and input
  - stdin accepts binary file object (text=True still applies to stdout/stderr)
  - input string written to stdin of child process
  - Return tuple (rc, stdout, stderr) preserved for all call patterns
"""

import sys
import tempfile
from pathlib import Path
from io import BytesIO

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.exec import run


class TestPlainRun:
    """Original behaviour must be unchanged."""

    def test_returns_three_tuple(self):
        rc, stdout, stderr = run(["python3", "-c", "print('hi')"])
        assert isinstance(rc, int)
        assert isinstance(stdout, str)
        assert isinstance(stderr, str)

    def test_success_returncode(self):
        rc, _, _ = run(["python3", "-c", "import sys; sys.exit(0)"])
        assert rc == 0

    def test_failure_returncode(self):
        rc, _, _ = run(["python3", "-c", "import sys; sys.exit(42)"])
        assert rc == 42

    def test_stdout_captured(self):
        _, stdout, _ = run(["python3", "-c", "print('hello world')"])
        assert stdout.strip() == "hello world"

    def test_stderr_captured(self):
        _, _, stderr = run(["python3", "-c", "import sys; sys.stderr.write('err\n')"])
        assert "err" in stderr


class TestInputParam:
    """Corresponds to cc_dispatch.py: subprocess.run(cmd, input=prompt, ...)"""

    def test_string_passed_to_stdin(self):
        """Child reads from stdin; we pass the string via input=."""
        _, stdout, _ = run(
            ["python3", "-c", "import sys; print(sys.stdin.read().strip())"],
            input="hello from input",
        )
        assert stdout.strip() == "hello from input"

    def test_empty_input_string(self):
        _, stdout, _ = run(
            ["python3", "-c", "import sys; data=sys.stdin.read(); print(len(data))"],
            input="",
        )
        assert stdout.strip() == "0"

    def test_multiline_input(self):
        _, stdout, _ = run(
            ["python3", "-c", "import sys; lines=sys.stdin.readlines(); print(len(lines))"],
            input="line1\nline2\nline3\n",
        )
        assert stdout.strip() == "3"

    def test_return_code_still_correct_with_input(self):
        rc, _, _ = run(["python3", "-c", "import sys; sys.stdin.read(); sys.exit(0)"], input="x")
        assert rc == 0


class TestStdinParam:
    """Corresponds to debugger.py: subprocess.run(cmd, stdin=f, ...) where f is a binary file."""

    def test_file_object_passed_as_stdin(self, tmp_path):
        """Child reads from stdin which is a file object (binary mode)."""
        f = tmp_path / "input.txt"
        f.write_bytes(b"data from file\n")
        with open(f, "rb") as fh:
            _, stdout, _ = run(
                ["python3", "-c", "import sys; print(sys.stdin.read().strip())"],
                stdin=fh,
            )
        assert stdout.strip() == "data from file"

    def test_binary_file_with_text_output(self, tmp_path):
        """stdin is binary; stdout is still decoded as text (text=True in run)."""
        f = tmp_path / "nums.bin"
        f.write_bytes(b"42\n")
        with open(f, "rb") as fh:
            _, stdout, _ = run(
                ["python3", "-c", "import sys; x=int(sys.stdin.read()); print(x*2)"],
                stdin=fh,
            )
        assert stdout.strip() == "84"

    def test_return_code_correct_with_stdin_file(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_bytes(b"ok")
        with open(f, "rb") as fh:
            rc, _, _ = run(
                ["python3", "-c", "import sys; sys.stdin.read(); sys.exit(0)"],
                stdin=fh,
            )
        assert rc == 0


class TestMutualExclusion:
    """stdin and input cannot be used together — mirrors subprocess.run behaviour."""

    def test_both_raises_value_error(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_bytes(b"")
        with open(f, "rb") as fh:
            with pytest.raises(ValueError, match="mutually exclusive"):
                run(["python3", "--version"], stdin=fh, input="text")


class TestEnvAndMergePattern:
    """Corresponds to static-analysis/codeql/env.py which merged stdout+stderr.

    After migration: rc, stdout, stderr = run(...) then output = stdout + stderr
    """

    def test_stdout_stderr_separate_then_combined(self, tmp_path):
        """Caller combines the two streams to replicate stderr=subprocess.STDOUT."""
        script = tmp_path / "script.py"
        script.write_text("import sys\nsys.stdout.write('out\\n')\nsys.stderr.write('err\\n')\n")
        rc, stdout, stderr = run(["python3", str(script)])
        combined = (stdout + stderr).strip()
        assert "out" in combined
        assert "err" in combined

    def test_version_command_output_reachable(self):
        """python3 --version writes to stdout on 3.4+; stderr on older.
        Combining both guarantees we always get it."""
        rc, stdout, stderr = run(["python3", "--version"])
        combined = (stdout + stderr).strip()
        assert "Python" in combined
        assert rc == 0
