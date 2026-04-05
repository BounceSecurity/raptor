"""Tests for core.output.make_run_dir."""

import os
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.output import make_run_dir
from core.config import RaptorConfig


class TestMakeRunDir:
    def test_creates_directory(self, tmp_path):
        with patch.dict(os.environ, {"RAPTOR_OUT_DIR": str(tmp_path)}):
            out = make_run_dir("test")
        assert out.exists()
        assert out.is_dir()

    def test_prefix_in_name(self, tmp_path):
        with patch.dict(os.environ, {"RAPTOR_OUT_DIR": str(tmp_path)}):
            out = make_run_dir("myscan")
        assert out.name.startswith("myscan_")

    def test_suffix_in_name(self, tmp_path):
        with patch.dict(os.environ, {"RAPTOR_OUT_DIR": str(tmp_path)}):
            out = make_run_dir("scan", "myrepo")
        assert "myrepo" in out.name
        assert out.name.startswith("scan_myrepo_")

    def test_no_suffix_omitted_cleanly(self, tmp_path):
        with patch.dict(os.environ, {"RAPTOR_OUT_DIR": str(tmp_path)}):
            out = make_run_dir("fuzz")
        # Should not have double underscores from a missing suffix
        assert "__" not in out.name

    def test_timestamp_in_name(self, tmp_path):
        with patch.dict(os.environ, {"RAPTOR_OUT_DIR": str(tmp_path)}):
            out = make_run_dir("run")
        # Timestamp portion is digits and underscores, e.g. 20240101_120000
        parts = out.name.split("_")
        assert len(parts) >= 3  # prefix + date + time

    def test_idempotent_mkdir(self, tmp_path):
        with patch.dict(os.environ, {"RAPTOR_OUT_DIR": str(tmp_path)}):
            out = make_run_dir("idempotent")
        # Calling mkdir again on an existing dir should not raise
        out.mkdir(parents=True, exist_ok=True)

    def test_respects_raptor_out_dir(self, tmp_path):
        custom = tmp_path / "custom_out"
        custom.mkdir()
        with patch.dict(os.environ, {"RAPTOR_OUT_DIR": str(custom)}):
            out = make_run_dir("check")
        assert str(out).startswith(str(custom))
