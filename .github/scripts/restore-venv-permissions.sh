#!/bin/bash
# Restore permissions for virtual environment after artifact download
# This is necessary because GitHub Actions artifacts don't preserve file permissions

set -euo pipefail

VENV_PATH="${1:-.venv}"

if [ ! -d "$VENV_PATH" ]; then
  echo "Error: Virtual environment not found at $VENV_PATH"
  exit 1
fi

echo "Restoring permissions for virtual environment at $VENV_PATH"

# Restore execute permissions for all binaries in bin/
if [ -d "$VENV_PATH/bin" ]; then
  chmod -R +x "$VENV_PATH/bin"/* 2>/dev/null || true
fi

# Fix permissions for osemgrep binaries (semgrep's Rust backend)
# Semgrep uses osemgrep as its core engine, which needs execute permissions
echo "Fixing permissions for semgrep/osemgrep binaries..."

# Make all files in semgrep package executable (osemgrep is a binary in there)
find "$VENV_PATH/lib/python"*/site-packages -type d -name "semgrep*" 2>/dev/null | while read -r dir; do
  find "$dir" -type f -exec chmod +x {} \; 2>/dev/null || true
done

# Also search for osemgrep specifically and make it executable
find "$VENV_PATH" -type f -name "*osemgrep*" -exec chmod +x {} \; 2>/dev/null || true

echo "Permission restoration complete"
