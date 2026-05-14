#!/usr/bin/env bash
# Create a local virtual environment and install Python dependencies.
#
# Usage:
#   bash scripts/setup.sh
#
# Environment variables:
#   VENV_DIR  Path to the venv directory (default: .venv)
#   PYBIN     Python executable to use   (default: python)
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
PYBIN="${PYBIN:-python}"

if [ ! -d "$VENV_DIR" ]; then
  echo "[1/3] Creating venv at $VENV_DIR"
  "$PYBIN" -m venv "$VENV_DIR"
else
  echo "[1/3] Reusing existing venv at $VENV_DIR"
fi

# Pick the right pip binary for the platform
if [ -x "$VENV_DIR/bin/pip" ]; then
  PIP="$VENV_DIR/bin/pip"
elif [ -x "$VENV_DIR/Scripts/pip.exe" ]; then
  PIP="$VENV_DIR/Scripts/pip.exe"
else
  echo "ERROR: could not locate pip in $VENV_DIR" >&2
  exit 1
fi

echo "[2/3] Upgrading pip"
"$PIP" install --upgrade pip

echo "[3/3] Installing requirements.txt"
"$PIP" install -r requirements.txt

echo ""
echo "Done. Activate the venv with:"
echo "  Linux/macOS: source $VENV_DIR/bin/activate"
echo "  Windows:     $VENV_DIR\\Scripts\\activate"
