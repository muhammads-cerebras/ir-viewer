#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

UV_BIN="$ROOT_DIR/.uv/bin/uv"
if [[ ! -x "$UV_BIN" ]]; then
  echo "uv not found in workspace. Installing..."
  mkdir -p "$ROOT_DIR/.uv"
  curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$ROOT_DIR/.uv" sh
else
  echo "Using workspace uv at $UV_BIN"
fi
export PATH="$ROOT_DIR/.uv/bin:$PATH"

if [[ ! -d ".venv" ]]; then
  echo "Setting up environment (uv sync)..."
  uv sync
fi

IR_PATH="${1:-$ROOT_DIR/ir.mlir}"
shift || true

uv run ir-viewer "$IR_PATH" "$@"
