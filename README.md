IR Viewer (Textual)
====================

Overview
--------
Terminal-first IR viewer for MLIR with navigation aids (layout/loc resolution, alloc metadata, and command shortening).

Requirements
------------
- bash
- curl

Quick Start
-----------
Run only the launcher script. It will:
- download micromamba locally under `.micromamba-bin` (if missing)
- create/reuse a project-local micromamba env
- install Python + NumPy + uv from conda-forge
- install/update this project in that env (editable mode)
- launch the app

Examples:
- `./ir-viewer`
- `./ir-viewer ir.mlir --tensor-dump merge`

Notes
-----
- If no file is provided, the app attempts to open ir.mlir in the current workspace.
- Toggle keys are shown in the app footer.
- Override env location with `IR_VIEWER_MAMBA_ENV_PATH`.

Tensor dump on older Linux (self-contained SDK)
-----------------------------------------------
If your OS glibc is older, prebuilt NumPy wheels may fail to import (for example missing `GLIBC_2.27`).

Use the launcher script for a strict self-contained runtime:

- Python is launched through SDK `ld-linux` (SDK glibc), not system glibc.
- Runtime virtualenv is created from SDK Python at `.runtime/venv`.
- NumPy is always rebuilt from source in tensor-dump mode (`--no-binary numpy`).
- If NumPy cannot import under the SDK runtime, launcher exits with an error.

Project-local runtime layout (default):

- `.runtime/sdk`
- `.runtime/sysroot` (optional)

Bootstrap (one command):

- `./scripts/bootstrap-runtime.sh --clean`

Optional custom source paths:

- `./scripts/bootstrap-runtime.sh --sdk /path/to/sdk --sysroot /path/to/sysroot --clean`

- `IR_VIEWER_SDK_ROOT` (default: `.runtime/sdk`)
- `IR_VIEWER_SYSROOT` (optional sysroot path)
- `IR_VIEWER_GCC` (defaults to `$IR_VIEWER_SDK_ROOT/bin/gcc`)
- `IR_VIEWER_GXX` (defaults to `$IR_VIEWER_SDK_ROOT/bin/g++`)
- `IR_VIEWER_LD_LINUX` (optional override for SDK dynamic loader)

Behavior:

- On each run, NumPy is checked first under the self-contained runtime.
- Rebuild/install happens only if import fails.

Example:

- `IR_VIEWER_SDK_ROOT=/path/to/sdk IR_VIEWER_SYSROOT=/path/to/sysroot ./ir-viewer ir.mlir --tensor-dump merge`
