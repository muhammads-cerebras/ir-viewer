IR Viewer (Textual)
====================

Overview
--------
Terminal-first IR viewer for MLIR with navigation aids (layout/loc resolution, alloc metadata, and command shortening).

Requirements
------------
- Python 3.11+
- uv

Quick Start
-----------
1) Install dependencies:
   - uv sync

2) Run the app:
   - uv run ir-viewer /cb/home/muhammads/ws/projects/ir-viewer/ir.mlir

Notes
-----
- If no file is provided, the app attempts to open ir.mlir in the current workspace.
- Toggle keys are shown in the app footer.
