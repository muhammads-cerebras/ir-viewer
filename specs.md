IR Viewer Specification
=======================

Date
----
February 6, 2026

Purpose
-------
Define the requirements, tooling, setup, and features for an IR viewer focused on the example MLIR file in this workspace. This document will be the single source of truth before implementation begins. The initial UI will be terminal-based using Textual, with a future web UI that reuses the same core logic.

Goals
-----
- Provide a fast, readable, and navigable view of MLIR files.
- Support exploring large IR files with minimal latency.
- Offer helpful UX for developers inspecting IR transformations.
- Keep the initial scope small, modular, and extensible.
- Ensure core logic is UI-agnostic to enable a future web UI.

Non-Goals
---------
- Implement a full MLIR parser or verifier.
- Provide a full IDE or compiler pipeline integration in v1.
- Support non-MLIR IR formats in v1 (possible future work).

Target Users
------------
- Compiler engineers inspecting MLIR.
- Developers learning or debugging MLIR passes.

Inputs and Outputs
------------------
Inputs
~~~~~~
- Primary input: MLIR text files (starting with the example [ir.mlir](ir.mlir)).
- Optional metadata: none in v1.

Outputs
~~~~~~~
- Rendered, interactive IR view in a local web UI.
- Optional export: plain text copy of selected regions (v1 optional).

Core Features (MVP)
-------------------
1) File loading
	- Load a local MLIR file from the workspace.
	- Display file name and size.

2) Syntax highlighting
	- Highlight MLIR keywords, types, attributes, SSA values, blocks, and comments.
	- Preserve whitespace and indentation.

3) Structure navigation
	- Collapsible regions: modules, functions, blocks, and ops (heuristic-based in v1).
	- Outline/sidebar of top-level entities.

4) Search and jump
	- Text search with highlights.
	- Jump to next/previous match.

5) Line numbers
	- Show line numbers alongside IR.
	- Maintain alignment with content.

6) Copy
	- Copy selected text or a folded region.

7) Performance baseline
	- Handle files up to 5 MB smoothly.

8) Contextual detail panel (navigation aid)
	- Selecting an instruction shows:
	  - Expanded layout metadata (resolved from #layout references).
	  - Expanded location info (resolved from #loc references).
	  - Attributes rendered in a human-friendly form.

Nice-to-Have (Post-MVP)
------------------------
- Semantic navigation using a real MLIR parser.
- Call graph or use-def navigation.
- Diff mode between two IR files.
- Annotations and comments.
- Export to HTML or PDF.

UX Requirements
---------------
- Clear monospace layout, readable at 12–16px.
- Light and dark themes.
- Collapsible sections with explicit affordances.
- Keyboard shortcuts for search and folding.
- A focused detail panel for the selected instruction.
- Nested panels that mirror the IR hierarchy (module → func → blocks).

IR Structure Observations (from example)
----------------------------------------
- Large top-level layout metadata blocks (#layout0, #layout1, ...).
- Many SSA instructions using layout/loc references rather than inline data.
- The `ws_rt` section uses a non-SSA handle/buffer form: `<handle> = <inst> <result_buf>, <in1_buf>, <in2_buf> ...`.
- Some `ws_rt` instructions (e.g., barriers) have no output handle and only take handles as inputs.
- Nested regions (e.g., thread/loop constructs) with deep indentation.
- Heavy cross-referencing makes raw IR hard to navigate.

Navigation-Focused Requirements (Agreed)
----------------------------------------
1) Layout resolution on selection
	 - When an instruction is selected, resolve `layout = #layoutX` and show the
		 full layout dictionary.
	 - Layouts are per-argument; display a layout subsection per operand/result.

2) Location resolution on selection
	 - When an instruction is selected, resolve `loc(#locX)` and show the
		 expanded location data.

3) Attribute rendering
	 - Render attributes in a natural, readable form (grouped, labeled, and
		 parsed).
	 - Map source-vars to the output(s) they name.
	 - Map argument types onto operands/results to avoid disjoint type lists.

4) Alloc/free as annotations
	 - Treat `alloc`/`free` as metadata rather than executable commands.
	 - Surface allocation metadata inline at the sites where the buffer/handle
		 is used.
	 - Provide a toggle to hide/show alloc/free instructions in the main view.

5) Command prefix shortening
	 - If all commands within a block share the same prefix (e.g., `ws_rt.cmdh.`),
		 display the shortened command name by default.
	 - Provide a toggle to show full command names.

Display Rules (Instruction Level)
---------------------------------
- Layout and location metadata are not shown inline; they are displayed in the
	details panel when an instruction is selected.
- The main instruction line is minimized to segments:
	- `<results> = <inst> <operands>` (or `<inst> <operands>` when no results).
- All attributes are shown in the side panel.
- The user can navigate between segments (results, inst, operands) for quick
	focus and copy.

Architecture Overview
---------------------
The project will be Textual-first for the terminal UI. Core functionality will live in a UI-agnostic library/module (parser, model, search, folding, navigation state). The terminal UI will be a thin layer over the core. A future web UI can reuse the same core library to avoid duplication. Initial parsing can be heuristic-based (regex and indentation), with a future path to a real MLIR grammar.

Recommended Tech Stack (Initial Proposal)
-----------------------------------------
Terminal UI (v1)
~~~~~~~~~~~~~~~
- Python
- Textual

Project Tooling
~~~~~~~~~~~~~~~
- Git for version control
- uv for Python environment and dependency management

Core Library (Shared)
~~~~~~~~~~~~~~~~~~~~~
- Python package/modules (UI-agnostic)

Future Web UI (post-v1)
~~~~~~~~~~~~~~~~~~~~~~~
- TypeScript
- React
- Vite
- CodeMirror 6 (for editor-like rendering and folding)

Project Setup
-------------
1) Initialize a Python project with Textual.
2) Create a core package for parsing, model, search, and folding (UI-agnostic).
3) Implement a Textual UI that consumes the core package.
4) Load [ir.mlir](ir.mlir) by default when available.

Implementation Plan (High Level)
--------------------------------
Phase 1: Core + Textual shell
- Core model: document, regions, tokens.
- Basic Textual layout: header, sidebar outline, main viewer.
- Load file content into viewer.

Phase 2: Highlighting and folding
- Tokenizer for MLIR-ish syntax.
- Folding by heuristics (modules, funcs, blocks).

Phase 3: Search and navigation
- Search box, highlight results, jump controls.

Phase 4: Polish
- Theme toggle, keyboard shortcuts, copy actions.

Open Questions
--------------
- Do we need VS Code extension integration later?
- What exact folding rules should be considered correct for MLIR?
- Are there any specific MLIR dialects to support beyond generic syntax?

Acceptance Criteria (MVP)
-------------------------
- Load and display [ir.mlir](ir.mlir) with visible syntax highlighting.
- Collapsible regions for modules and functions at minimum.
- Search works with next/previous navigation.
- UI remains responsive with 5 MB file.
- Documentation updated with setup and usage steps.

Security and Privacy
--------------------
- Local-only processing.
- No network access required for viewing local files.

Risks
-----
- Heuristic folding may mis-handle complex MLIR constructs.
- Very large files could exceed browser memory limits.

Future Extensions
-----------------
- MLIR-aware parser for accurate structure.
- Pass pipeline visualization.
- Integrate with build systems to auto-refresh on file change.

Change Log
----------
- 2026-02-06: Initial draft.
- 2026-02-06: Decision: Textual-first terminal UI with a UI-agnostic core to enable a future web UI.
- 2026-02-06: Added navigation-focused requirements (layout/loc/attribute resolution on selection).
- 2026-02-06: Treat ws_rt alloc/free as annotations and allow hiding them.
- 2026-02-06: Add block-level command prefix shortening with toggle.
- 2026-02-06: Use git for version control and uv for a self-contained Python project.
- 2026-02-06: Refined display rules (nested panels, minimized instruction line, attrs in side panel).
