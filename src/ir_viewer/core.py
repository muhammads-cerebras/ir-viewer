from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional

from rich.text import Text


_LAYOUT_RE = re.compile(r"^\s*(#layout\d+)\s*=\s*(.*)$")
_LOC_RE = re.compile(r"^\s*(#loc\d+)\s*=\s*(.*)$")
_ALLOC_RE = re.compile(r"^\s*(%\d+)\s*=\s*ws_rt\.cmdh\.waf\.alloc\b(.*)$")
_LAYOUT_REF_RE = re.compile(r"layout\s*=\s*(#layout\d+)")
_LOC_REF_RE = re.compile(r"loc\((#loc\d+)\)")
_SSA_RE = re.compile(r"%\d+")


@dataclass(frozen=True)
class LayoutDef:
    name: str
    text: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class LocDef:
    name: str
    text: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class AllocInfo:
    value: str
    text: str
    line_number: int
    summary: str
    alloc_type: str
    bytes: Optional[str]
    addr: Optional[str]
    alignment: Optional[str]
    extras: List[str]
    parent: Optional[str]


@dataclass(frozen=True)
class RenderedLine:
    source_index: int
    display: str


@dataclass(frozen=True)
class Instruction:
    source_index: int
    raw: str
    results: str
    inst: str
    operands: str
    attrs: Optional[str]
    layout_ref: Optional[str]
    loc_ref: Optional[str]
    arg_types: List[str]
    result_types: List[str]
    uniform_type: Optional[str]
    source_vars: List[str]
    # Semaphore/num_rx relationships
    semaphore_unblocker: Optional[int] = None  # Line index of instruction that unblocks this via semaphore
    num_rx_unblocker: Optional[int] = None  # Line index of instruction that unblocks this via num_rx
    semaphore_consumers: List[int] = None  # Line indices of instructions unblocked by this via semaphore
    num_rx_consumers: List[int] = None  # Line indices of instructions unblocked by this via num_rx

    def __post_init__(self):
        if self.semaphore_consumers is None:
            object.__setattr__(self, 'semaphore_consumers', [])
        if self.num_rx_consumers is None:
            object.__setattr__(self, 'num_rx_consumers', [])


@dataclass
class Node:
    label: str
    kind: str
    source_index: Optional[int]
    children: List["Node"]
    attrs: Optional[str] = None
    value: Optional[str] = None
    layout_index: Optional[int] = None


@dataclass(frozen=True)
class NodeData:
    kind: str
    source_index: Optional[int]
    attrs: Optional[str]
    value: Optional[str]
    layout_index: Optional[int]


@dataclass
class Document:
    path: Path
    lines: List[str]
    layouts: Dict[str, LayoutDef]
    locs: Dict[str, LocDef]
    allocs: Dict[str, List[AllocInfo]]

    @classmethod
    def from_path(cls, path: Path) -> "Document":
        lines = path.read_text(encoding="utf-8").splitlines()
        layouts = _collect_defs(lines, _LAYOUT_RE, LayoutDef)
        locs = _collect_defs(lines, _LOC_RE, LocDef)
        allocs = _collect_allocs(lines)
        return cls(path=path, lines=lines, layouts=layouts, locs=locs, allocs=allocs)


@dataclass
class RenderOptions:
    show_alloc_free: bool = False
    show_full_prefix: bool = False
    shorten_prefix: str = "ws_rt.cmdh."
    show_line_numbers: bool = False
    show_alloc_sizes: bool = False
    show_source_vars: bool = True
    show_full_source_vars: bool = False
    show_left_types: bool = True
    align_left_panel: bool = False
    show_left_loc: bool = True
    wrap_left_panel: bool = True
    emacs_mode: bool = False
    group_loc_prefixes: bool = False
    show_only_tx_rx: bool = False


class DocumentView:
    def __init__(self, document: Document, options: Optional[RenderOptions] = None) -> None:
        self.document = document
        self.options = options or RenderOptions()
        self.rendered: List[RenderedLine] = []
        self.instructions: Dict[int, Instruction] = {}
        self.ws_rt_io: Dict[int, tuple[List[str], List[str]]] = {}
        self.source_var_suffixes: Dict[int, List[str]] = {}

    def render(self) -> List[RenderedLine]:
        self._ensure_ws_rt_io()
        self._ensure_source_var_suffixes()
        rendered: List[RenderedLine] = []
        for idx, line in enumerate(self.document.lines):
            if _is_layout_or_loc_def(line):
                continue
            if not self.options.show_alloc_free and _is_alloc_or_free(line):
                continue
            display = _instruction_display(
                line,
                idx,
                self.options,
                self.instructions,
                self.document.allocs,
                self.ws_rt_io,
                self.source_var_suffixes.get(idx),
                self.options.show_full_source_vars,
            )
            if self.options.show_line_numbers:
                display = f"{idx + 1:5d} | {display}"
            rendered.append(RenderedLine(source_index=idx, display=display))
        self.rendered = rendered
        return rendered

    def source_line_for_rendered_index(self, rendered_index: int) -> Optional[int]:
        if rendered_index < 0 or rendered_index >= len(self.rendered):
            return None
        return self.rendered[rendered_index].source_index

    def details_for_node(self, node: NodeData, segment_index: int = 0) -> Text:
        if node.source_index is None:
            return Text("")
        self._ensure_ws_rt_io()
        self._ensure_source_var_suffixes()
        line = self.document.lines[node.source_index]
        instruction = self.instructions.get(node.source_index)
        layout_text = _resolve_ref(line, _LAYOUT_REF_RE, self.document.layouts)
        if not layout_text:
            layout_text = _extract_inline_layout(line)
        loc_text = _extract_loc_text(_resolve_ref(line, _LOC_REF_RE, self.document.locs))
        attrs_source = instruction.attrs if instruction and instruction.attrs else node.attrs
        attrs_lines = _format_attrs(attrs_source) if attrs_source else []

        parts: List[str] = []
        parts.append(f"Line {node.source_index + 1}")
        parts.append("")
        if instruction:
            parts.append(
                _instruction_label(
                    instruction,
                    self.options,
                    self.document.allocs,
                    node.source_index,
                    self.options.show_alloc_sizes,
                    self.options.show_left_types,
                    self.ws_rt_io,
                    None,
                    self.options.show_source_vars,
                    False,
                )
            )
        else:
            parts.append(line)

        if instruction:
            if instruction.inst.startswith("ws_rt."):
                res_buffers, arg_buffers = self.ws_rt_io.get(node.source_index, ([], []))
                res_types = instruction.arg_types[: len(res_buffers)]
                arg_types = instruction.arg_types[len(res_buffers) :]
                if instruction.uniform_type:
                    if len(res_types) < len(res_buffers):
                        res_types = res_types + [instruction.uniform_type] * (len(res_buffers) - len(res_types))
                    if len(arg_types) < len(arg_buffers):
                        arg_types = arg_types + [instruction.uniform_type] * (len(arg_buffers) - len(arg_types))
            else:
                _, res_buffers, arg_buffers = _instruction_buffers(instruction)
                res_types = instruction.result_types
                arg_types = instruction.arg_types
            if loc_text:
                parts.append("")
                parts.append("Location")
                parts.append(loc_text)
            alloc_lines = _format_alloc_table(
                res_buffers + arg_buffers,
                res_types + arg_types,
                self.document.allocs,
                node.source_index,
            )
            if alloc_lines:
                parts.append("")
                parts.append("Allocations")
                parts.extend(alloc_lines)
            layout_lines = _format_layout_section(
                res_buffers,
                arg_buffers,
                layout_text,
            )
            if layout_lines:
                parts.append("")
                parts.append("Layouts")
                parts.extend(layout_lines)

        if attrs_lines:
            parts.append("")
            parts.append("Attributes")
            parts.extend(attrs_lines)

        return _highlight_details("\n".join(parts))
        layout_text = _resolve_ref(line, _LAYOUT_REF_RE, self.document.layouts)
        if not layout_text:
            layout_text = _extract_inline_layout(line)
        loc_text = _extract_loc_text(_resolve_ref(line, _LOC_REF_RE, self.document.locs))
        attrs_source = instruction.attrs if instruction and instruction.attrs else node.attrs
        attrs_lines = _format_attrs(attrs_source) if attrs_source else []

        parts: List[str] = []
        parts.append(f"Line {node.source_index + 1}")
        parts.append("" )
        if instruction:
            parts.append(
                _instruction_label(
                    instruction,
                    self.options,
                    self.document.allocs,
                    node.source_index,
                    self.options.show_alloc_sizes,
                    self.options.show_left_types,
                    self.ws_rt_io,
                    None,
                    self.options.show_source_vars,
                    False,
                )
            )
        else:
            parts.append(line)

        if instruction:
            if instruction.inst.startswith("ws_rt."):
                res_buffers, arg_buffers = self.ws_rt_io.get(node.source_index, ([], []))
                res_types = instruction.arg_types[: len(res_buffers)]
                arg_types = instruction.arg_types[len(res_buffers) :]
                if instruction.uniform_type:
                    if len(res_types) < len(res_buffers):
                        res_types = res_types + [instruction.uniform_type] * (len(res_buffers) - len(res_types))
                    if len(arg_types) < len(arg_buffers):
                        arg_types = arg_types + [instruction.uniform_type] * (len(arg_buffers) - len(arg_types))
            else:
                _, res_buffers, arg_buffers = _instruction_buffers(instruction)
                res_types = instruction.result_types
                arg_types = instruction.arg_types
            alloc_lines = _format_alloc_table(
                res_buffers + arg_buffers,
                res_types + arg_types,
                self.document.allocs,
                node.source_index,
            )
            if alloc_lines:
                parts.append("")
                parts.append("Allocations")
                parts.extend(alloc_lines)
            layout_lines = _format_layout_section(
                res_buffers,
                arg_buffers,
                layout_text,
            )
            if layout_lines:
                parts.append("")
                parts.append("Layouts")
                parts.extend(layout_lines)

        if attrs_lines:
            parts.append("")
            parts.append("Attributes")
            parts.extend(attrs_lines)

        if loc_text:
            parts.append("")
            parts.append("Location")
            parts.append(loc_text)

        return _highlight_details("\n".join(parts))

    def build_hierarchy(self) -> Node:
        self._ensure_ws_rt_io()
        self._ensure_source_var_suffixes()
        root = Node(label=self.document.path.name, kind="root", source_index=None, children=[])
        stack: List[tuple[Node, int]] = [(root, 0)]
        depth = 0
        idx = 0
        skip_attrs_until = -1
        while idx < len(self.document.lines):
            line = self.document.lines[idx]
            if idx <= skip_attrs_until:
                idx += 1
                continue
            if _is_layout_or_loc_def(line):
                depth += _brace_delta(line)
                idx += 1
                continue
            if not self.options.show_alloc_free and _is_alloc_or_free(line):
                depth += _brace_delta(line)
                idx += 1
                continue
            if _is_region_start(line):
                label = _region_label(line, self.options)
                attrs, end_idx = _collect_region_attrs(self.document.lines, idx)
                if end_idx is not None:
                    skip_attrs_until = end_idx
                node = Node(label=label, kind="region", source_index=idx, children=[], attrs=attrs)
                stack[-1][0].children.append(node)
                depth_after = depth + max(1, _brace_delta(line))
                stack.append((node, depth_after))
            elif _is_instruction_line(line):
                label = _instruction_display(
                    line,
                    idx,
                    self.options,
                    self.instructions,
                    self.document.allocs,
                    self.ws_rt_io,
                    self.source_var_suffixes.get(idx),
                    self.options.show_full_source_vars,
                )
                node = Node(label=label, kind="inst", source_index=idx, children=[], attrs=None)
                stack[-1][0].children.append(node)
                # results/args details are shown in the right panel

            depth += _brace_delta(line)
            while len(stack) > 1 and depth < stack[-1][1]:
                stack.pop()
            idx += 1
        _prune_container_children(root)
        return root

    def _ensure_ws_rt_io(self) -> None:
        if self.ws_rt_io:
            return
        self.ws_rt_io = _compute_ws_rt_io(self.document.lines, self.document.allocs)

    def _ensure_source_var_suffixes(self) -> None:
        if self.source_var_suffixes:
            return
        if not self.instructions:
            for idx, line in enumerate(self.document.lines):
                instruction = _parse_instruction_line(line, idx)
                if instruction:
                    self.instructions[idx] = instruction
        self.source_var_suffixes = _compute_source_var_suffixes(self.document.lines, self.instructions)

    def _ensure_semaphore_relationships(self) -> None:
        """Compute semaphore and num_rx producer-consumer relationships."""
        if not self.instructions:
            for idx, line in enumerate(self.document.lines):
                instruction = _parse_instruction_line(line, idx)
                if instruction:
                    self.instructions[idx] = instruction
        
        # Check if relationships are already computed
        for inst in self.instructions.values():
            if inst.semaphore_unblocker is not None or inst.num_rx_unblocker is not None:
                return
            if inst.semaphore_consumers or inst.num_rx_consumers:
                return
            break
        
        # Compute relationships
        _compute_semaphore_relationships(self.instructions)


def _prune_container_children(node: Node) -> None:
    for child in node.children:
        _prune_container_children(child)
    while len(node.children) == 1:
        child = node.children[0]
        if child.kind == "region" and child.attrs is None:
            node.children = child.children
        else:
            break


def _collect_defs(lines: List[str], pattern: re.Pattern[str], cls_type):
    defs = {}
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        match = pattern.match(line)
        if not match:
            idx += 1
            continue
        name = match.group(1)
        text, end_idx = _collect_braced_definition(lines, idx)
        defs[name] = cls_type(name=name, text=text, start_line=idx, end_line=end_idx)
        idx = end_idx + 1
    return defs


def _collect_allocs(lines: List[str]) -> Dict[str, List[AllocInfo]]:
    allocs: Dict[str, List[AllocInfo]] = {}
    for idx, line in enumerate(lines):
        match = _ALLOC_RE.match(line)
        if not match:
            continue
        value = match.group(1)
        ssa_values = _SSA_RE.findall(line)
        parent = ssa_values[1] if len(ssa_values) > 1 else None
        alloc_type, pe_bytes, addr, alignment, extras = _extract_alloc_fields(line)
        summary = _summarize_alloc(alloc_type, pe_bytes, addr, alignment, extras)
        allocs.setdefault(value, []).append(
            AllocInfo(
                value=value,
                text=line.strip(),
                line_number=idx + 1,
                summary=summary,
                alloc_type=alloc_type,
                bytes=pe_bytes,
                addr=addr,
                alignment=alignment,
                extras=extras,
                parent=parent,
            )
        )
    for value in allocs:
        allocs[value].sort(key=lambda info: info.line_number)
    return allocs


def _collect_braced_definition(lines: List[str], start_idx: int) -> tuple[str, int]:
    text_parts = [lines[start_idx].rstrip()]
    brace_count = _brace_delta(lines[start_idx])
    idx = start_idx
    while brace_count > 0 and idx + 1 < len(lines):
        idx += 1
        text_parts.append(lines[idx].rstrip())
        brace_count += _brace_delta(lines[idx])
    return "\n".join(text_parts), idx


def _brace_delta(line: str) -> int:
    return line.count("{") - line.count("}")


def _is_alloc_or_free(line: str) -> bool:
    return "ws_rt.cmdh.waf.alloc" in line or "ws_rt.cmdh.waf.free" in line


def _is_layout_or_loc_def(line: str) -> bool:
    return line.lstrip().startswith("#layout") or line.lstrip().startswith("#loc")


def _is_region_start(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    if stripped.endswith("{"):
        return True
    return False


def _is_instruction_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    if stripped == "}" or stripped == "{" or stripped.endswith("} {") or stripped.endswith("{"):
        return False
    return " = " in stripped or stripped.startswith("ws_rt.") or stripped.startswith("ws.")


def _resolve_ref(line: str, pattern: re.Pattern[str], defs: Dict[str, object]) -> Optional[str]:
    match = pattern.search(line)
    if not match:
        return None
    name = match.group(1)
    definition = defs.get(name)
    if not definition:
        return f"{name} (definition not found)"
    return definition.text


def _allocs_for_line(line: str, allocs: Dict[str, List[AllocInfo]], line_index: int) -> List[str]:
    values = sorted(set(_SSA_RE.findall(line)))
    results: List[str] = []
    for value in values:
        info = _alloc_for_value(allocs, value, line_index)
        if info:
            results.append(info.summary)
    return results


def _alloc_for_value(
    allocs: Dict[str, List[AllocInfo]],
    value: str,
    line_index: int,
) -> Optional[AllocInfo]:
    candidates = allocs.get(value)
    if not candidates:
        return None
    line_number = line_index + 1
    best = None
    for info in candidates:
        if info.line_number <= line_number:
            best = info
        else:
            break
    return best or candidates[-1]


def _is_scratch_alloc(alloc: Optional[AllocInfo]) -> bool:
    if not alloc:
        return False
    return any("scratch" in extra.lower() for extra in alloc.extras)


def _compute_ws_rt_io(
    lines: List[str],
    allocs: Dict[str, List[AllocInfo]],
) -> Dict[int, tuple[List[str], List[str]]]:
    ws_rt_io: Dict[int, tuple[List[str], List[str]]] = {}
    last_def: Dict[str, int] = {}
    last_def_is_cast: Dict[str, bool] = {}
    suballoc_parent_def: Dict[str, int] = {}
    for idx, line in enumerate(lines):
        if _is_region_start(line) and "ws.func" in line:
            last_def = {}
            suballoc_parent_def = {}
        instruction = _parse_instruction_line(line, idx)
        if not instruction:
            continue
        if instruction.inst.startswith("ws_rt."):
            if instruction.inst.startswith("ws_rt.cmdh.waf.alloc"):
                ws_rt_io[idx] = ([], _operand_values(instruction.operands))
                continue
            if instruction.inst.startswith("ws_rt.cmdh.barrier"):
                ws_rt_io[idx] = ([], _operand_values(instruction.operands))
                continue
            operands = _operand_values(instruction.operands)
            if not operands:
                ws_rt_io[idx] = ([], [])
                continue
            arg_types = list(instruction.arg_types)
            if instruction.uniform_type and len(arg_types) < len(operands):
                arg_types = arg_types + [instruction.uniform_type] * (len(operands) - len(arg_types))
            handle_operands: set[str] = set()
            for pos, value in enumerate(operands):
                type_text = arg_types[pos] if pos < len(arg_types) else ""
                if "handle" in type_text:
                    handle_operands.add(value)
            counts: Dict[str, int] = {}
            for value in operands:
                counts[value] = counts.get(value, 0) + 1
            candidate_outputs: set[str] = set()
            is_cast_inst = "cast" in instruction.inst
            for value in operands:
                if value in handle_operands:
                    continue
                alloc = _alloc_for_value(allocs, value, idx)
                if _is_scratch_alloc(alloc):
                    continue
                parent = alloc.parent if alloc else None
                if parent:
                    parent_def = last_def.get(parent)
                    prev_parent_def = suballoc_parent_def.get(value)
                    is_cast_parent = parent_def is not None and last_def_is_cast.get(parent, False)
                    is_first_use = (
                        parent_def is not None
                        and parent_def != prev_parent_def
                        and not is_cast_parent
                    )
                else:
                    is_first_use = value not in last_def
                is_inplace = counts.get(value, 0) > 1
                if is_first_use or is_inplace:
                    candidate_outputs.add(value)
            outputs: List[str] = []
            inputs: List[str] = []
            inputs_started = False
            for value in operands:
                if value in handle_operands:
                    if value not in inputs:
                        inputs.append(value)
                    inputs_started = True
                    continue
                alloc = _alloc_for_value(allocs, value, idx)
                if _is_scratch_alloc(alloc):
                    if value not in inputs:
                        inputs.append(value)
                    inputs_started = True
                    continue
                parent = alloc.parent if alloc else None
                is_output = value in candidate_outputs and not (parent and parent in candidate_outputs)
                if is_output and value not in outputs:
                    outputs.append(value)
                is_inplace = counts.get(value, 0) > 1
                if inputs_started or not is_output or is_inplace:
                    if value not in inputs:
                        inputs.append(value)
                    inputs_started = True
                if parent:
                    parent_def = last_def.get(parent)
                    if parent_def is not None:
                        suballoc_parent_def[value] = parent_def
            for value in outputs:
                last_def[value] = idx
                last_def_is_cast[value] = is_cast_inst
            ws_rt_io[idx] = (outputs, inputs)
        else:
            for value in _split_values(instruction.results):
                if value:
                    last_def[value] = idx
    return ws_rt_io


def _instruction_display(
    line: str,
    source_index: int,
    options: RenderOptions,
    instructions: Dict[int, Instruction],
    allocs: Dict[str, List[AllocInfo]],
    ws_rt_io: Dict[int, tuple[List[str], List[str]]],
    source_vars: Optional[List[str]],
    show_full_source_vars: bool,
) -> str:
    instruction = _parse_instruction_line(line, source_index)
    if instruction:
        display = _instruction_label(
            instruction,
            options,
            allocs,
            source_index,
            options.show_alloc_sizes,
            options.show_left_types,
            ws_rt_io,
            source_vars,
            options.show_source_vars,
            show_full_source_vars,
        )
        instructions[source_index] = instruction
        return display
    return line.rstrip()


def _extract_inline_layout(line: str) -> Optional[str]:
    key_idx = line.find("layout")
    if key_idx == -1:
        return None
    eq_idx = line.find("=", key_idx)
    if eq_idx == -1:
        return None
    brace_idx = line.find("{", eq_idx)
    if brace_idx == -1:
        return None
    depth = 0
    for idx in range(brace_idx, len(line)):
        ch = line[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return line[brace_idx : idx + 1]
    return None


def _parse_instruction_line(line: str, source_index: int) -> Optional[Instruction]:
    if not _is_instruction_line(line):
        return None
    raw = line.rstrip()
    loc_ref = None
    layout_ref = None

    loc_match = _LOC_REF_RE.search(raw)
    if loc_match:
        loc_ref = loc_match.group(1)
        raw = raw.split(" loc(", 1)[0].rstrip()

    layout_match = _LAYOUT_REF_RE.search(raw)
    if layout_match:
        layout_ref = layout_match.group(1)

    attrs = None
    raw_body = raw
    type_idx = None
    depth = 0
    idx = 0
    while idx < len(raw_body) - 2:
        ch = raw_body[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        if depth == 0 and raw_body[idx : idx + 3] == " : ":
            type_idx = idx
            break
        idx += 1
    if type_idx is not None:
        raw_body = raw_body[:type_idx].rstrip()
        type_sig = raw[type_idx + 3 :].strip()
    else:
        type_sig = ""
    attr_idx = None
    depth = 0
    for idx, ch in enumerate(raw_body):
        if ch == "{":
            if depth == 0:
                attr_idx = idx
                break
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
    if attr_idx is not None:
        depth = 0
        end_idx = None
        for idx in range(attr_idx, len(raw_body)):
            ch = raw_body[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = idx
                    break
        if end_idx is not None:
            attrs = raw_body[attr_idx : end_idx + 1].strip()
            raw_body = (raw_body[:attr_idx] + raw_body[end_idx + 1 :]).strip()

    if " = " in raw_body:
        results, rest = raw_body.split(" = ", 1)
        results = results.strip()
    else:
        results = ""
        rest = raw_body.strip()

    parts = rest.split(maxsplit=1)
    inst = parts[0] if parts else rest
    operands = parts[1].strip() if len(parts) > 1 else ""

    arg_types, result_types, uniform_type = _parse_type_signature(type_sig)
    source_vars = _extract_source_vars(attrs) if attrs else []

    return Instruction(
        source_index=source_index,
        raw=line.rstrip(),
        results=results,
        inst=inst,
        operands=operands,
        attrs=attrs,
        layout_ref=layout_ref,
        loc_ref=loc_ref,
        arg_types=arg_types,
        result_types=result_types,
        uniform_type=uniform_type,
        source_vars=source_vars,
    )


def _region_label(line: str, options: RenderOptions) -> str:
    stripped = line.strip().rstrip("{").strip()
    if " attributes " in stripped:
        stripped = stripped.split(" attributes ", 1)[0].rstrip()
    if "{" in stripped:
        stripped = stripped.split("{", 1)[0].rstrip()
    if not options.show_full_prefix:
        stripped = stripped.replace(options.shorten_prefix, "")
    return stripped


def _collect_region_attrs(lines: List[str], start_idx: int) -> tuple[Optional[str], Optional[int]]:
    line = lines[start_idx]
    if "attributes" not in line:
        return None, None
    attr_start = line.find("attributes")
    brace_start = line.find("{", attr_start)
    if brace_start == -1:
        return None, None

    text_parts: List[str] = []
    idx = start_idx
    depth = 0
    started = False
    while idx < len(lines):
        segment = lines[idx]
        if idx == start_idx:
            segment = segment[brace_start:]
        for ch in segment:
            if ch == "{":
                depth += 1
                started = True
            if started:
                text_parts.append(ch)
            if ch == "}":
                depth -= 1
                if started and depth == 0:
                    return "".join(text_parts).strip(), idx
        if started:
            text_parts.append("\n")
        idx += 1

    return "".join(text_parts).strip(), idx


def _format_segments(instruction: Instruction, segment_index: int) -> str:
    segments = [instruction.results, instruction.inst, instruction.operands]
    labels = ["results", "inst", "operands"]
    rendered = []
    for idx, (label, value) in enumerate(zip(labels, segments)):
        if not value:
            continue
        if idx == segment_index:
            rendered.append(f">> {label}: {value}")
        else:
            rendered.append(f"{label}: {value}")
    return "\n".join(rendered) if rendered else instruction.raw


def _format_attrs(attr_text: str) -> List[str]:
    if not attr_text:
        return []
    text = attr_text.strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()
    if not text:
        return []
    parts = [part.strip() for part in _split_top_level(text, ",") if part.strip()]
    formatted: List[str] = []
    for part in parts:
        if re.match(r"^c\s*=", part):
            continue
        if re.match(r"^layout\s*=", part):
            continue
        if ":" in part and "=" in part:
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value.startswith("{") and value.endswith("}"):
                inner = value[1:-1].strip()
                formatted.append(_strip_type_annotations(key) + ":")
                if inner:
                    entries = [e.strip() for e in _split_top_level(inner, ",") if e.strip()]
                    for entry in entries:
                        formatted.append(f"  {_strip_type_annotations(entry)}")
                continue
        if "source_vars" in part:
            match = re.search(r"source_vars\s*=\s*\[(.*)\]", part)
            if match:
                list_body = match.group(1)
                entries = [e.strip() for e in _split_top_level(list_body, ",") if e.strip()]
                cleaned: List[str] = []
                for entry in entries:
                    if entry.startswith("\"") and entry.endswith("\"") and len(entry) >= 2:
                        entry = entry[1:-1]
                    if entry:
                        cleaned.append(entry)
                if cleaned:
                    formatted.append("source_vars:")
                    formatted.extend([f"  {item}" for item in cleaned])
                without = re.sub(r"source_vars\s*=\s*\[[^\]]*\]\s*,?\s*", "", part)
                without = without.replace("{ ,", "{").replace(", }", "}").replace("{ }", "{}")
                without = without.strip()
                if without and without not in {"cs.internal = {}", "cs.internal ={}"}:
                    formatted.append(_strip_type_annotations(without))
                continue
        formatted.append(_strip_type_annotations(part))
    return formatted


def _const_from_attrs(attrs: Optional[str]) -> Optional[str]:
    if not attrs:
        return None
    match = re.search(r"\bc\s*=\s*([^,}]+)", attrs)
    if not match:
        return None
    return _strip_type_annotations(match.group(1).strip())


def _op_attr_value(attrs: Optional[str]) -> Optional[str]:
    if not attrs:
        return None
    match = re.search(r"\b(?:_?op)\b\s*=\s*([^,}]+)", attrs)
    if not match:
        return None
    value = match.group(1).strip()
    value = _strip_type_annotations(value)
    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
        value = value[1:-1]
    return value or None


def _split_top_level(text: str, sep: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    in_quote = False
    prev = ""
    for ch in text:
        if ch == '"' and prev != "\\":
            in_quote = not in_quote
            buf.append(ch)
            prev = ch
            continue
        if not in_quote:
            if ch in "[{(<":
                depth += 1
            elif ch in "]})>":
                depth = max(0, depth - 1)
            if ch == sep and depth == 0:
                parts.append("".join(buf))
                buf = []
                prev = ch
                continue
        buf.append(ch)
        prev = ch
    if buf:
        parts.append("".join(buf))
    return parts


def _layout_for_index(layout_text: Optional[str], index: int) -> Optional[str]:
    info = _layout_for_index_with_type(layout_text, index)
    if not info:
        return None
    return info[0]


def _layout_type_for_index(layout_text: Optional[str], index: int) -> Optional[str]:
    info = _layout_for_index_with_type(layout_text, index)
    if not info:
        return None
    return info[1]


def _layout_for_index_with_type(
    layout_text: Optional[str],
    index: int,
) -> Optional[tuple[str, Optional[str]]]:
    if not layout_text:
        return None
    key = f"\"{index}\""
    start = layout_text.find(key)
    if start == -1:
        return None
    brace_start = layout_text.find("{", start)
    if brace_start == -1:
        return None
    depth = 0
    idx = brace_start
    while idx < len(layout_text):
        ch = layout_text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                raw = layout_text[start: idx + 1].strip()
                return _pretty_layout(raw)
        idx += 1
    return None


def _pretty_layout(raw: str) -> tuple[str, Optional[str]]:
    text = raw.replace("=", ":")
    text = _remove_value_types(text)
    brace_start = text.find("{")
    if brace_start == -1:
        return text.strip(), None
    prefix = text[:brace_start].strip()
    if re.match(r"^\"?\d+\"?\s*:$", prefix):
        prefix = ""
    body, _ = _extract_brace_body(text, brace_start)
    fields = [f.strip() for f in _split_top_level(body, ",") if f.strip()]
    formatted_fields: List[str] = []
    shape_value: Optional[str] = None
    reshape_value: Optional[str] = None
    shape_insert_idx: Optional[int] = None
    target_shape: Optional[str] = None
    layout_type: Optional[str] = None
    for field in fields:
        if ":" in field:
            key, value = field.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key == "Type":
                layout_type = value.strip('"')
                continue
            if key == "TargetShape":
                target_shape = _compact_layout_value(value)
                continue
            if key in {"Shape", "Reshape"}:
                if key == "Shape":
                    shape_value = _compact_layout_value(value)
                else:
                    reshape_value = _compact_layout_value(value)
                if shape_insert_idx is None:
                    shape_insert_idx = len(formatted_fields)
                continue
            if key in {"SrcTgts", "Ordering"} and value.startswith("["):
                items = _split_top_level(value[1:-1], ",")
                if key == "Ordering":
                    ordering = _format_ordering_list_from_body(value[1:-1])
                    formatted_fields.append(f"  {key}: {ordering}")
                    continue
                formatted_fields.append(f"  {key}: [")
                if key == "SrcTgts":
                    list_body = value[1:-1]
                    brace_entries: List[str] = []
                    depth = 0
                    start_idx = None
                    for idx, ch in enumerate(list_body):
                        if ch == "{":
                            if depth == 0:
                                start_idx = idx
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0 and start_idx is not None:
                                brace_entries.append(list_body[start_idx + 1 : idx])
                                start_idx = None
                    entries: List[List[str]] = []
                    if brace_entries:
                        for entry_text in brace_entries:
                            fields = [f.strip() for f in _split_top_level(entry_text, ",") if f.strip()]
                            entries.append(fields)
                    else:
                        raw_fields = [f.strip() for f in items if f.strip()]
                        current: List[str] = []
                        for field in raw_fields:
                            if field.startswith("HierSrc") and current:
                                entries.append(current)
                                current = [field]
                            else:
                                current.append(field)
                        if current:
                            entries.append(current)
                    max_hs = max_ht = max_ss = max_st = 0
                    parsed_entries: List[tuple[List[str], List[str], Optional[dict[str, str]]]] = []
                    for entry in entries:
                        perm_parts = [p for p in entry if p.lstrip().startswith("Perm")]
                        base_parts = [p for p in entry if p not in perm_parts]
                        fields = _parse_srctgts_fields(base_parts)
                        if fields:
                            max_hs = max(max_hs, len(fields.get("HierSrc", "")))
                            max_ht = max(max_ht, len(fields.get("HierTgt", "")))
                            max_ss = max(max_ss, len(fields.get("SizeSrc", "")))
                            max_st = max(max_st, len(fields.get("SizeTgt", "")))
                        parsed_entries.append((base_parts, perm_parts, fields))
                    parsed_entries.sort(key=lambda entry: _srctgts_sort_key(entry[2]))
                    for base_parts, perm_parts, fields in parsed_entries:
                        base_line = None
                        if fields:
                            base_line = _format_srctgts_entry(fields, max_hs, max_ss, max_ht, max_st)
                        if base_line:
                            perm_suffix = _format_perm_suffix(perm_parts)
                            if perm_suffix:
                                formatted_fields.append(f"    {base_line} {perm_suffix}")
                            else:
                                formatted_fields.append(f"    {base_line}")
                        elif base_parts:
                            formatted_fields.append(f"    { _compact_layout_value(', '.join(base_parts)) }")
                        if not base_parts and not perm_parts:
                            formatted_fields.append(f"    { _compact_layout_value(', '.join(entry)) }")
                else:
                    for item in items:
                        item = item.strip()
                        if not item:
                            continue
                        formatted_fields.append(f"    { _compact_layout_value(item) }")
                formatted_fields.append("  ]")
            else:
                formatted_fields.append(f"  {key}: {_compact_layout_value(value)}")
        else:
            formatted_fields.append(f"  {field}")
    if shape_value is not None or reshape_value is not None:
        if shape_value is not None and reshape_value is not None:
            shape_line = f"  Shape: {shape_value} → {reshape_value}"
        elif shape_value is not None:
            shape_line = f"  Shape: {shape_value}"
        else:
            shape_line = f"  Reshape: {reshape_value}"
        if target_shape:
            shape_line = f"{shape_line} | Target: {target_shape}"
        insert_at = shape_insert_idx if shape_insert_idx is not None else len(formatted_fields)
        formatted_fields.insert(insert_at, shape_line)
    lines = formatted_fields
    if prefix:
        lines = [prefix] + lines
    return "\n".join(lines), layout_type


def _remove_value_types(text: str) -> str:
    return re.sub(r"(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*:\s*[A-Za-z0-9_<>.]+", r"\1", text)


def _extract_brace_body(text: str, start: int) -> tuple[str, int]:
    depth = 0
    i = start
    body_start = None
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            if body_start is None:
                body_start = i + 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and body_start is not None:
                return text[body_start:i], i
        i += 1
    return text[start + 1 :], len(text)


def _compact_layout_value(value: str) -> str:
    compact = re.sub(r"\s+", " ", value)
    compact = compact.replace("[ ", "[").replace(" ]", "]")
    compact = compact.replace("{ ", "{").replace(" }", "}")
    compact = compact.replace(" ,", ",")
    compact = compact.replace("{", "").replace("}", "")
    return compact.strip()


def _parse_srctgts_fields(parts: List[str]) -> Optional[Dict[str, str]]:
    fields: Dict[str, str] = {}
    for part in parts:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        value = _compact_layout_value(value.strip())
        fields[key] = value
    return fields or None


def _format_srctgts_entry(
    fields: Dict[str, str],
    max_hier_src: int,
    max_size_src: int,
    max_hier_tgt: int,
    max_size_tgt: int,
) -> Optional[str]:
    hier_src = fields.get("HierSrc")
    hier_tgt = fields.get("HierTgt")
    size_src = fields.get("SizeSrc")
    size_tgt = fields.get("SizeTgt")
    if not (hier_src and hier_tgt and size_src and size_tgt):
        return None
    hs = hier_src.ljust(max_hier_src)
    ss = size_src.ljust(max_size_src)
    hier_tgt = _replace_first_dim(hier_tgt)
    ht = hier_tgt.ljust(max_hier_tgt)
    st = size_tgt.ljust(max_size_tgt)
    return f"{hs}({ss}) → {ht}({st})"


def _replace_first_dim(value: str) -> str:
    text = value.strip()
    if not (text.startswith("[") and "]" in text):
        return value
    inner = text[1 : text.find("]")]
    parts = [p.strip() for p in inner.split(",")]
    if not parts:
        return value
    if parts[0] == "0":
        parts[0] = "X"
    elif parts[0] == "1":
        parts[0] = "Y"
    return "[" + ", ".join(parts) + "]" + text[text.find("]") + 1 :]


def _format_perm_suffix(parts: List[str]) -> Optional[str]:
    if not parts:
        return None
    items: List[str] = []
    for part in parts:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        if key.startswith("Perm"):
            key = key[len("Perm") :]
        value = _compact_layout_value(value.strip())
        items.append(f"{key} : {value}")
    if not items:
        return None
    return f"Perm({', '.join(items)})"


def _format_ordering_list_from_body(list_body: str) -> str:
    brace_entries: List[str] = []
    depth = 0
    start_idx = None
    for idx, ch in enumerate(list_body):
        if ch == "{":
            if depth == 0:
                start_idx = idx
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start_idx is not None:
                brace_entries.append(list_body[start_idx + 1 : idx])
                start_idx = None

    entries: List[List[str]] = []
    if brace_entries:
        for entry_text in brace_entries:
            fields = [f.strip() for f in _split_top_level(entry_text, ",") if f.strip()]
            entries.append(fields)
    else:
        items = _split_top_level(list_body, ",")
        current: List[str] = []
        for item in items:
            item = item.strip().lstrip("{").rstrip("}").strip()
            if not item:
                continue
            if item.startswith("Dim") and current:
                entries.append(current)
                current = [item]
            else:
                current.append(item)
        if current:
            entries.append(current)

    formatted: List[str] = []
    for entry in entries:
        fields = _parse_srctgts_fields(entry) or {}
        dim = fields.get("Dim")
        stride = fields.get("Stride")
        if dim is not None and stride is not None:
            if stride == "0":
                formatted.append(f"{dim}")
            else:
                formatted.append(f"{dim}({stride})")
        else:
            formatted.append(_compact_layout_value(", ".join(entry)))
    return "[ " + ", ".join(formatted) + " ]"


def _format_shape_arrow(value: str) -> str:
    left, right = value.split("->", 1)
    left = _compact_layout_value(left.strip())
    right = _compact_layout_value(right.strip())
    return f"{left} → {right}"


def _srctgts_sort_key(fields: Optional[Dict[str, str]]) -> tuple:
    if not fields:
        return (1, ())
    hier_src = fields.get("HierSrc")
    if not hier_src:
        return (1, ())
    cleaned = hier_src.strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    numbers: List[int] = []
    for part in parts:
        try:
            numbers.append(int(part))
        except ValueError:
            return (0, tuple(parts))
    return (0, tuple(numbers))


def _strip_type_annotations(text: str) -> str:
    return re.sub(r"\s*:\s*[^,}\]]+", "", text)


def _extract_alloc_fields(line: str) -> tuple[str, Optional[str], Optional[str], Optional[str], List[str]]:
    type_match = re.search(r"<([^>]+)>", line)
    alloc_type = type_match.group(1) if type_match else "?"
    attrs_match = re.search(r"\{([^}]*)\}", line)
    attrs_text = attrs_match.group(1) if attrs_match else ""
    addr = _extract_attr_value(attrs_text, "addr")
    pe_bytes = _extract_attr_value(attrs_text, "pe_bytes")
    alignment = _extract_attr_value(attrs_text, "alignment")
    if addr:
        addr = _strip_type_annotations(addr)
    if pe_bytes:
        pe_bytes = _strip_type_annotations(pe_bytes)
    if alignment:
        alignment = _strip_type_annotations(alignment)
    extras: List[str] = []
    if attrs_text:
        parts = [p.strip() for p in _split_top_level(attrs_text, ",") if p.strip()]
        for part in parts:
            if part.startswith("addr") or part.startswith("pe_bytes") or part.startswith("alignment"):
                continue
            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip()
                value = _strip_type_annotations(value.strip())
                extras.append(f"{key}={value}")
            else:
                extras.append(part)
    return alloc_type, pe_bytes, addr, alignment, extras


def _summarize_alloc(
    alloc_type: str,
    pe_bytes: Optional[str],
    addr: Optional[str],
    alignment: Optional[str],
    extras: List[str],
) -> str:
    suffix = f", {', '.join(extras)}" if extras else ""
    if pe_bytes and addr and alignment:
        return f"{alloc_type}, {pe_bytes} b @ {addr}, align {alignment}{suffix}"
    if pe_bytes and addr:
        return f"{alloc_type}, {pe_bytes} b @ {addr}{suffix}"
    if pe_bytes:
        return f"{alloc_type}, {pe_bytes} b{suffix}"
    if addr:
        return f"{alloc_type} @ {addr}{suffix}"
    return f"{alloc_type}{suffix}"


def _instruction_label(
    instruction: Instruction,
    options: RenderOptions,
    allocs: Dict[str, List[AllocInfo]],
    line_index: int,
    show_alloc_sizes: bool,
    show_types: bool,
    ws_rt_io: Dict[int, tuple[List[str], List[str]]],
    source_vars: Optional[List[str]],
    show_source_vars: bool,
    show_full_source_vars: bool,
) -> str:
    const_value = _const_from_attrs(instruction.attrs)
    inst_name = instruction.inst
    op_suffix = _op_attr_value(instruction.attrs)
    if op_suffix:
        inst_name = f"{inst_name}.{op_suffix}"
    if not options.show_full_prefix and inst_name.startswith(options.shorten_prefix):
        inst_name = inst_name[len(options.shorten_prefix) :]
    if not options.show_full_prefix:
        inst_name = re.sub(r"\bmaster_(tx|rx)\b", r"\1", inst_name)
    semaphore_value = _semaphore_value(instruction.attrs)
    if semaphore_value:
        inst_name = f"{inst_name} → ({semaphore_value} tx)"
    num_rx_value = _num_rx_value(instruction.attrs)
    if not num_rx_value:
        op_value = _op_attr_value(instruction.attrs)
        inst_tail = instruction.inst.split(".")[-1]
        if (
            op_value
            and (op_value.startswith("slave") or op_value.startswith("switchroot"))
            and not op_value.startswith("slaveCacheStore")
            and inst_tail in {"master_tx"}
        ):
            num_rx_value = "1"
        elif (
            op_value
            and not op_value.startswith("slave")
            and not op_value.startswith("switchroot")
            and not op_value.startswith("none")
            and inst_tail in {"master_tx"}
        ):
            num_rx_value = "1"
    if num_rx_value:
        inst_name = f"{inst_name} → ({num_rx_value} rx)"
    iter_marker = ""
    if instruction.attrs and re.search(r"\bfirst_iteration_only\b", instruction.attrs):
        iter_marker = "➊ "
    onxy_prefix = _onxy_prefix(instruction.attrs)
    operands_display = instruction.operands
    if const_value:
        operands_display = f"{operands_display} [{const_value}]".strip()
    if instruction.inst.startswith("ws_rt."):
        outputs, inputs = ws_rt_io.get(line_index, ([], _operand_values(instruction.operands)))
        handle = instruction.results.strip() if instruction.results else ""
        output_types = instruction.arg_types[: len(outputs)] if instruction.arg_types else []
        input_types = instruction.arg_types[len(outputs) :] if instruction.arg_types else []
        if instruction.uniform_type:
            if len(output_types) < len(outputs):
                output_types = output_types + [instruction.uniform_type] * (len(outputs) - len(output_types))
            if len(input_types) < len(inputs):
                input_types = input_types + [instruction.uniform_type] * (len(inputs) - len(input_types))
        typed_outputs = ", ".join(
            _format_typed_values(outputs, output_types, allocs, line_index, show_alloc_sizes, show_types, is_arg=False)
        )
        typed_inputs = ", ".join(
            _format_typed_values(inputs, input_types, allocs, line_index, show_alloc_sizes, show_types, is_arg=True)
        )
        if const_value:
            typed_inputs = f"{typed_inputs} [{const_value}]".strip()
        if handle:
            handle_prefix = f"[{handle}] "
        else:
            handle_prefix = ""
        if typed_outputs:
            base = f"{iter_marker}{handle_prefix}{onxy_prefix}{typed_outputs} = {inst_name} {typed_inputs}".rstrip()
        else:
            base = f"{iter_marker}{handle_prefix}{onxy_prefix}{inst_name} {typed_inputs}".rstrip()
        return _append_source_vars(base, source_vars, show_source_vars, show_full_source_vars)
    if instruction.results:
        results = _split_values(instruction.results)
        operands = _split_values(instruction.operands)
        result_types = instruction.result_types
        arg_types = instruction.arg_types
        if instruction.uniform_type:
            if len(result_types) < len(results):
                result_types = result_types + [instruction.uniform_type] * (len(results) - len(result_types))
            if len(arg_types) < len(operands):
                arg_types = arg_types + [instruction.uniform_type] * (len(operands) - len(arg_types))
        typed_results = ", ".join(
            _format_typed_values(results, result_types, allocs, line_index, show_alloc_sizes, show_types, is_arg=False)
        )
        typed_operands = ", ".join(
            _format_typed_values(operands, arg_types, allocs, line_index, show_alloc_sizes, show_types, is_arg=True)
        )
        base = f"{iter_marker}{onxy_prefix}{typed_results} = {inst_name} {typed_operands}".rstrip()
        return _append_source_vars(base, source_vars, show_source_vars, show_full_source_vars)
    base = f"{iter_marker}{onxy_prefix}{inst_name} {operands_display}".rstrip()
    return _append_source_vars(base, source_vars, show_source_vars, show_full_source_vars)


def _format_alloc_table(
    values: List[str],
    types: List[str],
    allocs: Dict[str, List[AllocInfo]],
    line_index: int,
) -> List[str]:
    rows: List[tuple[str, str, str, str, str, str]] = []
    for idx, value in enumerate(values):
        if not value:
            continue
        alloc = _alloc_for_value(allocs, value, line_index)
        suffix = ""
        if alloc:
            suffix = _alloc_suffix(alloc.extras)
        value_label = f"{_value_with_parent(value, alloc)}{suffix}"
        alloc_type = _short_type(alloc.alloc_type) if alloc else None
        if alloc_type is None:
            alloc_type = _short_type(types[idx]) if idx < len(types) else None
        alloc_type = alloc_type or "?"
        bytes_text = alloc.bytes if alloc and alloc.bytes else "-"
        addr_text = alloc.addr if alloc and alloc.addr else "-"
        align_text = alloc.alignment if alloc and alloc.alignment else "-"
        extra_text = ", ".join(alloc.extras) if alloc and alloc.extras else ""
        rows.append((value_label, alloc_type, bytes_text, addr_text, align_text, extra_text))
    if not rows:
        return []
    max_value = max(len(r[0]) for r in rows)
    max_type = max(len(r[1]) for r in rows)
    max_bytes = max(len(r[2]) for r in rows)
    max_addr = max(len(r[3]) for r in rows)
    max_align = max(len(r[4]) for r in rows)
    lines: List[str] = []
    for value, alloc_type, bytes_text, addr_text, align_text, extra_text in rows:
        line = (
            f"{value.ljust(max_value)} : "
            f"{alloc_type.ljust(max_type)}, "
            f"{bytes_text.rjust(max_bytes)} b @ "
            f"{addr_text.ljust(max_addr)}, "
            f"align {align_text.ljust(max_align)}"
        )
        if extra_text:
            line = f"{line}, {extra_text}"
        lines.append(line)
    return lines


def _alloc_suffix(extras: List[str]) -> str:
    suffixes: List[str] = []
    extras_text = " ".join(extras).lower()
    meta_item = _single_metadata_item(extras)
    if "scratch" in extras_text:
        suffixes.append("scratch")
    if meta_item:
        suffixes.append(meta_item)
    elif "metadata" in extras_text:
        suffixes.append("meta")
    if "weightcontenttype" in extras_text or "weight" in extras_text:
        suffixes.append("wgt")
    if "cachecontenttype" in extras_text or "kv_cache" in extras_text:
        suffixes.append("kv")
    if not suffixes:
        return ""
    return "." + "".join(suffixes)


def _value_with_parent(value: str, alloc: Optional[AllocInfo]) -> str:
    if alloc and alloc.parent:
        return f"{value}⦿{alloc.parent}"
    return value


def _append_source_vars(
    base: str,
    source_vars: Optional[List[str]],
    show_source_vars: bool,
    show_full: bool,
) -> str:
    if not show_source_vars or not source_vars:
        return base
    rendered = source_vars
    if not show_full:
        rendered = [var.split(" ← ", 1)[0] for var in source_vars]
    return f"{base} ⟨{', '.join(rendered)}⟩"


def _onxy_prefix(attrs: Optional[str]) -> str:
    if not attrs:
        return ""
    prefixes: List[str] = []
    if re.search(r"\bonX\b", attrs):
        prefixes.append("onX")
    if re.search(r"\bonY\b", attrs):
        prefixes.append("onY")
    return " ".join(prefixes) + " " if prefixes else ""


def _semaphore_value(attrs: Optional[str]) -> Optional[str]:
    if not attrs:
        return None
    match = re.search(r"\bsemaphore\s*=\s*([^,}]+)", attrs)
    if not match:
        return None
    value = match.group(1).strip()
    value = _strip_type_annotations(value)
    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
        value = value[1:-1]
    value = value.strip()
    if not value:
        return None
    if _is_zero_literal(value):
        return None
    return value


def _num_rx_value(attrs: Optional[str]) -> Optional[str]:
    if not attrs:
        return None
    match = re.search(r"\bnum_rx\s*=\s*([^,}]+)", attrs)
    if not match:
        return None
    value = match.group(1).strip()
    value = _strip_type_annotations(value)
    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
        value = value[1:-1]
    value = value.strip()
    if not value:
        return None
    if _is_zero_literal(value):
        return None
    return value


def _is_zero_literal(value: str) -> bool:
    return re.fullmatch(r"0+(?:\.0+)?(?:e[+-]?0+)?", value, re.IGNORECASE) is not None


def _format_layout_section(
    results: List[str],
    args: List[str],
    layout_text: Optional[str],
) -> List[str]:
    if not layout_text:
        return []
    grouped: Dict[str, List[tuple[str, Optional[str]]]] = {}
    for idx, value in enumerate(results):
        info = _layout_for_index_with_type(layout_text, idx)
        if not info:
            continue
        layout_info, layout_type = info
        grouped.setdefault(layout_info, []).append((value, layout_type))
    offset = len(results)
    for idx, value in enumerate(args):
        info = _layout_for_index_with_type(layout_text, offset + idx)
        if not info:
            continue
        layout_info, layout_type = info
        grouped.setdefault(layout_info, []).append((value, layout_type))

    lines: List[str] = []
    for layout_info, values in grouped.items():
        labeled = []
        for value, layout_type in values:
            if layout_type:
                labeled.append(f"{value} ({layout_type})")
            else:
                labeled.append(value)
        lines.append(f"{', '.join(labeled)}:")
        lines.extend([f"  {line}" for line in layout_info.splitlines()])
    return lines


def _extract_attr_value(attrs_text: str, key: str) -> Optional[str]:
    match = re.search(rf"{re.escape(key)}\s*=\s*([^,}}]+)", attrs_text)
    if not match:
        return None
    return match.group(1).strip()


def _extract_loc_text(loc_text: Optional[str]) -> Optional[str]:
    if not loc_text:
        return None
    match = re.search(r"loc\(\"(.*?)\"\)", loc_text)
    if match:
        return match.group(1)
    return loc_text


def _extract_source_vars(attrs: Optional[str]) -> List[str]:
    if not attrs:
        return []
    key_idx = attrs.find("source_vars")
    if key_idx == -1:
        return []
    bracket_idx = attrs.find("[", key_idx)
    if bracket_idx == -1:
        return []
    depth = 0
    end_idx = None
    for idx in range(bracket_idx, len(attrs)):
        ch = attrs[idx]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end_idx = idx
                break
    if end_idx is None:
        return []
    list_body = attrs[bracket_idx + 1 : end_idx]
    entries = [e.strip() for e in _split_top_level(list_body, ",") if e.strip()]
    cleaned: List[str] = []
    for entry in entries:
        if entry.startswith("\"") and entry.endswith("\"") and len(entry) >= 2:
            entry = entry[1:-1]
        if entry:
            if "," in entry:
                cleaned.extend([part.strip() for part in entry.split(",") if part.strip()])
            else:
                cleaned.append(entry)
    return cleaned


def _compute_source_var_suffixes(
    lines: List[str],
    instructions: Dict[int, Instruction],
) -> Dict[int, List[str]]:
    suffixes: Dict[int, List[str]] = {}
    func_instrs: List[tuple[int, List[str]]] = []

    def flush() -> None:
        if not func_instrs:
            return
        all_vars = [v for _, vars_list in func_instrs for v in vars_list]
        prefix = _trim_prefix(_common_prefix(all_vars)) if all_vars else ""
        for line_idx, vars_list in func_instrs:
            trimmed = [_strip_prefix(v, prefix) for v in vars_list]
            suffixes[line_idx] = [_format_source_var(v) for v in trimmed]

    for idx, line in enumerate(lines):
        if _is_region_start(line) and "ws.func" in line:
            flush()
            func_instrs = []
        instruction = instructions.get(idx)
        if instruction and instruction.source_vars:
            func_instrs.append((idx, instruction.source_vars))
    flush()
    return suffixes


def _common_prefix(items: List[str]) -> str:
    if not items:
        return ""
    prefix = items[0]
    for item in items[1:]:
        i = 0
        while i < len(prefix) and i < len(item) and prefix[i] == item[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            break
    return prefix


def _trim_prefix(prefix: str) -> str:
    if not prefix:
        return ""
    last = max(prefix.rfind(":"), prefix.rfind("/"), prefix.rfind("."))
    if last == -1:
        return ""
    return prefix[: last + 1]


def _strip_prefix(value: str, prefix: str) -> str:
    if prefix and value.startswith(prefix):
        value = value[len(prefix) :]
    return value.lstrip(":/.")


def _format_source_var(value: str) -> str:
    parts = [p for p in value.split(":") if p]
    if not parts:
        return value
    return " ← ".join(reversed(parts))


def _highlight_details(text: str) -> Text:
    if not text:
        return Text("")
    raw_lines = text.splitlines()
    max_len = max((len(line) for line in raw_lines), default=0)
    padded_lines = [line.ljust(max_len) for line in raw_lines]
    padded_text = "\n".join(padded_lines)
    rendered = Text(padded_text)
    lines = padded_text.splitlines(keepends=True)
    offset = 0
    in_srctgts = False
    section_titles = {"Allocations", "Layouts", "Attributes", "Location"}
    for line in lines:
        stripped = line.strip()
        if stripped in section_titles:
            rendered.stylize("bold white on grey23", offset, offset + len(line))
        if set(stripped) == {"-"} and len(stripped) >= 3:
            rendered.stylize("grey50", offset, offset + len(line))
        if stripped.startswith("SrcTgts:"):
            in_srctgts = True
        elif in_srctgts and stripped == "]":
            in_srctgts = False
        if in_srctgts and "→" in line:
            for match in re.finditer(r"\[[^\]]*\]", line):
                rendered.stylize("cyan", offset + match.start(), offset + match.end())
            for match in re.finditer(r"\([^\)]*\)", line):
                rendered.stylize("magenta", offset + match.start(), offset + match.end())
            tgt_match = re.search(r"→\s*(\[[^\]]*\])", line)
            if tgt_match:
                tgt_start = offset + tgt_match.start(1)
                tgt_text = tgt_match.group(1)
                for m in re.finditer(r"\bX\b", tgt_text):
                    rendered.stylize("light_green", tgt_start + m.start(), tgt_start + m.end())
                for m in re.finditer(r"\bY\b", tgt_text):
                    rendered.stylize("yellow", tgt_start + m.start(), tgt_start + m.end())
        if "Ordering:" in line:
            for match in re.finditer(r"([^\s,\[]+)\(([^\)]+)\)", line):
                dim_start = offset + match.start(1)
                dim_end = offset + match.end(1)
                stride_start = offset + match.start(2)
                stride_end = offset + match.end(2)
                rendered.stylize("cyan", dim_start, dim_end)
                rendered.stylize("magenta", stride_start, stride_end)
        if "Shape:" in line and "→" in line:
            arrow_idx = line.find("→")
            if arrow_idx != -1:
                target_idx = line.find("| Target")
                end_idx = target_idx if target_idx != -1 else len(line)
                rendered.stylize("magenta", offset + arrow_idx + 1, offset + end_idx)
        for match in re.finditer(r"%\d+\.[A-Za-z0-9_.-]+", line):
            suffix_start = offset + match.start() + match.group(0).find(".")
            suffix_end = offset + match.end()
            rendered.stylize("orange1", suffix_start, suffix_end)
        for match in re.finditer(r"\(T_[A-Za-z0-9_]+\)", line):
            rendered.stylize("magenta", offset + match.start(), offset + match.end())
        alloc_match = re.search(
            r"^(?P<value>[^:]+)\s*:\s*(?P<type>[^,]+),\s*(?P<bytes>[^ ]+)\s*b\s*@\s*(?P<addr>[^,]+),\s*align\s*(?P<align>.+)$",
            stripped,
        )
        if alloc_match:
            type_span = alloc_match.span("type")
            bytes_span = alloc_match.span("bytes")
            addr_span = alloc_match.span("addr")
            align_span = alloc_match.span("align")
            rendered.stylize("green", offset + type_span[0], offset + type_span[1])
            rendered.stylize("yellow", offset + bytes_span[0], offset + bytes_span[1])
            rendered.stylize("cyan", offset + addr_span[0], offset + addr_span[1])
            rendered.stylize("magenta", offset + align_span[0], offset + align_span[1])
        offset += len(line)
    return rendered


def _instruction_buffers(instruction: Instruction) -> tuple[List[str], List[str], List[str]]:
    results = _split_values(instruction.results)
    operands = _operand_values(instruction.operands)
    if instruction.inst.startswith("ws_rt."):
        if instruction.inst.startswith("ws_rt.cmdh.waf.alloc"):
            return [], results, operands
        if instruction.inst.startswith("ws_rt.cmdh.waf.free"):
            return [], [], operands
        handles = results
        res_buffers = operands[:1]
        arg_buffers = operands[1:]
        return handles, res_buffers, arg_buffers
    return [], results, operands


def _format_values_section(
    values: List[str],
    layout_offset: int,
    layout_text: Optional[str],
    allocs: Dict[str, List[AllocInfo]],
    line_index: int,
) -> List[str]:
    lines: List[str] = []
    for idx, value in enumerate(values):
        if not value:
            continue
        lines.append(value)
        alloc = _alloc_for_value(allocs, value, line_index)
        if alloc:
            lines.append(alloc.summary)
        layout_info = _layout_for_index(layout_text, layout_offset + idx)
        if layout_info:
            lines.append(layout_info)
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _split_values(text: str) -> List[str]:
    if not text:
        return []
    return [part.strip() for part in _split_top_level(text, ",") if part.strip()]


def _operand_values(text: str) -> List[str]:
    if not text:
        return []
    return _SSA_RE.findall(text)


def _parse_type_signature(type_sig: str) -> tuple[List[str], List[str], Optional[str]]:
    if not type_sig or "->" not in type_sig:
        return [], [], None
    left, right = type_sig.split("->", 1)
    left = left.strip()
    right = right.strip()
    uniform_type = None
    arg_text = left
    if left.startswith("(") and left.endswith(")"):
        arg_text = left[1:-1].strip()
    else:
        if "," not in left and left:
            uniform_type = left
    arg_types = [t.strip() for t in _split_top_level(arg_text, ",") if t.strip()]
    if right.startswith("(") and right.endswith(")"):
        right = right[1:-1].strip()
    result_types = [t.strip() for t in _split_top_level(right, ",") if t.strip()]
    return arg_types, result_types, uniform_type


def _short_type(type_text: Optional[str]) -> Optional[str]:
    if not type_text:
        return None
    match = re.search(r"<([^>]+)>", type_text)
    if match:
        return match.group(1)
    return type_text


def _format_typed_value(
    value: str,
    type_text: Optional[str],
    allocs: Dict[str, List[AllocInfo]],
    line_index: int,
    show_alloc_sizes: bool,
    show_types: bool,
    is_arg: bool,
) -> str:
    if not value:
        return value
    if type_text == "!ws_rt.prefetch_handle":
        alloc = _alloc_for_value(allocs, value, line_index)
        suffix = ".handle"
        size_text = ""
        if show_alloc_sizes and alloc and alloc.bytes:
            size_text = f"({alloc.bytes})"
        return f"{_value_with_parent(value, alloc)}{suffix}{size_text}"
    short = _short_type(type_text)
    alloc = _alloc_for_value(allocs, value, line_index)
    suffix = ""
    if alloc:
        if is_arg:
            meta_item = _single_metadata_item(alloc.extras)
            if meta_item:
                suffix = f".{meta_item}"
        if not suffix:
            suffix = _alloc_suffix(alloc.extras)
    size_text = ""
    if show_alloc_sizes and alloc and alloc.bytes:
        size_text = f"({alloc.bytes})"
    value_display = f"{_value_with_parent(value, alloc)}{suffix}{size_text}"
    if not show_types or not short:
        return value_display
    return f"{short} {value_display}"


def _format_typed_values(
    values: List[str],
    types: List[str],
    allocs: Dict[str, List[AllocInfo]],
    line_index: int,
    show_alloc_sizes: bool,
    show_types: bool,
    is_arg: bool = False,
) -> List[str]:
    typed: List[str] = []
    for idx, value in enumerate(values):
        type_text = types[idx] if idx < len(types) else None
        typed.append(
            _format_typed_value(value, type_text, allocs, line_index, show_alloc_sizes, show_types, is_arg)
        )
    return typed


def _single_metadata_item(extras: List[str]) -> Optional[str]:
    for extra in extras:
        match = re.search(r"cs\.inference\.metadata\s*=\s*\[([^\]]*)\]", extra, re.IGNORECASE)
        if not match:
            continue
        content = match.group(1).strip()
        if not content:
            return None
        parts = [part.strip() for part in content.split(",") if part.strip()]
        if len(parts) != 1:
            return None
        item = parts[0]
        if item.startswith("\"") and item.endswith("\"") and len(item) >= 2:
            item = item[1:-1]
        return item.strip() or None
    return None


def _is_tx_instruction(inst_name: str) -> bool:
    name = inst_name.split(".")[-1]
    return name in {"tx", "txact", "request_txact", "master_tx", "slave_reconfig"}


def _is_rx_consumer(inst_name: str, attrs: Optional[str]) -> bool:
    name = inst_name.split(".")[-1]
    if name not in {"rx", "rxact", "request_rxact", "master_rx"}:
        return False
    op_value = _op_attr_value(attrs)
    # rx is a consumer if it's slave/switchroot OR if it's non-slave, non-switchroot, non-none
    if op_value:
        return not op_value.startswith("none")
    return False


def _axis_from_attrs(attrs: Optional[str]) -> Optional[str]:
    if not attrs:
        return None
    if "onX" in attrs:
        return "X"
    if "onY" in attrs:
        return "Y"
    return None


def _parse_semaphore_count(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    try:
        as_float = float(value)
    except ValueError:
        return None
    if as_float.is_integer():
        return int(as_float)
    return None


def _compute_semaphore_relationships(instructions: Dict[int, Instruction]) -> None:
    """Compute semaphore and num_rx producer-consumer relationships for all instructions."""
    # Separate queues for same-axis and opposite-axis num_rx consumption
    semaphore_queues: Dict[Optional[str], List[tuple[int, int]]] = {"X": [], "Y": [], None: []}
    num_rx_queues_same: Dict[Optional[str], List[tuple[int, int]]] = {"X": [], "Y": [], None: []}
    num_rx_queues_opposite: Dict[Optional[str], List[tuple[int, int]]] = {"X": [], "Y": [], None: []}
    
    # Process all instructions in order
    for line_idx in sorted(instructions.keys()):
        instruction = instructions[line_idx]
        axis = _axis_from_attrs(instruction.attrs)
        
        # Handle semaphore production
        sem_value = _semaphore_value(instruction.attrs)
        sem_count = _parse_semaphore_count(sem_value)
        if sem_count and sem_count > 0:
            semaphore_queues[axis].append((line_idx, sem_count))
        
        # Handle num_rx production
        num_rx_value = _num_rx_value(instruction.attrs)
        num_rx_count = _parse_semaphore_count(num_rx_value)
        is_slave_or_switchroot_tx = False
        is_other_tx = False
        
        if not num_rx_count:
            op_value = _op_attr_value(instruction.attrs)
            inst_tail = instruction.inst.split(".")[-1]
            if (
                op_value
                and (op_value.startswith("slave") or op_value.startswith("switchroot"))
                and not op_value.startswith("slaveCacheStore")
                and inst_tail in {"master_tx"}
            ):
                num_rx_count = 1
                is_slave_or_switchroot_tx = True
            elif (
                op_value
                and not op_value.startswith("slave")
                and not op_value.startswith("switchroot")
                and not op_value.startswith("none")
                and inst_tail in {"master_tx"}
            ):
                num_rx_count = 1
                is_other_tx = True
        
        if num_rx_count and num_rx_count > 0:
            # Route to appropriate queue
            if is_slave_or_switchroot_tx:
                num_rx_queues_opposite[axis].append((line_idx, num_rx_count))
            elif is_other_tx:
                num_rx_queues_same[axis].append((line_idx, num_rx_count))
            else:
                # Explicit num_rx - check op to determine queue
                op_value = _op_attr_value(instruction.attrs)
                if op_value and (op_value.startswith("slave") or op_value.startswith("switchroot")):
                    num_rx_queues_opposite[axis].append((line_idx, num_rx_count))
                else:
                    num_rx_queues_same[axis].append((line_idx, num_rx_count))
        
        # Handle semaphore consumption
        if _is_tx_instruction(instruction.inst):
            queue = semaphore_queues[axis]
            if queue:
                producer_idx, remaining = queue[0]
                # Update both producer and consumer
                object.__setattr__(instruction, 'semaphore_unblocker', producer_idx)
                producer = instructions[producer_idx]
                new_consumers = list(producer.semaphore_consumers) + [line_idx]
                object.__setattr__(producer, 'semaphore_consumers', new_consumers)
                remaining -= 1
                if remaining <= 0:
                    queue.pop(0)
                else:
                    queue[0] = (producer_idx, remaining)
        
        # Handle num_rx consumption
        if _is_rx_consumer(instruction.inst, instruction.attrs):
            consumer_axis = _axis_from_attrs(instruction.attrs)
            consumer_op = _op_attr_value(instruction.attrs)
            
            # Determine which queue based on consumer's op
            is_opposite_axis_rx = consumer_op and (
                consumer_op.startswith("slave") or consumer_op.startswith("switchroot")
            )
            
            if is_opposite_axis_rx:
                # Try opposite-axis queue
                queue_axes: List[Optional[str]] = []
                if consumer_axis in {"X", "Y"}:
                    opposite = "Y" if consumer_axis == "X" else "X"
                    queue_axes = [opposite, None]
                else:
                    queue_axes = ["X", "Y", None]
                
                for axis_key in queue_axes:
                    queue = num_rx_queues_opposite[axis_key]
                    if not queue:
                        continue
                    producer_idx, remaining = queue[0]
                    # Update both producer and consumer
                    object.__setattr__(instruction, 'num_rx_unblocker', producer_idx)
                    producer = instructions[producer_idx]
                    new_consumers = list(producer.num_rx_consumers) + [line_idx]
                    object.__setattr__(producer, 'num_rx_consumers', new_consumers)
                    remaining -= 1
                    if remaining <= 0:
                        queue.pop(0)
                    else:
                        queue[0] = (producer_idx, remaining)
                    break
            else:
                # Try same-axis queue
                queue_axes: List[Optional[str]] = []
                if consumer_axis in {"X", "Y"}:
                    queue_axes = [consumer_axis, None]
                else:
                    queue_axes = ["X", "Y", None]
                
                for axis_key in queue_axes:
                    queue = num_rx_queues_same[axis_key]
                    if not queue:
                        continue
                    producer_idx, remaining = queue[0]
                    # Update both producer and consumer
                    object.__setattr__(instruction, 'num_rx_unblocker', producer_idx)
                    producer = instructions[producer_idx]
                    new_consumers = list(producer.num_rx_consumers) + [line_idx]
                    object.__setattr__(producer, 'num_rx_consumers', new_consumers)
                    remaining -= 1
                    if remaining <= 0:
                        queue.pop(0)
                    else:
                        queue[0] = (producer_idx, remaining)
                    break

