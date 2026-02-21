from __future__ import annotations

from pathlib import Path
import argparse
import re
import sys

from textual.app import App, ComposeResult
from textual import events
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, RichLog, ListView, ListItem, Label, Checkbox, Input, DataTable
from rich.text import Text
from textual.binding import Binding
from textual_autocomplete import AutoComplete, DropdownItem

from .core import (
    Document,
    DocumentView,
    RenderOptions,
    Node,
    NodeData,
    _split_values,
    _expanded_result_values,
    _operand_values,
    _alloc_for_value,
    _brace_delta,
    _split_top_level,
    _strip_type_annotations,
    _op_attr_value,
    _semaphore_value,
    _num_rx_value,
    _instruction_label,
    _highlight_details,
    _axis_from_attrs,
)


_TYPE_RE = re.compile(r"\b([A-Za-z0-9_<>!.]+)\s+%[A-Za-z0-9_$.]+")
_ALLOC_SUFFIX_RE = re.compile(r"%[A-Za-z0-9_$.]+\.[A-Za-z0-9_.-]+")
_SIZE_RE = re.compile(r"\(\d+\)")
_SOURCE_VARS_RE = re.compile(r"⟨[^⟩]*⟩")
_LOC_REF_RE = re.compile(r"loc\((#loc\d+)\)")


class IRViewerApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #main {
        height: 1fr;
    }

    #list {
        width: 1fr;
    }

    #details {
        width: 1fr;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("i", "toggle_details", "More info"),
        Binding("z", "toggle_zoom", "Zoom details"),
        Binding("r", "open_report", "Histogram report"),
        Binding("f", "open_jump", "Switch function"),
        Binding("enter", "open_quick_jump", "Quick jump", show=False),
        Binding("o", "open_toggles", "Options"),
        Binding("w", "toggle_left_wrap", "Switch wrap"),
        Binding("/", "open_search", "Search"),
        Binding("?", "open_attr_search", "Attr search"),
        Binding("H", "open_help", "Help"),
        Binding("n", "search_next", "Next match", show=False),
        Binding("N", "search_prev", "Prev match", show=False),
        Binding("ctrl+c", "command_palette", "Command palette", show=False),
        Binding("ctrl+p", "emacs_prev_line", "", show=False),
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("pageup", "page_up", "Page up", show=False),
        Binding("pagedown", "page_down", "Page down", show=False),
        Binding("home", "go_home", "Home", show=False),
        Binding("end", "go_end", "End", show=False),
        Binding("space", "select_current", "Select", show=False),
        Binding("j", "move_down", "Down", show=False),
        Binding("k", "move_up", "Up", show=False),
        Binding("h", "scroll_left", "Scroll left", show=False),
        Binding("l", "scroll_right", "Scroll right", show=False),
        Binding("0", "scroll_leftmost", "Scroll leftmost", show=False),
        Binding("$", "scroll_rightmost", "Scroll rightmost", show=False),
        Binding("ctrl+d", "page_down", "Page down", show=False),
        Binding("ctrl+u", "page_up", "Page up", show=False),
        Binding("g", "go_home", "Home", show=False),
        Binding("G", "go_end", "End", show=False),
        Binding("tab", "toggle_focus", "Toggle focus"),
        Binding("shift+tab", "toggle_focus_back", "Toggle focus back"),
    ]

    def __init__(
        self,
        path: Path,
        profile_mode: bool = False,
        options: RenderOptions | None = None,
        file_choices: list[Path] | None = None,
        file_root: Path | None = None,
    ) -> None:
        super().__init__()
        self.options = options or RenderOptions()
        self._file_choices = file_choices or []
        self._file_root = file_root
        self._current_path = path
        self._load_path(path)
        self.profile_mode = profile_mode
        self.segment_index = 0
        self.selected_node = NodeData(kind="inst", source_index=0, attrs=None, value=None, layout_index=None)
        self.def_map: dict[str, list[int]] = {}
        self.use_map: dict[str, list[int]] = {}
        self.use_by_def: dict[int, list[int]] = {}
        self._pending_highlight: NodeData | None = None
        self._highlight_timer = None
        self._jump_sections: list[tuple[str, int]] = []
        self._ws_funcs: list[tuple[str, int, int]] = _collect_ws_funcs(self.document.lines)
        self._active_func_index: int | None = None
        self._container_by_line: dict[int, str | None] = _collect_host_container_by_line(self.document.lines)
        self._def_container_by_line: dict[int, str | None] = {}
        self._highlight_level1: set[int] = set()
        self._highlight_level2: set[int] = set()
        self._highlight_level3: set[int] = set()
        self._highlight_semaphore: int | None = None
        self._highlight_num_rx: int | None = None
        self._highlight_semaphore_consumers: set[int] = set()
        self._highlight_num_rx_consumers: set[int] = set()
        self._label_cache: dict[tuple[int, str], Text] = {}
        self._flat_entries: list[LineEntry] = []
        self._selected_index = 0
        self._window_start = 0
        self._select_timer = None
        self._last_search: str | None = None
        self._last_search_target: str = "list"
        self._details_lines: list[str] = []
        self._details_matches: list[int] = []
        self._details_match_pos: int = -1
        self._focus_target: str = "list"
        self._last_attr_search: tuple[str, str | None] | None = None
        self._last_list_width: int | None = None
        self._split_focus_side: str = "left"
        self._zoom_details: bool = False
        self._zoom_prev_details_visible: bool = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            yield RichLog(id="list", auto_scroll=False, wrap=True)
            yield RichLog(id="details", auto_scroll=False, wrap=True)
        yield Input(placeholder="/", id="search_bar")
        yield Footer()

    def on_mount(self) -> None:
        search_bar = self.query_one("#search_bar", Input)
        search_bar.display = False
        details = self.query_one("#details", RichLog)
        details.display = False
        details.can_focus = True
        list_view = self.query_one("#list", RichLog)
        list_view.can_focus = True
        list_view.wrap = False if self.options.split_axis_view else self.options.wrap_left_panel
        self._last_list_width = list_view.size.width
        if self._file_choices and len(self._file_choices) > 1 and not self.profile_mode:
            self.push_screen(
                FileSelectScreen(self._file_choices, self._file_root),
                self._select_file,
            )
            return
        self._post_load_setup()
        if self.profile_mode:
            self.set_timer(0.05, self.exit)

    def _post_load_setup(self) -> None:
        if self._ws_funcs:
            if len(self._ws_funcs) == 1:
                self._active_func_index = 0
                self._render_list()
            else:
                if self.profile_mode:
                    self._active_func_index = 0
                    self._render_list()
                else:
                    self.push_screen(FuncSelectScreen(self._ws_funcs), self._select_func)
        else:
            self._render_list()

    def _select_file(self, selection: int | None) -> None:
        if selection is None:
            return
        path = self._file_choices[selection]
        self._load_path(path)
        self._post_load_setup()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        return


    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        return

    def on_mouse_down(self, event: events.MouseDown) -> None:
        widget = event.widget
        if isinstance(widget, RichLog):
            if widget.id == "list":
                self._focus_target = "list"
                return
            if widget.id == "details":
                self._focus_target = "details"
                return

    def _apply_pending_highlight(self) -> None:
        if self._pending_highlight is None:
            return
        self._update_related_highlights(self._pending_highlight)

    def action_toggle_details(self) -> None:
        if self._zoom_details:
            return
        details = self.query_one("#details", RichLog)
        details.display = not details.display
        list_view = self.query_one("#list", RichLog)
        list_view.wrap = self.options.wrap_left_panel
        list_view.refresh(layout=True)
        details.refresh(layout=True)
        self.call_after_refresh(self._render_list)
        if not details.display:
            self._focus_target = "list"
            self.query_one("#list", RichLog).focus()

    def action_toggle_zoom(self) -> None:
        list_view = self.query_one("#list", RichLog)
        details = self.query_one("#details", RichLog)
        if not self._zoom_details:
            self._zoom_prev_details_visible = bool(details.display)
            list_view.display = False
            details.display = True
            self._zoom_details = True
            self._focus_target = "details"
            details.focus()
        else:
            list_view.display = True
            details.display = self._zoom_prev_details_visible
            self._zoom_details = False
            if details.display:
                self._focus_target = "details"
                details.focus()
            else:
                self._focus_target = "list"
                list_view.focus()
        list_view.refresh(layout=True)
        details.refresh(layout=True)

    def _ensure_details_visible(self) -> None:
        details = self.query_one("#details", RichLog)
        if details.display:
            return
        details.display = True
        if not self._zoom_details:
            list_view = self.query_one("#list", RichLog)
            list_view.wrap = self.options.wrap_left_panel
            list_view.refresh(layout=True)
        details.refresh(layout=True)
        self.call_after_refresh(self._render_list)

    def action_toggle_left_wrap(self) -> None:
        self.options.wrap_left_panel = not self.options.wrap_left_panel
        list_view = self.query_one("#list", RichLog)
        list_view.wrap = self.options.wrap_left_panel
        list_view.refresh(layout=True)
        self.call_after_refresh(self._render_list)

    def on_resize(self, event: events.Resize) -> None:
        list_view = self.query_one("#list", RichLog)
        width = list_view.size.width
        if self._last_list_width != width:
            self._last_list_width = width
            self.call_after_refresh(self._render_list)

    def action_next_segment(self) -> None:
        self.segment_index = (self.segment_index + 1) % 3
        self._update_details(self.selected_node)

    def action_prev_segment(self) -> None:
        self.segment_index = (self.segment_index - 1) % 3
        self._update_details(self.selected_node)

    def action_move_up(self) -> None:
        if self._focus_target == "details":
            self._scroll_details(-1)
        else:
            self._move_selection(-1)

    def action_move_down(self) -> None:
        if self._focus_target == "details":
            self._scroll_details(1)
        else:
            self._move_selection(1)

    def action_page_up(self) -> None:
        if self._focus_target == "details":
            self._scroll_details(-self._details_page_delta())
        else:
            self._move_selection(-max(1, self._viewport_size() - 3))

    def action_page_down(self) -> None:
        if self._focus_target == "details":
            self._scroll_details(self._details_page_delta())
        else:
            self._move_selection(max(1, self._viewport_size() - 3))

    def action_emacs_prev_line(self) -> None:
        self.action_move_up()

    def action_scroll_left(self) -> None:
        self._scroll_horizontal(-4)

    def action_scroll_right(self) -> None:
        self._scroll_horizontal(4)

    def action_scroll_leftmost(self) -> None:
        self._scroll_horizontal_to(0)

    def action_scroll_rightmost(self) -> None:
        self._scroll_horizontal_to(10_000)

    def action_go_home(self) -> None:
        if self._focus_target == "details":
            self._scroll_details_to(0)
        else:
            self._set_selected_index(0)

    def action_go_end(self) -> None:
        if self._focus_target == "details":
            self._scroll_details_to(1_000_000)
        else:
            self._set_selected_index(len(self._flat_entries) - 1)

    def action_select_current(self) -> None:
        self._apply_selection(apply_highlights=True)

    def action_open_jump(self) -> None:
        if not self._ws_funcs:
            return
        self.push_screen(FuncSelectScreen(self._ws_funcs), self._select_func)

    def action_open_search(self) -> None:
        search_bar = self.query_one("#search_bar", Input)
        search_bar.value = "/"
        search_bar.display = True
        search_bar.focus()

    def action_open_attr_search(self) -> None:
        self.push_screen(AttrSearchScreen(self._collect_attr_options()), self._apply_attr_search)

    def _collect_attr_options(self) -> list[tuple[str, str | None]]:
        names: set[str] = set()
        values_by_key: dict[str, set[str]] = {}
        for entry in self._flat_entries:
            attr_sources: list[str] = []
            if entry.node.source_index is not None:
                instruction = self.view.instructions.get(entry.node.source_index)
                if instruction and instruction.attrs:
                    attr_sources.append(instruction.attrs)
            if entry.node.attrs:
                attr_sources.append(entry.node.attrs)
            if not attr_sources:
                continue
            for attrs_source in attr_sources:
                attrs = _parse_attrs_flat(attrs_source)
                names.update(attrs.keys())
                for key, value in attrs.items():
                    if value is None:
                        continue
                    values_by_key.setdefault(key, set()).add(value)
        options: list[tuple[str, str | None]] = []
        for name in sorted(names, key=str.lower):
            values = values_by_key.get(name)
            peek = next(iter(values)) if values and len(values) == 1 else None
            options.append((name, peek))
        return options

    def action_open_help(self) -> None:
        self.push_screen(HelpScreen(self._help_items()))

    def action_open_report(self) -> None:
        self.push_screen(ReportScreen(self._current_path))

    def _help_items(self) -> list[tuple[str, str]]:
        items: list[tuple[str, str]] = [
            ("j / ↑ / C-p", "Move up"),
            ("k / ↓ / C-n", "Move down"),
            ("PgUp / C-u / M-v", "Page up"),
            ("PgDn / C-d / C-v", "Page down"),
            ("g / Home", "Go to top"),
            ("G / End", "Go to bottom"),
            ("h / C-b", "Scroll left"),
            ("l / C-f", "Scroll right"),
            ("0 / C-a", "Scroll leftmost"),
            ("$ / C-e", "Scroll rightmost"),
            ("Tab / Shift-Tab / C-o", "Toggle focus"),
            ("/", "Search"),
            ("n / C-s", "Next match"),
            ("N / C-r", "Prev match"),
            ("?", "Attribute search"),
            ("f", "Jump section"),
            ("o", "Options"),
            ("i", "Toggle details panel"),
            ("z", "Zoom details panel"),
            ("r", "Histogram report"),
            ("Enter", "Quick jump"),
            ("Space", "Select"),
            ("H", "Help"),
            ("q", "Quit"),
        ]
        return items

    def action_quit(self) -> None:
        self.push_screen(ConfirmQuitScreen(), self._confirm_quit)

    def _confirm_quit(self, confirm: bool | None) -> None:
        if confirm:
            self.exit()

    def action_search_next(self) -> None:
        if not self._last_search:
            return
        if self._last_search_target == "details":
            self._find_next_details(forward=True)
        elif self._last_search_target == "attr" and self._last_attr_search:
            self._find_next_attr(*self._last_attr_search, forward=True)
        else:
            self._find_next(self._last_search, forward=True)

    def action_search_prev(self) -> None:
        if not self._last_search:
            return
        if self._last_search_target == "details":
            self._find_next_details(forward=False)
        elif self._last_search_target == "attr" and self._last_attr_search:
            self._find_next_attr(*self._last_attr_search, forward=False)
        else:
            self._find_next(self._last_search, forward=False)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "search_bar":
            return
        self._apply_search(event.value)
        event.input.display = False
        self.query_one("#list", RichLog).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "search_bar":
            return
        if event.value == "":
            event.input.display = False
            self.query_one("#list", RichLog).focus()

    def on_key(self, event: events.Key) -> None:
        if isinstance(self.screen, ModalScreen):
            return
        search_bar = self.query_one("#search_bar", Input)
        if search_bar.display and event.key == "escape":
            search_bar.display = False
            self.query_one("#list", RichLog).focus()
            event.stop()
            return
        if isinstance(self.focused, Input):
            return
        emacs_actions = {
            "ctrl+n": self.action_move_down,
            "ctrl+p": self.action_move_up,
            "ctrl+v": self.action_page_down,
            "alt+v": self.action_page_up,
            "ctrl+f": self.action_scroll_right,
            "ctrl+b": self.action_scroll_left,
            "ctrl+a": self.action_scroll_leftmost,
            "ctrl+e": self.action_scroll_rightmost,
            "ctrl+s": self.action_search_next,
            "ctrl+r": self.action_search_prev,
            "ctrl+o": self.action_toggle_focus,
        }
        action = emacs_actions.get(event.key)
        if action is not None:
            action()
            event.stop()
            return
        key_actions = {
            "up": self.action_move_up,
            "down": self.action_move_down,
            "pageup": self.action_page_up,
            "pagedown": self.action_page_down,
            "home": self.action_go_home,
            "end": self.action_go_end,
            "tab": self.action_toggle_focus,
            "shift+tab": self.action_toggle_focus_back,
        }
        action = key_actions.get(event.key)
        if action is not None:
            action()
            event.stop()

    def action_open_toggles(self) -> None:
        self.push_screen(ToggleScreen(self.options), self._apply_toggles)

    def action_open_quick_jump(self) -> None:
        if not self.selected_node or self.selected_node.source_index is None:
            return
        instruction = self.view.instructions.get(self.selected_node.source_index)
        if not instruction:
            return
        items: list[tuple[Text, int]] = []
        seen: set[tuple[str, int]] = set()
        line_idx = self.selected_node.source_index
        container_id = self._container_by_line.get(line_idx)
        defs, uses = self._instruction_defs_uses(instruction)
        label_width = 10

        def _make_label(kind: str, label: str, color: str) -> Text:
            prefix = f"{kind:<{label_width}}"
            text = Text(f"{prefix} {label}")
            text.stylize(color, 0, len(prefix))
            return text

        for value in uses:
            alloc = _alloc_for_value(self.document.allocs, value, line_idx)
            parent = alloc.parent if alloc and alloc.parent else None
            key = parent or value
            def_line = self._lookup_def_line(key, line_idx, container_id, exclude_current=True)
            if def_line is None:
                continue
            inst = self.view.instructions.get(def_line)
            label = _instruction_label(
                inst,
                self.options,
                self.document.allocs,
                def_line,
                self.options.show_alloc_sizes,
                self.options.show_left_types,
                self.view.ws_rt_io,
                None,
                self.options.show_source_vars,
                False,
            ) if inst else self.document.lines[def_line].strip()
            input_name = f"{value} (via {key})" if key != value else value
            entry = (_make_label("Input", f"{input_name} ← {label}", "cyan"), def_line)
            if ("input", def_line) not in seen:
                items.append(entry)
                seen.add(("input", def_line))

        for use_line in self.use_by_def.get(line_idx, []):
            inst = self.view.instructions.get(use_line)
            label = _instruction_label(
                inst,
                self.options,
                self.document.allocs,
                use_line,
                self.options.show_alloc_sizes,
                self.options.show_left_types,
                self.view.ws_rt_io,
                None,
                self.options.show_source_vars,
                False,
            ) if inst else self.document.lines[use_line].strip()
            entry = (_make_label("User", label, "magenta"), use_line)
            if ("user", use_line) not in seen:
                items.append(entry)
                seen.add(("user", use_line))

        for def_value in defs:
            for use_line in self.use_by_def.get(line_idx, []):
                inst = self.view.instructions.get(use_line)
                label = _instruction_label(
                    inst,
                    self.options,
                    self.document.allocs,
                    use_line,
                    self.options.show_alloc_sizes,
                    self.options.show_left_types,
                    self.view.ws_rt_io,
                    None,
                    self.options.show_source_vars,
                    False,
                ) if inst else self.document.lines[use_line].strip()
                entry = (_make_label("User", f"{def_value} → {label}", "magenta"), use_line)
                if ("user", use_line) not in seen:
                    items.append(entry)
                    seen.add(("user", use_line))
        for def_value in defs:
            for use_line in self.use_map.get(def_value, []):
                inst = self.view.instructions.get(use_line)
                label = _instruction_label(
                    inst,
                    self.options,
                    self.document.allocs,
                    use_line,
                    self.options.show_alloc_sizes,
                    self.options.show_left_types,
                    self.view.ws_rt_io,
                    None,
                    self.options.show_source_vars,
                    False,
                ) if inst else self.document.lines[use_line].strip()
                entry = (_make_label("User", f"{def_value} → {label}", "magenta"), use_line)
                if ("user", use_line) not in seen:
                    items.append(entry)
                    seen.add(("user", use_line))

        for producer, label in (
            (instruction.semaphore_unblocker, "Producer (sem)"),
            (instruction.num_rx_unblocker, "Producer (data)"),
        ):
            if producer is None:
                continue
            inst = self.view.instructions.get(producer)
            line_label = _instruction_label(
                inst,
                self.options,
                self.document.allocs,
                producer,
                self.options.show_alloc_sizes,
                self.options.show_left_types,
                self.view.ws_rt_io,
                None,
                self.options.show_source_vars,
                False,
            ) if inst else self.document.lines[producer].strip()
            entry = (_make_label("Producer", line_label, "green"), producer)
            if ("producer", producer) not in seen:
                items.append(entry)
                seen.add(("producer", producer))

        for consumer in instruction.semaphore_consumers:
            inst = self.view.instructions.get(consumer)
            line_label = _instruction_label(
                inst,
                self.options,
                self.document.allocs,
                consumer,
                self.options.show_alloc_sizes,
                self.options.show_left_types,
                self.view.ws_rt_io,
                None,
                self.options.show_source_vars,
                False,
            ) if inst else self.document.lines[consumer].strip()
            entry = (_make_label("Consumer", line_label, "yellow"), consumer)
            if ("consumer", consumer) not in seen:
                items.append(entry)
                seen.add(("consumer", consumer))

        for consumer in instruction.num_rx_consumers:
            inst = self.view.instructions.get(consumer)
            line_label = _instruction_label(
                inst,
                self.options,
                self.document.allocs,
                consumer,
                self.options.show_alloc_sizes,
                self.options.show_left_types,
                self.view.ws_rt_io,
                None,
                self.options.show_source_vars,
                False,
            ) if inst else self.document.lines[consumer].strip()
            entry = (_make_label("Consumer", line_label, "yellow"), consumer)
            if ("consumer", consumer) not in seen:
                items.append(entry)
                seen.add(("consumer", consumer))

        if not items:
            return
        order_by_source: dict[int, int] = {}
        for idx, entry in enumerate(self._flat_entries):
            if entry.node.source_index is None:
                continue
            order_by_source.setdefault(entry.node.source_index, idx)
        items.sort(key=lambda item: (order_by_source.get(item[1], 10**9), item[1]))
        self.push_screen(QuickJumpScreen(items), self._apply_quick_jump)

    def _apply_quick_jump(self, line_idx: int | None) -> None:
        if line_idx is None:
            return
        self._jump_to_index(line_idx)

    def action_focus_details(self) -> None:
        details = self.query_one("#details", RichLog)
        details.can_focus = True
        details.focus()
        self._focus_target = "details"

    def action_focus_list(self) -> None:
        list_view = self.query_one("#list", RichLog)
        list_view.can_focus = True
        list_view.focus()
        self._focus_target = "list"

    def action_toggle_focus(self) -> None:
        details = self.query_one("#details", RichLog)
        if not details.display:
            if self.options.split_axis_view:
                if self._split_focus_side == "left":
                    self._switch_split_focus_side("right")
                else:
                    self._switch_split_focus_side("left")
            return
        if not self.options.split_axis_view:
            if self._details_focused():
                self.action_focus_list()
            else:
                self.action_focus_details()
            return

        # Split mode cycle order: left -> right -> details -> left ...
        if self._details_focused():
            self.action_focus_list()
            self._switch_split_focus_side("left")
            return
        if self._split_focus_side == "left":
            self._switch_split_focus_side("right")
            return
        self.action_focus_details()

    def action_toggle_focus_back(self) -> None:
        details = self.query_one("#details", RichLog)
        if not details.display:
            if self.options.split_axis_view:
                if self._split_focus_side == "right":
                    self._switch_split_focus_side("left")
                else:
                    self._switch_split_focus_side("right")
            return
        if not self.options.split_axis_view:
            if self._details_focused():
                self.action_focus_list()
            else:
                self.action_focus_details()
            return

        # Reverse split cycle: left <- right <- details <- left ...
        if self._details_focused():
            self.action_focus_list()
            self._switch_split_focus_side("right")
            return
        if self._split_focus_side == "right":
            self._switch_split_focus_side("left")
            return
        self.action_focus_details()

    def _render_list(self) -> None:
        list_log = self.query_one("#list", RichLog)
        list_log.clear()
        self._jump_sections = []
        self._label_cache.clear()

        root = self.view.build_hierarchy()
        if len(root.children) == 1 and root.children[0].kind == "region" and root.children[0].attrs is None:
            nodes = root.children[0].children
        else:
            nodes = root.children

        if self._active_func_index is not None and 0 <= self._active_func_index < len(self._ws_funcs):
            _, start, end = self._ws_funcs[self._active_func_index]
            nodes = _filter_nodes_by_range(nodes, start, end)

        flat_nodes: list[tuple[Node, int]] = []
        self._flatten_nodes(nodes, 0, flat_nodes)

        self._flat_entries = []
        selected_index = 0
        for idx, (node, depth) in enumerate(flat_nodes):
            if (
                self.selected_node
                and self.selected_node.source_index is not None
                and node.source_index == self.selected_node.source_index
            ):
                selected_index = idx
            if node.kind == "region" and depth == 0:
                self._jump_sections.append((node.label, idx))

        for node, depth in flat_nodes:
            self._flat_entries.append(self._build_line_entry(node, depth))

        self._apply_loc_suffixes()
        self._apply_loc_groups()
        self.view._ensure_semaphore_relationships()

        if self.options.align_left_panel:
            self._apply_alignment()
        else:
            for entry in self._flat_entries:
                prefix = self._line_number_prefix(entry.node)
                group_marker = "│ " if entry.group_bar else ""
                indent = self._indent_prefix(entry.depth + entry.group_level)
                label_string = f"{prefix}{group_marker}{indent}{entry.raw_label}"
                if entry.loc_suffix:
                    label_string = f"{label_string} ‹{entry.loc_suffix}›"
                entry.text = self._styled_label_text(entry.node.kind, entry.node.source_index, label_string)

        self._rebuild_def_use_maps()
        selected_index = 0
        if self.selected_node and self.selected_node.source_index is not None:
            for idx, entry in enumerate(self._flat_entries):
                if entry.node.source_index == self.selected_node.source_index:
                    selected_index = idx
                    break
        self._set_selected_index(selected_index)

    def _update_details(self, node: NodeData) -> None:
        details = self.query_one("#details", RichLog)
        rendered = self.view.details_for_node(node, self.segment_index)
        details.clear()
        text = rendered if isinstance(rendered, Text) else Text(str(rendered))
        if node.kind == "inst" and node.source_index is not None:
            extra_lines: list[str] = []
            instruction = self.view.instructions.get(node.source_index)
            if instruction:
                producer_items: list[tuple[str, str]] = []
                consumer_items: list[tuple[str, str]] = []
                sem_source = instruction.semaphore_unblocker
                if sem_source is not None:
                    inst = self.view.instructions.get(sem_source)
                    if inst:
                        label = _instruction_label(
                            inst,
                            self.options,
                            self.document.allocs,
                            sem_source,
                            self.options.show_alloc_sizes,
                            self.options.show_left_types,
                            self.view.ws_rt_io,
                            None,
                            self.options.show_source_vars,
                            False,
                        )
                        producer_items.append(("S", label))
                for consumer_idx in instruction.semaphore_consumers:
                    inst = self.view.instructions.get(consumer_idx)
                    if not inst:
                        continue
                    label = _instruction_label(
                        inst,
                        self.options,
                        self.document.allocs,
                        consumer_idx,
                        self.options.show_alloc_sizes,
                        self.options.show_left_types,
                        self.view.ws_rt_io,
                        None,
                        self.options.show_source_vars,
                        False,
                    )
                    consumer_items.append(("S", label))
                num_rx_source = instruction.num_rx_unblocker
                if num_rx_source is not None:
                    inst = self.view.instructions.get(num_rx_source)
                    if inst:
                        label = _instruction_label(
                            inst,
                            self.options,
                            self.document.allocs,
                            num_rx_source,
                            self.options.show_alloc_sizes,
                            self.options.show_left_types,
                            self.view.ws_rt_io,
                            None,
                            self.options.show_source_vars,
                            False,
                        )
                        producer_items.append(("D", label))
                for consumer_idx in instruction.num_rx_consumers:
                    inst = self.view.instructions.get(consumer_idx)
                    if not inst:
                        continue
                    label = _instruction_label(
                        inst,
                        self.options,
                        self.document.allocs,
                        consumer_idx,
                        self.options.show_alloc_sizes,
                        self.options.show_left_types,
                        self.view.ws_rt_io,
                        None,
                        self.options.show_source_vars,
                        False,
                    )
                    consumer_items.append(("D", label))
                if producer_items:
                    extra_lines.append("Producer:")
                    for kind, label in producer_items:
                        extra_lines.append(f"\u200b{kind}  - {label}")
                if consumer_items:
                    extra_lines.append("Consumers:")
                    for kind, label in consumer_items:
                        extra_lines.append(f"\u200b{kind}  - {label}")
            if extra_lines:
                lines = text.plain.splitlines()
                insert_at = 3 if len(lines) >= 3 else len(lines)
                for offset, line in enumerate(extra_lines):
                    lines.insert(insert_at + offset, line)
                text = _highlight_details("\n".join(lines), self.options.highlight_non_simple_srctgt)
        details.write(text)
        self._details_lines = text.plain.splitlines()

    def _build_line_entry(self, node: Node, depth: int) -> "LineEntry":
        data = NodeData(
            kind=node.kind,
            source_index=node.source_index,
            attrs=node.attrs,
            value=node.value,
            layout_index=node.layout_index,
        )
        return LineEntry(node=data, depth=depth, raw_label=node.label, text=Text(""))

    def _apply_loc_groups(self) -> None:
        if not self.options.show_left_loc or not self.options.group_loc_prefixes:
            for entry in self._flat_entries:
                entry.group_bar = False
                entry.group_level = 0
                entry.group_prefix = None
                entry.group_chain = []
            return
        prefixes_by_entry: list[list[str]] = [
            _loc_group_chain(
                _group_loc_suffix(
                    entry.loc_group_suffix,
                    entry.loc_prefix_base,
                    self._display_inst_name(entry.node.source_index) if entry.node.source_index is not None else None,
                    self._display_inst_base_name(entry.node.source_index) if entry.node.source_index is not None else None,
                )
            )
            if entry.node.kind == "inst" and entry.node.source_index is not None
            else []
            for entry in self._flat_entries
        ]
        max_level = max((len(p) for p in prefixes_by_entry), default=0)
        level_enabled: list[set[int]] = [set() for _ in self._flat_entries]
        for level in range(1, max_level + 1):
            prefixes = [p[level - 1] if len(p) >= level else None for p in prefixes_by_entry]
            idx = 0
            while idx < len(prefixes):
                current = prefixes[idx]
                if current is None:
                    idx += 1
                    continue
                end = idx + 1
                while end < len(prefixes) and prefixes[end] == current:
                    end += 1
                if end - idx >= 2:
                    for i in range(idx, end):
                        level_enabled[i].add(level)
                idx = end
        child_map: dict[str, set[str]] = {}
        for chain in prefixes_by_entry:
            for idx, parent in enumerate(chain):
                child = chain[idx + 1] if idx + 1 < len(chain) else "__leaf__"
                child_map.setdefault(parent, set()).add(child)
        prefixes_to_skip = {
            parent
            for parent, children in child_map.items()
            if len(children) == 1 and "__leaf__" not in children
        }

        new_entries: list[LineEntry] = []
        current_chain: list[str] = []
        for entry, chain, enabled_levels in zip(self._flat_entries, prefixes_by_entry, level_enabled):
            eligible_chain_raw = [chain[i - 1] for i in sorted(enabled_levels) if i - 1 < len(chain)]
            eligible_chain = [p for p in eligible_chain_raw if p not in prefixes_to_skip]
            common = 0
            while common < len(current_chain) and common < len(eligible_chain) and current_chain[common] == eligible_chain[common]:
                common += 1
            current_chain = current_chain[:common]
            last_emitted = current_chain[-1] if current_chain else ""
            for prefix in eligible_chain[common:]:
                level = len(current_chain)
                current_chain.append(prefix)
                if last_emitted and prefix.startswith(last_emitted + "."):
                    label = prefix[len(last_emitted) + 1 :]
                else:
                    label = prefix
                last_emitted = prefix
                group_node = NodeData(kind="loc_group", source_index=None, attrs=None, value=None, layout_index=None)
                group_entry = LineEntry(
                    node=group_node,
                    depth=entry.depth,
                    raw_label=f"▸ {label}",
                    text=Text(""),
                    loc_suffix=None,
                    group_bar=False,
                    group_level=level,
                    group_prefix=prefix,
                )
                new_entries.append(group_entry)
            entry.group_bar = bool(eligible_chain)
            entry.group_level = len(eligible_chain)
            entry.group_chain = eligible_chain
            if eligible_chain:
                group_prefix = eligible_chain[-1]
                base_prefix = entry.loc_prefix_base or ""
                if base_prefix and group_prefix.startswith(base_prefix + "."):
                    group_prefix = group_prefix[len(base_prefix) + 1 :]
                entry.loc_suffix = _strip_group_prefix(entry.loc_suffix, group_prefix)
                if entry.node.source_index is not None and entry.loc_suffix:
                    if self._loc_suffix_matches_inst(entry.node.source_index, entry.loc_suffix):
                        entry.loc_suffix = None
            new_entries.append(entry)
        self._flat_entries = new_entries

    def _styled_label_text(self, kind: str, source_index: int | None, label_string: str) -> Text:
        if kind == "loc_group":
            label_text = Text(label_string)
            label_text.stylize("grey58", 0, len(label_text))
            return label_text
        if kind == "inst" and source_index is not None:
            cache_key = (source_index, label_string)
            cached = self._label_cache.get(cache_key)
            if cached is not None:
                return cached.copy()

        label_text = Text(label_string)
        if kind == "inst" and source_index is not None:
            instruction = self.view.instructions.get(source_index)
            if instruction:
                inst_name = instruction.inst
                if not self.options.show_full_prefix and inst_name.startswith(self.options.shorten_prefix):
                    inst_name = inst_name[len(self.options.shorten_prefix) :]
                if not self.options.show_full_prefix:
                    inst_name = re.sub(r"\bmaster_(tx|rx)\b", r"\1", inst_name)
                if instruction.attrs:
                    for match in re.finditer(r"\bonX\b", label_string):
                        label_text.stylize("green", match.start(), match.end())
                    for match in re.finditer(r"\bonY\b", label_string):
                        label_text.stylize("yellow", match.start(), match.end())
                idx = label_string.find(inst_name)
                if idx != -1:
                    label_text.stylize("bold cyan", idx, idx + len(inst_name))
                handle_match = re.search(r"\[[^\]]+\]", label_string)
                if handle_match:
                    label_text.stylize("grey62", handle_match.start(), handle_match.end())
                for match in _TYPE_RE.finditer(label_string):
                    token = match.group(1)
                    if not _looks_like_type(token):
                        continue
                    start = match.start(1)
                    end = match.end(1)
                    label_text.stylize("magenta", start, end)
                for match in _ALLOC_SUFFIX_RE.finditer(label_string):
                    suffix_start = match.start() + match.group(0).find(".")
                    suffix_end = match.end()
                    label_text.stylize("orange1", suffix_start, suffix_end)
                for match in _SIZE_RE.finditer(label_string):
                    label_text.stylize("green", match.start(), match.end())
                for match in _SOURCE_VARS_RE.finditer(label_string):
                    label_text.stylize("violet", match.start(), match.end())
                type_sep = label_string.find(" : ")
                if type_sep != -1:
                    label_text.stylize("magenta", type_sep + 3, len(label_string))
                for match in re.finditer(r"(?<!\S)([^\s]+)\s+%[A-Za-z0-9_$.]+", label_string):
                    token = match.group(1)
                    if not _looks_like_type(token):
                        continue
                    type_start = match.start(1)
                    type_end = match.end(1)
                    label_text.stylize("magenta", type_start, type_end)
                sem_value = _semaphore_value(instruction.attrs)
                if sem_value:
                    sem_text = f"→ ({sem_value} tx)"
                    sem_idx = label_string.find(sem_text)
                    if sem_idx != -1:
                        label_text.stylize("dodger_blue1", sem_idx, sem_idx + len(sem_text))
                num_rx_value = _num_rx_value(instruction.attrs)
                op_value = _op_attr_value(instruction.attrs)
                inst_tail = instruction.inst.split(".")[-1]
                is_io_tx = (
                    op_value
                    and op_value.startswith("io")
                    and inst_tail in {"tx", "txact", "request_txact", "master_tx"}
                )
                if is_io_tx:
                    num_rx_value = None
                elif not num_rx_value:
                    if (
                        op_value
                        and (op_value.startswith("slave") or op_value.startswith("switchroot"))
                        and not op_value.startswith("slaveCacheStore")
                        and inst_tail in {
                        "tx",
                        "txact",
                        "request_txact",
                        "master_tx",
                        }
                    ):
                        num_rx_value = "1"
                    elif (
                        op_value
                        and not op_value.startswith("slave")
                        and not op_value.startswith("switchroot")
                        and not op_value.startswith("none")
                        and not op_value.startswith("io")
                        and inst_tail in {
                        "tx",
                        "txact",
                        "request_txact",
                        "master_tx",
                        }
                    ):
                        num_rx_value = "1"
                if num_rx_value:
                    rx_text = f"→ ({num_rx_value} rx)"
                    rx_idx = label_string.find(rx_text)
                    if rx_idx != -1:
                        label_text.stylize("yellow", rx_idx, rx_idx + len(rx_text))
                self._label_cache[(source_index, label_string)] = label_text
        return label_text

    def _indent_prefix(self, depth: int) -> str:
        if depth <= 0:
            return ""
        return "  " * depth

    def _line_number_prefix(self, node: NodeData) -> str:
        if not self.options.show_line_numbers:
            return ""
        if node.source_index is None:
            return ""
        return f"{node.source_index + 1:5d} | "

    def _flatten_nodes(self, nodes: list[Node], depth: int, out: list[tuple[Node, int]]) -> None:
        for node in nodes:
            if self.options.show_only_tx_rx:
                if node.kind == "inst" and node.source_index is not None:
                    instruction = self.view.instructions.get(node.source_index)
                    if instruction:
                        # Check if this instruction produces semaphore or num_rx
                        sem_value = _semaphore_value(instruction.attrs)
                        sem_count = _parse_semaphore_count(sem_value)
                        num_rx_value = _num_rx_value(instruction.attrs)
                        num_rx_count = _parse_semaphore_count(num_rx_value)
                        op_value = _op_attr_value(instruction.attrs)
                        inst_tail = instruction.inst.split(".")[-1]
                        is_io_tx = (
                            op_value
                            and op_value.startswith("io")
                            and inst_tail in {"tx", "txact", "request_txact", "master_tx"}
                        )
                        if is_io_tx:
                            num_rx_count = 0
                        
                        # Check default num_rx for slave tx ops and non-slave, non-none tx ops
                        if not num_rx_count:
                            if (
                                op_value
                                and (op_value.startswith("slave") or op_value.startswith("switchroot"))
                                and not op_value.startswith("slaveCacheStore")
                                and inst_tail in {"tx", "txact", "request_txact", "master_tx"}
                            ):
                                num_rx_count = 1
                            elif (
                                op_value
                                and not op_value.startswith("slave")
                                and not op_value.startswith("switchroot")
                                and not op_value.startswith("none")
                                and not op_value.startswith("io")
                                and inst_tail in {"tx", "txact", "request_txact", "master_tx"}
                            ):
                                num_rx_count = 1
                        
                        # Check if this instruction consumes semaphore or num_rx
                        is_producer = (sem_count and sem_count > 0) or (num_rx_count and num_rx_count > 0)
                        is_consumer = (
                            instruction.semaphore_unblocker is not None
                            or instruction.num_rx_unblocker is not None
                        )
                        
                        if not (is_producer or is_consumer):
                            continue
                    out.append((node, depth))
                else:
                    if node.children:
                        self._flatten_nodes(node.children, depth, out)
                continue
            out.append((node, depth))
            if node.children:
                self._flatten_nodes(node.children, depth + 1, out)

    def _apply_alignment(self) -> None:
        grouped: dict[int, list[LineEntry]] = {}
        for entry in self._flat_entries:
            grouped.setdefault(entry.depth, []).append(entry)

        for depth, entries in grouped.items():
            parts_list: list[dict[str, str] | None] = []
            widths = {"handle": 0, "outputs": 0, "inst": 0, "inputs": 0, "const": 0}
            for entry in entries:
                if entry.node.kind != "inst":
                    parts_list.append(None)
                    continue
                parts = self._split_instruction_columns(entry.raw_label)
                parts_list.append(parts)
                if parts:
                    widths["handle"] = max(widths["handle"], len(parts["handle"]))
                    widths["outputs"] = max(widths["outputs"], len(parts["outputs"]))
                    widths["inst"] = max(widths["inst"], len(parts["inst"]))
                    widths["inputs"] = max(widths["inputs"], len(parts["inputs"]))
                    widths["const"] = max(widths["const"], len(parts["const"]))

            for entry, parts in zip(entries, parts_list):
                prefix = self._line_number_prefix(entry.node)
                group_marker = "│ " if entry.group_bar else ""
                indent = self._indent_prefix(entry.depth + entry.group_level)
                if not parts:
                    label_string = f"{prefix}{group_marker}{indent}{entry.raw_label}"
                    entry.text = self._styled_label_text(entry.node.kind, entry.node.source_index, label_string)
                    continue
                label_string = self._aligned_label_string(parts, widths, f"{prefix}{group_marker}", indent, entry.loc_suffix)
                entry.text = self._styled_label_text(entry.node.kind, entry.node.source_index, label_string)

    def _split_instruction_columns(self, label: str) -> dict[str, str] | None:
        if not label:
            return None
        source = ""
        base = label
        if " ⟨" in label:
            base, source_tail = label.split(" ⟨", 1)
            source = "⟨" + source_tail
        base = base.strip()
        handle = ""
        if base.startswith("["):
            close_idx = base.find("]")
            if close_idx != -1:
                handle = base[: close_idx + 1]
                base = base[close_idx + 1 :].lstrip()
        outputs = ""
        eq = ""
        rest = base
        if " = " in base:
            outputs, rest = base.split(" = ", 1)
            eq = "="
        rest = rest.strip()
        inst = rest
        inputs = ""
        if rest:
            parts = rest.split(" ", 1)
            inst = parts[0]
            inputs = parts[1] if len(parts) > 1 else ""
        const = ""
        if inputs:
            match = re.match(r"^(.*)\s+\[([^\]]+)\]$", inputs)
            if match:
                inputs = match.group(1)
                const = f"[{match.group(2)}]"
        return {
            "handle": handle,
            "outputs": outputs,
            "eq": eq,
            "inst": inst,
            "inputs": inputs,
            "const": const,
            "source": source,
        }

    def _aligned_label_string(
        self,
        parts: dict[str, str],
        widths: dict[str, int],
        prefix: str,
        indent: str,
        loc_suffix: str | None,
    ) -> str:
        columns = [
            parts["handle"].ljust(widths["handle"]),
            parts["outputs"].ljust(widths["outputs"]),
            parts["eq"].ljust(1),
            parts["inst"].ljust(widths["inst"]),
            parts["inputs"].ljust(widths["inputs"]),
            parts["const"].ljust(widths["const"]),
        ]
        base = " ".join(columns).rstrip()
        if parts["source"]:
            if base:
                base = f"{base} {parts['source']}"
            else:
                base = parts["source"]
        if loc_suffix:
            if base:
                base = f"{base} ‹{loc_suffix}›"
            else:
                base = f"‹{loc_suffix}›"
        return f"{prefix}{indent}{base}"

    def _apply_loc_suffixes(self) -> None:
        if not self._flat_entries:
            return
        if not self.options.show_left_loc:
            for entry in self._flat_entries:
                entry.loc_suffix = None
                entry.loc_group_suffix = None
                entry.loc_prefix_base = None
            return
        loc_by_line: dict[int, str] = {}
        by_container: dict[str, list[str]] = {}
        for entry in self._flat_entries:
            if entry.node.source_index is None:
                continue
            container_id = self._loc_container_id_for_line(entry.node.source_index)
            loc_name = self._loc_name_for_line(entry.node.source_index)
            if not loc_name:
                continue
            loc_by_line[entry.node.source_index] = loc_name
            by_container.setdefault(container_id, []).append(loc_name)

        prefix_by_container: dict[str, str] = {}
        for container_id, names in by_container.items():
            prefix = _trim_prefix(_common_prefix(names)) if names else ""
            prefix_by_container[container_id] = prefix

        for entry in self._flat_entries:
            if entry.node.source_index is None:
                entry.loc_suffix = None
                entry.loc_group_suffix = None
                entry.loc_prefix_base = None
                continue
            container_id = self._loc_container_id_for_line(entry.node.source_index)
            loc_name = loc_by_line.get(entry.node.source_index)
            if not loc_name:
                entry.loc_suffix = None
                entry.loc_group_suffix = None
                entry.loc_prefix_base = None
                continue
            prefix = prefix_by_container.get(container_id, "")
            suffix = _strip_prefix(loc_name, prefix)
            if self._loc_suffix_matches_inst(entry.node.source_index, suffix):
                entry.loc_suffix = None
            else:
                entry.loc_suffix = suffix
            entry.loc_group_suffix = suffix
            entry.loc_prefix_base = prefix

    def _loc_container_id_for_line(self, line_idx: int) -> str:
        container_id = self._container_by_line.get(line_idx)
        if container_id is not None:
            return container_id
        for func_label, start, end in self._ws_funcs:
            if start <= line_idx <= end:
                return f"func:{func_label}"
        return "__global__"

    def _loc_name_for_line(self, line_idx: int) -> str | None:
        def _extract_from_text(text: str) -> str | None:
            inline_loc_match = re.search(r"loc\(\"([^\"]+)\"\)", text)
            if inline_loc_match:
                full = inline_loc_match.group(1)
                first = full.split(",", 1)[0].strip()
                return first or None
            match = _LOC_REF_RE.search(text)
            if not match:
                return None
            loc_def = self.document.locs.get(match.group(1))
            if not loc_def:
                return None
            loc_match = re.search(r"loc\(\"([^\"]+)\"\)", loc_def.text)
            if not loc_match:
                return None
            full = loc_match.group(1)
            first = full.split(",", 1)[0].strip()
            return first or None

        instruction = self.view.instructions.get(line_idx)
        if instruction and instruction.loc_ref:
            loc_def = self.document.locs.get(instruction.loc_ref)
            if loc_def:
                found = _extract_from_text(loc_def.text)
                if found:
                    return found

        line = self.document.lines[line_idx]
        found = _extract_from_text(line)
        if found:
            return found

        # SSA ops may place loc(...) on a following continuation line.
        for look_ahead in range(1, 21):
            idx = line_idx + look_ahead
            if idx >= len(self.document.lines):
                break
            next_line = self.document.lines[idx].strip()
            if not next_line:
                continue
            found = _extract_from_text(next_line)
            if found:
                return found
        return None

    def _loc_suffix_matches_inst(self, line_idx: int, loc_suffix: str) -> bool:
        inst_name = self._display_inst_name(line_idx)
        if not inst_name:
            return False
        return loc_suffix == inst_name

    def _display_inst_name(self, line_idx: int) -> str | None:
        instruction = self.view.instructions.get(line_idx)
        if not instruction:
            return None
        inst_name = instruction.inst
        op_suffix = _op_attr_value(instruction.attrs)
        if op_suffix:
            inst_name = f"{inst_name}.{op_suffix}"
        if not self.options.show_full_prefix and inst_name.startswith(self.options.shorten_prefix):
            inst_name = inst_name[len(self.options.shorten_prefix) :]
        if not self.options.show_full_prefix:
            inst_name = re.sub(r"\bmaster_(tx|rx)\b", r"\1", inst_name)
        return inst_name

    def _display_inst_base_name(self, line_idx: int) -> str | None:
        instruction = self.view.instructions.get(line_idx)
        if not instruction:
            return None
        inst_name = instruction.inst
        if not self.options.show_full_prefix and inst_name.startswith(self.options.shorten_prefix):
            inst_name = inst_name[len(self.options.shorten_prefix) :]
        if not self.options.show_full_prefix:
            inst_name = re.sub(r"\bmaster_(tx|rx)\b", r"\1", inst_name)
        return inst_name

    def _viewport_size(self) -> int:
        list_log = self.query_one("#list", RichLog)
        height = list_log.size.height if list_log.is_attached else 40
        return max(10, height - 2)

    def _styled_entry_line(self, entry: LineEntry, selected_chain: set[str]) -> Text:
        line = entry.text.copy()
        if entry.node.kind == "loc_group":
            if entry.group_prefix and entry.group_prefix in selected_chain:
                line.stylize("on rgb(60,60,80)", 0, len(line))
            return line
        if entry.node.source_index in self._highlight_level1:
            line.stylize("on rgb(100,170,100)", 0, len(line))
        elif entry.node.source_index in self._highlight_level2:
            line.stylize("on rgb(70,120,70)", 0, len(line))
        elif entry.node.source_index in self._highlight_level3:
            line.stylize("on rgb(50,85,50)", 0, len(line))
        if entry.node.source_index == self._highlight_semaphore:
            line.stylize("on rgb(130,50,130)", 0, len(line))
        if entry.node.source_index == self._highlight_num_rx:
            line.stylize("on rgb(120,95,60)", 0, len(line))
        if entry.node.source_index in self._highlight_semaphore_consumers:
            line.stylize("on rgb(50,95,150)", 0, len(line))
        if entry.node.source_index in self._highlight_num_rx_consumers:
            line.stylize("on rgb(160,130,50)", 0, len(line))
        for match in re.finditer(r"‹[^›]+›", line.plain):
            line.stylize("light_green", match.start(), match.end())
        return line

    def _entry_axis_side(self, entry: LineEntry) -> str:
        if entry.node.kind == "inst" and entry.node.source_index is not None:
            instruction = self.view.instructions.get(entry.node.source_index)
            axis = _axis_from_attrs(instruction.attrs) if instruction else None
            if axis == "X":
                return "left"
            return "right"
        return "left"

    def _pad_cell(self, cell: Text, width: int) -> Text:
        padded = cell.copy()
        if len(padded.plain) > width:
            padded.truncate(width)
        pad_len = width - len(padded.plain)
        if pad_len > 0:
            padded.append(" " * pad_len)
        return padded

    def _split_view_visible(self, entry: LineEntry) -> bool:
        if entry.node.kind != "inst" or entry.node.source_index is None:
            return False
        instruction = self.view.instructions.get(entry.node.source_index)
        if not instruction:
            return False
        inst_tail = instruction.inst.split(".")[-1]
        return inst_tail in {"master_tx", "master_rx", "slave_reconfig"}

    def _compute_split_rows(self) -> tuple[list[int], dict[int, dict[str, int]]]:
        row_for_entry: list[int] = [-1] * len(self._flat_entries)
        row_for_source: dict[int, int] = {}
        row_to_entries: dict[int, dict[str, int]] = {}
        last_row_left = -1
        last_row_right = -1
        for idx, entry in enumerate(self._flat_entries):
            if not self._split_view_visible(entry):
                continue
            side = self._entry_axis_side(entry)
            min_row = (last_row_left + 1) if side == "left" else (last_row_right + 1)
            if entry.node.kind == "inst" and entry.node.source_index is not None:
                instruction = self.view.instructions.get(entry.node.source_index)
                if instruction:
                    for pred in (instruction.semaphore_unblocker, instruction.num_rx_unblocker):
                        if pred is None:
                            continue
                        pred_row = row_for_source.get(pred)
                        if pred_row is not None:
                            min_row = max(min_row, pred_row + 1)
            row_for_entry[idx] = min_row
            row_to_entries.setdefault(min_row, {})[side] = idx
            if entry.node.kind == "inst" and entry.node.source_index is not None:
                row_for_source[entry.node.source_index] = min_row
            if side == "left":
                last_row_left = min_row
            else:
                last_row_right = min_row
        return row_for_entry, row_to_entries

    def _switch_split_focus_side(self, side: str) -> None:
        if side not in {"left", "right"}:
            return
        if not self._flat_entries:
            return
        self._split_focus_side = side
        row_for_entry, row_to_entries = self._compute_split_rows()
        if not row_to_entries:
            return
        current_row = row_for_entry[self._selected_index] if row_for_entry else -1
        if current_row < 0:
            current_row = min(row_to_entries.keys(), default=0)
        entry_map = row_to_entries.get(current_row, {})
        target_idx = entry_map.get(side)
        if target_idx is None:
            # Find nearest row with an entry on this side
            rows = sorted(row_to_entries.keys())
            for row in rows:
                if row >= current_row and side in row_to_entries[row]:
                    target_idx = row_to_entries[row][side]
                    break
            if target_idx is None:
                for row in reversed(rows):
                    if row < current_row and side in row_to_entries[row]:
                        target_idx = row_to_entries[row][side]
                        break
        if target_idx is not None:
            self._set_selected_index(target_idx)

    def _render_viewport_split(self) -> None:
        list_log = self.query_one("#list", RichLog)
        list_log.clear()
        if not self._flat_entries:
            return
        selected_chain: set[str] = set()
        if 0 <= self._selected_index < len(self._flat_entries):
            selected_chain = set(self._flat_entries[self._selected_index].group_chain)
        row_for_entry, row_to_entries = self._compute_split_rows()
        selected_row = row_for_entry[self._selected_index] if row_for_entry else 0
        if selected_row < 0:
            selected_row = min(row_to_entries.keys(), default=0)
        selected_entry = row_to_entries.get(selected_row, {}).get(self._split_focus_side)
        if selected_entry is None:
            other_side = "right" if self._split_focus_side == "left" else "left"
            selected_entry = row_to_entries.get(selected_row, {}).get(other_side)
        window_size = self._viewport_size()
        max_row = max(row_to_entries.keys(), default=0)
        self._window_start = _clamp_window_start(selected_row, max_row + 1, window_size)
        end = min(max_row + 1, self._window_start + window_size)
        list_width = list_log.size.width if list_log.is_attached else 120
        sep = " │ "
        col_width = max(10, (list_width - len(sep)) // 2)
        for row in range(self._window_start, end):
            entry_map = row_to_entries.get(row, {})
            left_cell = Text("")
            right_cell = Text("")
            left_idx = entry_map.get("left")
            right_idx = entry_map.get("right")
            if left_idx is not None:
                left_cell = self._styled_entry_line(self._flat_entries[left_idx], selected_chain)
            if right_idx is not None:
                right_cell = self._styled_entry_line(self._flat_entries[right_idx], selected_chain)
            left_cell = self._pad_cell(left_cell, col_width)
            right_cell = self._pad_cell(right_cell, col_width)
            line = Text.assemble(left_cell, sep, right_cell)
            if row == selected_row and selected_entry is not None:
                if selected_entry == left_idx:
                    line.stylize("reverse", 0, col_width)
                elif selected_entry == right_idx:
                    start = col_width + len(sep)
                    line.stylize("reverse", start, start + col_width)
            list_log.write(line)

    def _render_viewport(self) -> None:
        if self.options.split_axis_view:
            self._render_viewport_split()
            return
        list_log = self.query_one("#list", RichLog)
        list_log.clear()
        if not self._flat_entries:
            return
        selected_chain: set[str] = set()
        if 0 <= self._selected_index < len(self._flat_entries):
            selected_chain = set(self._flat_entries[self._selected_index].group_chain)
        window_size = self._viewport_size()
        self._window_start = _clamp_window_start(self._selected_index, len(self._flat_entries), window_size)
        end = min(len(self._flat_entries), self._window_start + window_size)
        for idx in range(self._window_start, end):
            entry = self._flat_entries[idx]
            line = self._styled_entry_line(entry, selected_chain)
            if idx == self._selected_index:
                line.stylize("reverse", 0, len(line))
            list_log.write(line)

    def _set_selected_index(self, index: int) -> None:
        if not self._flat_entries:
            return
        self._selected_index = max(0, min(index, len(self._flat_entries) - 1))
        self._render_viewport()
        self._schedule_apply_selection()

    def _apply_selection(self, apply_highlights: bool = False) -> None:
        if not self._flat_entries:
            return
        entry = self._flat_entries[self._selected_index]
        self.selected_node = entry.node
        self.segment_index = 0
        self._update_details(entry.node)
        if apply_highlights:
            self._update_related_highlights(entry.node)
        else:
            prev_semaphore = self._highlight_semaphore
            prev_num_rx = self._highlight_num_rx
            prev_sem_consumers = set(self._highlight_semaphore_consumers)
            prev_num_rx_consumers = set(self._highlight_num_rx_consumers)
            if entry.node.kind == "inst" and entry.node.source_index is not None:
                instruction = self.view.instructions.get(entry.node.source_index)
                if instruction:
                    self._highlight_semaphore = instruction.semaphore_unblocker
                    self._highlight_num_rx = instruction.num_rx_unblocker
                    self._highlight_semaphore_consumers = set(instruction.semaphore_consumers)
                    self._highlight_num_rx_consumers = set(instruction.num_rx_consumers)
                else:
                    self._highlight_semaphore = None
                    self._highlight_num_rx = None
                    self._highlight_semaphore_consumers = set()
                    self._highlight_num_rx_consumers = set()
            else:
                self._highlight_semaphore = None
                self._highlight_num_rx = None
                self._highlight_semaphore_consumers = set()
                self._highlight_num_rx_consumers = set()
            if self._highlight_level1 or self._highlight_level2 or self._highlight_level3:
                self._highlight_level1 = set()
                self._highlight_level2 = set()
                self._highlight_level3 = set()
            if (
                prev_semaphore != self._highlight_semaphore
                or prev_num_rx != self._highlight_num_rx
                or prev_sem_consumers != self._highlight_semaphore_consumers
                or prev_num_rx_consumers != self._highlight_num_rx_consumers
                or self._highlight_level1
                or self._highlight_level2
                or self._highlight_level3
            ):
                self._render_viewport()

    def _move_selection(self, delta: int) -> None:
        if not self._flat_entries:
            return
        if self.options.split_axis_view:
            row_for_entry, row_to_entries = self._compute_split_rows()
            if not row_for_entry:
                return
            current_row = row_for_entry[self._selected_index]
            if current_row < 0:
                current_row = min(row_to_entries.keys(), default=0)
            max_row = max(row_to_entries.keys(), default=current_row)
            target_row = max(0, min(max_row, current_row + delta))
            if target_row == current_row:
                return
            step = 1 if target_row > current_row else -1
            target_idx = None
            row = target_row
            while 0 <= row <= max_row:
                entry_map = row_to_entries.get(row, {})
                target_idx = entry_map.get(self._split_focus_side)
                if target_idx is not None:
                    break
                row += step
            if target_idx is None:
                return
            self._set_selected_index(target_idx)
            return
        self._set_selected_index(self._selected_index + delta)

    def _schedule_apply_selection(self) -> None:
        if self._select_timer is not None:
            self._select_timer.stop()
        self._select_timer = self.set_timer(0.15, self._apply_selection)

    def _jump_to_index(self, index: int | None) -> None:
        if index is None or not self._flat_entries:
            return
        target = None
        for idx, entry in enumerate(self._flat_entries):
            if entry.node.source_index == index:
                target = idx
                break
        if target is None:
            return
        if self.options.split_axis_view:
            self._split_focus_side = self._entry_axis_side(self._flat_entries[target])
        self._set_selected_index(target)

    def _select_func(self, selection: int | None) -> None:
        if selection is None:
            return
        self._active_func_index = selection
        self.selected_node = NodeData(kind="inst", source_index=0, attrs=None, value=None, layout_index=None)
        self.segment_index = 0
        self._render_list()

    def _load_path(self, path: Path) -> None:
        self._current_path = path
        self.document = Document.from_path(path)
        self.view = DocumentView(self.document, self.options)
        self._ws_funcs = _collect_ws_funcs(self.document.lines)
        self._active_func_index = None
        self._container_by_line = _collect_host_container_by_line(self.document.lines)
        self._def_container_by_line = {}
        self._highlight_level1 = set()
        self._highlight_level2 = set()
        self._highlight_level3 = set()
        self._semaphore_unblocker_by_tx = {}
        self._highlight_semaphore = None
        self._num_rx_unblocker_by_rx = {}
        self._highlight_num_rx = None
        self._semaphore_consumers_by_source = {}
        self._num_rx_consumers_by_source = {}
        self._highlight_semaphore_consumers = set()
        self._highlight_num_rx_consumers = set()
        self._label_cache = {}
        self._flat_entries = []
        self._selected_index = 0
        self._window_start = 0
        self._details_lines = []
        self._details_matches = []
        self._details_match_pos = -1
        self.selected_node = NodeData(kind="inst", source_index=0, attrs=None, value=None, layout_index=None)

    def _apply_toggles(self, values: dict[str, bool] | None) -> None:
        if values is None:
            return
        self.options.show_alloc_free = values.get("alloc_free", self.options.show_alloc_free)
        self.options.show_full_prefix = values.get("full_prefix", self.options.show_full_prefix)
        self.options.show_line_numbers = values.get("line_numbers", self.options.show_line_numbers)
        self.options.show_alloc_sizes = values.get("alloc_sizes", self.options.show_alloc_sizes)
        self.options.show_source_vars = values.get("show_source_vars", self.options.show_source_vars)
        self.options.show_full_source_vars = values.get("full_source_vars", self.options.show_full_source_vars)
        self.options.show_left_types = values.get("show_left_types", self.options.show_left_types)
        self.options.wrap_left_panel = values.get("wrap_left_panel", self.options.wrap_left_panel)
        self.options.emacs_mode = values.get("emacs_mode", self.options.emacs_mode)
        self.options.group_loc_prefixes = values.get("group_loc_prefixes", self.options.group_loc_prefixes)
        self.options.align_left_panel = values.get("align_left_panel", self.options.align_left_panel)
        self.options.show_left_loc = values.get("show_left_loc", self.options.show_left_loc)
        self.options.show_only_tx_rx = values.get("show_only_tx_rx", self.options.show_only_tx_rx)
        self.options.split_axis_view = values.get("split_axis_view", self.options.split_axis_view)
        self.options.highlight_non_simple_srctgt = values.get(
            "highlight_non_simple_srctgt", self.options.highlight_non_simple_srctgt
        )
        list_view = self.query_one("#list", RichLog)
        list_view.wrap = False if self.options.split_axis_view else self.options.wrap_left_panel
        self._render_list()

    def _apply_search(self, query: str | None) -> None:
        if not query:
            return
        query = query.strip()
        if query.startswith("/"):
            query = query[1:]
        if not query:
            return
        self._last_search = query
        if self._focus_target == "details":
            self._last_search_target = "details"
            self._prepare_details_matches(query)
            self._find_next_details(forward=True)
        else:
            self._last_search_target = "list"
            self._find_next(query, forward=True)

    def _apply_attr_search(self, payload: dict[str, str | None] | None) -> None:
        if not payload:
            return
        attr_value = payload.get("attr")
        attr = str(attr_value).strip() if attr_value else ""
        value = (payload.get("value") or "").strip()
        if not attr:
            return
        self._last_search_target = "attr"
        self._last_search = attr
        self._last_attr_search = (attr, value or None)
        self._find_next_attr(attr, value or None, forward=True)

    def _find_next(self, query: str, forward: bool = True) -> None:
        if not self._flat_entries:
            return
        q = query.lower()
        step = 1 if forward else -1
        start = self._selected_index + step
        total = len(self._flat_entries)
        for offset in range(total):
            idx = (start + offset * step) % total
            entry = self._flat_entries[idx]
            if q in entry.raw_label.lower():
                self._set_selected_index(idx)
                return

    def _find_next_attr(self, attr: str, value: str | None, forward: bool = True) -> None:
        if not self._flat_entries:
            return
        step = 1 if forward else -1
        start = self._selected_index + step
        total = len(self._flat_entries)
        needle = attr.strip()
        value_needle = value.strip() if value else None
        for offset in range(total):
            idx = (start + offset * step) % total
            entry = self._flat_entries[idx]
            attr_sources: list[str] = []
            if entry.node.source_index is not None:
                instruction = self.view.instructions.get(entry.node.source_index)
                if instruction and instruction.attrs:
                    attr_sources.append(instruction.attrs)
            if entry.node.attrs:
                attr_sources.append(entry.node.attrs)
            if not attr_sources:
                continue
            for attrs_source in attr_sources:
                attrs = _parse_attrs_flat(attrs_source)
                matched_key = None
                for key in attrs.keys():
                    if _fuzzy_match(needle, key):
                        matched_key = key
                        break
                if not matched_key:
                    continue
                attr_value = attrs.get(matched_key)
                if value_needle:
                    if attr_value is None:
                        continue
                    if not _fuzzy_match(value_needle, attr_value):
                        continue
                self._ensure_details_visible()
                self._set_selected_index(idx)
                return


    def _prepare_details_matches(self, query: str) -> None:
        q = query.lower()
        self._details_matches = [i for i, line in enumerate(self._details_lines) if q in line.lower()]
        self._details_match_pos = -1

    def _find_next_details(self, forward: bool = True) -> None:
        if not self._details_matches:
            return
        step = 1 if forward else -1
        self._details_match_pos = (self._details_match_pos + step) % len(self._details_matches)
        line_idx = self._details_matches[self._details_match_pos]
        self._scroll_details_to(line_idx)

    def _details_focused(self) -> bool:
        return self._focus_target == "details"

    def _details_page_delta(self) -> int:
        details = self.query_one("#details", RichLog)
        return max(3, details.size.height - 2)

    def _scroll_details(self, delta: int) -> None:
        details = self.query_one("#details", RichLog)
        current = getattr(details, "scroll_y", 0)
        self._scroll_details_to(current + delta)

    def _scroll_details_to(self, line: int) -> None:
        details = self.query_one("#details", RichLog)
        try:
            details.scroll_to(y=line)
        except Exception:
            pass

    def _scroll_horizontal(self, delta: int) -> None:
        target = self.query_one("#details", RichLog) if self._details_focused() else self.query_one("#list", RichLog)
        current = getattr(target, "scroll_x", 0)
        self._scroll_horizontal_to(max(0, current + delta), target)

    def _scroll_horizontal_to(self, x: int, target: RichLog | None = None) -> None:
        if target is None:
            target = self.query_one("#details", RichLog) if self._details_focused() else self.query_one("#list", RichLog)
        try:
            target.scroll_to(x=x)
        except Exception:
            pass

    def _rebuild_def_use_maps(self) -> None:
        self.def_map = {}
        self.use_map = {}
        self.use_by_def = {}
        last_def_by_container: dict[str | None, dict[str, int]] = {}
        self._container_by_line = _collect_host_container_by_line(self.document.lines)
        def_container_by_line: dict[int, str | None] = {}
        for line_idx in sorted(self.view.instructions.keys()):
            line_text = self.document.lines[line_idx]
            if line_text.strip().endswith("{") and "ws.func" in line_text:
                last_def_by_container = {}
            instruction = self.view.instructions[line_idx]
            container_id = self._container_by_line.get(line_idx)
            last_def = last_def_by_container.setdefault(container_id, {})
            if instruction.inst.startswith("ws_rt."):
                if instruction.inst.startswith("ws_rt.cmdh.waf.alloc"):
                    for value in _operand_values(instruction.operands):
                        alloc = _alloc_for_value(self.document.allocs, value, line_idx)
                        parent = alloc.parent if alloc else None
                        key = parent or value
                        prev_def = last_def.get(key)
                        if prev_def is not None:
                            self.use_by_def.setdefault(prev_def, []).append(line_idx)
                            self.use_map.setdefault(value, []).append(line_idx)
                        if parent and parent in last_def:
                            last_def[value] = last_def[parent]
                    continue
                outputs, inputs = self.view.ws_rt_io.get(line_idx, ([], []))
                for value in inputs:
                    alloc = _alloc_for_value(self.document.allocs, value, line_idx)
                    parent = alloc.parent if alloc else None
                    key = parent or value
                    prev_def = last_def.get(key)
                    if prev_def is not None:
                        self.use_by_def.setdefault(prev_def, []).append(line_idx)
                        self.use_map.setdefault(value, []).append(line_idx)
                    if parent and parent in last_def:
                        last_def[value] = last_def[parent]
                for value in outputs:
                    if value:
                        last_def[value] = line_idx
                        self.def_map.setdefault(value, []).append(line_idx)
                        def_container_by_line[line_idx] = container_id
            else:
                defs = _expanded_result_values(instruction.results)
                uses = _operand_values(instruction.operands)
                for value in uses:
                    alloc = _alloc_for_value(self.document.allocs, value, line_idx)
                    parent = alloc.parent if alloc else None
                    key = parent or value
                    prev_def = last_def.get(key)
                    if prev_def is not None:
                        self.use_by_def.setdefault(prev_def, []).append(line_idx)
                        self.use_map.setdefault(value, []).append(line_idx)
                    if parent and parent in last_def:
                        last_def[value] = last_def[parent]
                for value in defs:
                    if value:
                        last_def[value] = line_idx
                        self.def_map.setdefault(value, []).append(line_idx)
                        def_container_by_line[line_idx] = container_id
        self._def_container_by_line = def_container_by_line
        for value in self.def_map:
            self.def_map[value].sort()
        for value in self.use_map:
            self.use_map[value].sort()
        for def_line in self.use_by_def:
            self.use_by_def[def_line].sort()

    def _instruction_defs_uses(self, instruction) -> tuple[list[str], list[str]]:
        uses = _operand_values(instruction.operands)
        if instruction.inst.startswith("ws_rt."):
            return [], uses
        return _expanded_result_values(instruction.results), uses

    def _lookup_def_line(
        self,
        value: str,
        line_idx: int,
        container_id: str | None,
        exclude_current: bool = False,
    ) -> int | None:
        defs = self.def_map.get(value)
        if not defs:
            return None
        best = None
        for def_line in defs:
            if exclude_current and def_line == line_idx:
                continue
            if self._def_container_by_line.get(def_line) != container_id:
                continue
            if def_line <= line_idx:
                best = def_line
            else:
                break
        return best

    def _update_related_highlights(self, node: NodeData) -> None:
        if node.kind != "inst" or node.source_index is None:
            self._highlight_level1 = set()
            self._highlight_level2 = set()
            self._highlight_level3 = set()
            self._highlight_semaphore = None
            self._highlight_num_rx = None
            self._highlight_semaphore_consumers = set()
            self._highlight_num_rx_consumers = set()
            self._render_viewport()
            return
        instruction = self.view.instructions.get(node.source_index)
        if not instruction:
            self._highlight_level1 = set()
            self._highlight_level2 = set()
            self._highlight_level3 = set()
            self._highlight_semaphore = None
            self._highlight_num_rx = None
            self._highlight_semaphore_consumers = set()
            self._highlight_num_rx_consumers = set()
            self._render_viewport()
            return
        self._highlight_semaphore = instruction.semaphore_unblocker
        self._highlight_num_rx = instruction.num_rx_unblocker
        self._highlight_semaphore_consumers = set(instruction.semaphore_consumers)
        self._highlight_num_rx_consumers = set(instruction.num_rx_consumers)
        defs, uses = self._instruction_defs_uses(instruction)
        container_id = self._container_by_line.get(node.source_index)

        level1_up: set[int] = set()
        level1_down: set[int] = set()
        for value in uses:
            alloc = _alloc_for_value(self.document.allocs, value, node.source_index)
            key = alloc.parent if alloc and alloc.parent else value
            def_line = self._lookup_def_line(
                key,
                node.source_index,
                container_id,
                exclude_current=True,
            )
            if def_line is not None and def_line != node.source_index:
                level1_up.add(def_line)
        for use_line in self.use_by_def.get(node.source_index, []):
            if use_line != node.source_index:
                level1_down.add(use_line)

        level2_up: set[int] = set()
        for line_idx in level1_up:
            inst = self.view.instructions.get(line_idx)
            if not inst:
                continue
            _, uses2 = self._instruction_defs_uses(inst)
            line_container = self._container_by_line.get(line_idx)
            for value in uses2:
                alloc = _alloc_for_value(self.document.allocs, value, line_idx)
                key = alloc.parent if alloc and alloc.parent else value
                def_line = self._lookup_def_line(
                    key,
                    line_idx,
                    line_container,
                    exclude_current=True,
                )
                if def_line is not None and def_line != node.source_index:
                    level2_up.add(def_line)

        level2_down: set[int] = set()
        for line_idx in level1_down:
            inst = self.view.instructions.get(line_idx)
            if not inst:
                continue
            for use_line in self.use_by_def.get(line_idx, []):
                if use_line != node.source_index:
                    level2_down.add(use_line)

        level1 = level1_up | level1_down
        level2 = (level2_up | level2_down) - level1
        level3_up: set[int] = set()
        for line_idx in level2_up:
            inst = self.view.instructions.get(line_idx)
            if not inst:
                continue
            _, uses3 = self._instruction_defs_uses(inst)
            line_container = self._container_by_line.get(line_idx)
            for value in uses3:
                alloc = _alloc_for_value(self.document.allocs, value, line_idx)
                key = alloc.parent if alloc and alloc.parent else value
                def_line = self._lookup_def_line(
                    key,
                    line_idx,
                    line_container,
                    exclude_current=True,
                )
                if def_line is not None and def_line != node.source_index:
                    level3_up.add(def_line)

        level3_down: set[int] = set()
        for line_idx in level2_down:
            inst = self.view.instructions.get(line_idx)
            if not inst:
                continue
            for use_line in self.use_by_def.get(line_idx, []):
                if use_line != node.source_index:
                    level3_down.add(use_line)

        level3 = (level3_up | level3_down) - level1 - level2
        self._highlight_level1 = set(level1)
        self._highlight_level2 = set(level2)
        self._highlight_level3 = set(level3)
        self._render_viewport()


class LineEntry:
    def __init__(
        self,
        node: NodeData,
        depth: int,
        raw_label: str,
        text: Text,
        loc_suffix: str | None = None,
        loc_group_suffix: str | None = None,
        loc_prefix_base: str | None = None,
        group_bar: bool = False,
        group_level: int = 0,
        group_prefix: str | None = None,
        group_chain: list[str] | None = None,
    ) -> None:
        self.node = node
        self.depth = depth
        self.raw_label = raw_label
        self.text = text
        self.loc_suffix = loc_suffix
        self.loc_group_suffix = loc_group_suffix
        self.loc_prefix_base = loc_prefix_base
        self.group_bar = group_bar
        self.group_level = group_level
        self.group_prefix = group_prefix
        self.group_chain = group_chain or []


class FuncSelectScreen(ModalScreen[int]):
    CSS = """
    FuncSelectScreen {
        align: center middle;
    }

    #jump {
        width: 80%;
        height: 80%;
        border: round $accent;
        padding: 1 2;
        background: $surface;
    }
    """

    def __init__(self, sections: list[tuple[str, int, int]]) -> None:
        super().__init__()
        self.sections = sections

    def compose(self) -> ComposeResult:
        items: list[ListItem] = []
        for idx, (label, _start, _end) in enumerate(self.sections):
            item = ListItem(Label(label))
            item._jump_index = idx  # type: ignore[attr-defined]
            items.append(item)
        yield ListView(*items, id="jump")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        index = getattr(item, "_jump_index", None)
        self.dismiss(index)

    def on_key(self, event: events.Key) -> None:
        if event.key == "q":
            self.app.exit()
            event.stop()


class FileSelectScreen(ModalScreen[int]):
    CSS = """
    FileSelectScreen {
        align: center middle;
    }

    #file-pick {
        width: 80%;
        height: 80%;
        border: round $accent;
        padding: 1 2;
        background: $surface;
    }
    """

    def __init__(self, files: list[Path], root: Path | None) -> None:
        super().__init__()
        self.files = files
        self.root = root
        self._resolved_root = root.resolve() if root else None

    def compose(self) -> ComposeResult:
        items: list[ListItem] = []
        for idx, path in enumerate(self.files):
            resolved_path = path.resolve()
            label = str(resolved_path)
            if self._resolved_root:
                try:
                    label = str(resolved_path.relative_to(self._resolved_root))
                except ValueError:
                    label = str(resolved_path)
            item = ListItem(Label(label))
            item._file_index = idx  # type: ignore[attr-defined]
            items.append(item)
        yield ListView(*items, id="file-pick")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        index = getattr(item, "_file_index", None)
        self.dismiss(index)

    def on_key(self, event: events.Key) -> None:
        if event.key == "q":
            self.app.exit()
            event.stop()


class QuickJumpScreen(ModalScreen[int | None]):
    CSS = """
    QuickJumpScreen {
        align: center middle;
    }

    #jump-panel {
        width: 80%;
        height: 80%;
        border: round $accent;
        padding: 1 2;
        background: $surface;
    }
    """

    def __init__(self, items: list[tuple[Text, int]]) -> None:
        super().__init__()
        self.items = items

    def compose(self) -> ComposeResult:
        items: list[ListItem] = []
        for idx, (label, line_idx) in enumerate(self.items):
            item = ListItem(Label(label))
            item._jump_index = idx  # type: ignore[attr-defined]
            items.append(item)
        with Vertical(id="jump-panel"):
            yield Label("Quick jump")
            yield ListView(*items, id="jump-list")
            yield Label("j/k, ↑/↓, C-n/C-p, PgUp/PgDn, g/G; Enter = jump, Esc = cancel")

    def _move_jump_cursor(self, delta: int) -> None:
        jump_list = self.query_one("#jump-list", ListView)
        count = len(jump_list.children)
        if count <= 0:
            return
        current = jump_list.index if jump_list.index is not None else 0
        target = max(0, min(count - 1, current + delta))
        jump_list.index = target

    def _set_jump_cursor(self, index: int) -> None:
        jump_list = self.query_one("#jump-list", ListView)
        count = len(jump_list.children)
        if count <= 0:
            return
        jump_list.index = max(0, min(count - 1, index))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        index = getattr(item, "_jump_index", None)
        if index is None:
            self.dismiss(None)
            return
        line_idx = self.items[index][1]
        self.dismiss(line_idx)

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()
            return
        if event.key in {"down", "j", "ctrl+n"}:
            self._move_jump_cursor(1)
            event.stop()
            return
        if event.key in {"up", "k", "ctrl+p"}:
            self._move_jump_cursor(-1)
            event.stop()
            return
        if event.key in {"pagedown", "ctrl+v"}:
            self._move_jump_cursor(10)
            event.stop()
            return
        if event.key in {"pageup", "ctrl+u", "alt+v"}:
            self._move_jump_cursor(-10)
            event.stop()
            return
        if event.key in {"home", "g"}:
            self._set_jump_cursor(0)
            event.stop()
            return
        if event.key in {"end", "G"}:
            self._set_jump_cursor(10_000)
            event.stop()


class ToggleScreen(ModalScreen[dict[str, bool] | None]):
    CSS = """
    ToggleScreen {
        align: center middle;
    }

    #toggle-panel {
        width: 60%;
        border: round $accent;
        padding: 1 2;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("enter", "apply", "Apply"),
        Binding("escape", "cancel", "Cancel"),
    ]

    _TOGGLE_ITEMS: list[tuple[str, str, str, str]] = [
        ("a", "Show alloc/free", "show_alloc_free", "alloc_free"),
        ("b", "Show full prefix", "show_full_prefix", "full_prefix"),
        ("c", "Show line numbers", "show_line_numbers", "line_numbers"),
        ("d", "Show alloc sizes", "show_alloc_sizes", "alloc_sizes"),
        ("e", "Show source vars", "show_source_vars", "show_source_vars"),
        ("f", "Show full source vars", "show_full_source_vars", "full_source_vars"),
        ("g", "Show types", "show_left_types", "show_left_types"),
        ("h", "Group loc prefixes", "group_loc_prefixes", "group_loc_prefixes"),
        ("i", "Show only tx/rx", "show_only_tx_rx", "show_only_tx_rx"),
        ("j", "Align columns", "align_left_panel", "align_left_panel"),
        ("k", "Show loc suffix", "show_left_loc", "show_left_loc"),
        ("l", "Split axis view", "split_axis_view", "split_axis_view"),
        (
            "m",
            "Highlight non-simple src/tgt",
            "highlight_non_simple_srctgt",
            "highlight_non_simple_srctgt",
        ),
    ]

    def __init__(self, options: RenderOptions) -> None:
        super().__init__()
        self.options = options
        self._toggle_key_to_id = {key: checkbox_id for key, _, _, checkbox_id in self._TOGGLE_ITEMS}

    def compose(self) -> ComposeResult:
        with Vertical(id="toggle-panel"):
            yield Label("Left panel options")
            for key, label, option_attr, checkbox_id in self._TOGGLE_ITEMS:
                value = getattr(self.options, option_attr)
                yield Checkbox(f"({key}) {label}", value=value, id=checkbox_id)
            yield Label("Shortcuts: press the letter in parentheses to toggle")
            yield Label("Enter = apply, Esc = cancel")

    def action_apply(self) -> None:
        values = {
            "alloc_free": self.query_one("#alloc_free", Checkbox).value,
            "full_prefix": self.query_one("#full_prefix", Checkbox).value,
            "line_numbers": self.query_one("#line_numbers", Checkbox).value,
            "alloc_sizes": self.query_one("#alloc_sizes", Checkbox).value,
            "show_source_vars": self.query_one("#show_source_vars", Checkbox).value,
            "full_source_vars": self.query_one("#full_source_vars", Checkbox).value,
            "show_left_types": self.query_one("#show_left_types", Checkbox).value,
            "group_loc_prefixes": self.query_one("#group_loc_prefixes", Checkbox).value,
            "show_only_tx_rx": self.query_one("#show_only_tx_rx", Checkbox).value,
            "align_left_panel": self.query_one("#align_left_panel", Checkbox).value,
            "show_left_loc": self.query_one("#show_left_loc", Checkbox).value,
            "split_axis_view": self.query_one("#split_axis_view", Checkbox).value,
            "highlight_non_simple_srctgt": self.query_one(
                "#highlight_non_simple_srctgt", Checkbox
            ).value,
        }
        self.dismiss(values)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            self.action_apply()
            event.stop()
        elif event.key == "escape":
            self.action_cancel()
            event.stop()
        else:
            key = event.key.lower()
            item_id = self._toggle_key_to_id.get(key)
            if item_id:
                checkbox = self.query_one(f"#{item_id}", Checkbox)
                checkbox.value = not checkbox.value
                event.stop()


class AttrSearchScreen(ModalScreen[dict[str, str | None] | None]):
    CSS = """
    AttrSearchScreen {
        align: center middle;
    }

    #attr-panel {
        width: 70%;
        height: 70%;
        border: round $accent;
        padding: 1 2;
        background: $surface;
    }

    """

    def __init__(self, attr_items: list[tuple[str, str | None]]) -> None:
        super().__init__()
        self._unique_values: dict[str, str] = {}
        self._autofilled_attr: str | None = None
        self._autofilled_value: str | None = None
        self._dropdown_items: list[DropdownItem] = []
        for name, peek in attr_items:
            normalized_peek = None
            if peek is not None:
                normalized_peek = " ".join(peek.split())
                self._unique_values[name] = peek
            prefix = None
            if normalized_peek is not None and self._should_show_value_preview(normalized_peek):
                prefix = f"= {normalized_peek}  "
            self._dropdown_items.append(DropdownItem(main=name, prefix=prefix))

    @staticmethod
    def _should_show_value_preview(value: str) -> bool:
        if len(value) > 48:
            return False
        if any(ch in value for ch in "{}[]"):
            return False
        return True

    def compose(self) -> ComposeResult:
        with Vertical(id="attr-panel"):
            yield Label("Attribute search")
            attr_input = Input(placeholder="Attribute name (fuzzy, e.g. sq, atrn)", id="attr_input")
            yield attr_input
            yield AutoComplete(target=attr_input, candidates=self._dropdown_items)
            yield Input(placeholder="Optional value (fuzzy)", id="attr_value")
            yield Label("Enter = search, Esc = cancel")

    def on_mount(self) -> None:
        self.query_one("#attr_input", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "attr_input":
            attr_name = event.value.strip()
            value_input = self.query_one("#attr_value", Input)
            unique_value = self._unique_values.get(attr_name)
            if unique_value is not None:
                value_input.value = unique_value
                self._autofilled_attr = attr_name
                self._autofilled_value = unique_value
            elif (
                self._autofilled_attr is not None
                and value_input.value == (self._autofilled_value or "")
            ):
                value_input.value = ""
                self._autofilled_attr = None
                self._autofilled_value = None
        elif event.input.id == "attr_value":
            if self._autofilled_value is not None and event.value != self._autofilled_value:
                self._autofilled_attr = None
                self._autofilled_value = None

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            value_input = self.query_one("#attr_value", Input)
            chosen = self.query_one("#attr_input", Input).value.strip()
            if not chosen:
                event.stop()
                return
            payload = {"attr": chosen, "value": value_input.value.strip()}
            self.dismiss(payload)
            event.stop()
        elif event.key == "escape":
            self.dismiss(None)
            event.stop()


class ConfirmQuitScreen(ModalScreen[bool | None]):
    CSS = """
    ConfirmQuitScreen {
        align: center middle;
    }

    #quit-panel {
        width: auto;
        max-width: 40;
        height: auto;
        border: round $accent;
        padding: 1 2;
        background: $surface;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="quit-panel"):
            yield Label("Quit IR viewer?")
            yield Label("Y = quit, N/Esc = cancel")

    def on_key(self, event: events.Key) -> None:
        if event.key in {"y", "enter"}:
            self.dismiss(True)
            event.stop()
        elif event.key in {"n", "escape"}:
            self.dismiss(False)
            event.stop()


class HelpScreen(ModalScreen[None]):
    CSS = """
    HelpScreen {
        align: center middle;
    }

    #help-panel {
        width: 80%;
        height: 80%;
        border: round $accent;
        padding: 1 2;
        background: $surface;
    }

    #help-log {
        height: 1fr;
    }
    """

    def __init__(self, items: list[tuple[str, str]]) -> None:
        super().__init__()
        self.items = items

    def compose(self) -> ComposeResult:
        with Vertical(id="help-panel"):
            yield Label("Instruction format (left panel):")
            yield Label("Prefix: ➊ (first iteration), [handle], onX/onY")
            yield Label("Outputs: listed before '='")
            yield Label("Inst: operation name (with optional .op suffix)")
            yield Label("Inputs: listed after inst name")
            yield Label("Source vars: ⟨...⟩ suffix when enabled")
            yield Label("Shortcuts")
            yield RichLog(id="help-log", wrap=True)
            yield Label("Esc = close")

    def on_mount(self) -> None:
        log = self.query_one("#help-log", RichLog)
        for key, desc in self.items:
            log.write(f"{key.ljust(10)} {desc}")

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()


class ReportScreen(ModalScreen[None]):
    CSS = """
    ReportScreen {
        align: center middle;
    }

    #report-panel {
        width: 90%;
        height: 90%;
        border: round $accent;
        padding: 1 2;
        background: $surface;
    }

    #report-log {
        height: 1fr;
    }

    #report-filter {
        margin: 0 0 1 0;
    }

    #report-table {
        height: 1fr;
    }
    """

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self._rows: list[tuple[str, dict[str, int]]] = []
        self._total_instructions: int = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="report-panel"):
            yield Label("Instruction histogram report", id="report-summary")
            yield Label("Type to fuzzy-filter by instruction name")
            yield Input(placeholder="Filter instructions (fuzzy)", id="report-filter")
            yield DataTable(id="report-table")
            yield Label("Esc = close")

    def on_mount(self) -> None:
        self._total_instructions, self._rows = _instruction_histogram_data(self.path)
        table = self.query_one("#report-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Instruction", "onX", "onY", "others", "total")
        table.cursor_type = "row"
        table.fixed_rows = 0
        self._refresh_table()
        self.query_one("#report-filter", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "report-filter":
            self._refresh_table()

    def _refresh_table(self) -> None:
        table = self.query_one("#report-table", DataTable)
        query = self.query_one("#report-filter", Input).value.strip()
        table.fixed_rows = 0
        table.clear(columns=False)
        shown = 0
        def fmt(value: int) -> str:
            return "-" if value == 0 else str(value)
        for name, row in self._rows:
            if query and not _fuzzy_match(query, name):
                continue
            table.add_row(name, fmt(row["onX"]), fmt(row["onY"]), fmt(row["others"]), fmt(row["total"]))
            shown += 1
        self.query_one("#report-summary", Label).update(
            f"Instruction histogram report — total: {self._total_instructions}, shown: {shown}, unique: {len(self._rows)}"
        )

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()




def _collect_ws_funcs(lines: list[str]) -> list[tuple[str, int, int]]:
    funcs: list[tuple[str, int, int]] = []
    func_re = re.compile(r"\bws\.func\s+@([^\s{]+)")
    idx = 0
    depth = 0
    in_func = False
    func_start = 0
    func_name = ""
    while idx < len(lines):
        line = lines[idx]
        if not in_func:
            match = func_re.search(line)
            if match:
                func_name = match.group(1)
                func_start = idx
                depth = _brace_delta(line)
                if depth <= 0:
                    depth = 1
                in_func = True
        else:
            depth += _brace_delta(line)
            if depth <= 0:
                funcs.append((f"ws.func @{func_name}", func_start, idx))
                in_func = False
                func_name = ""
        idx += 1
    if in_func:
        funcs.append((f"ws.func @{func_name}", func_start, len(lines) - 1))
    return funcs


def _collect_host_container_by_line(lines: list[str]) -> dict[int, str | None]:
    container_by_line: dict[int, str | None] = {}
    host_re = re.compile(r"\bws_rt\.host\.(cmd|wgt|act)\b")
    stack: list[tuple[str, int]] = []
    depth = 0
    counter = 0
    for idx, line in enumerate(lines):
        match = host_re.search(line)
        if match:
            kind = match.group(1)
            counter += 1
            start_depth = depth + max(1, _brace_delta(line))
            stack.append((f"{kind}:{counter}", start_depth))
        if stack:
            container_by_line[idx] = stack[-1][0]
        else:
            container_by_line[idx] = None
        depth += _brace_delta(line)
        while stack and depth < stack[-1][1]:
            stack.pop()
    return container_by_line


def _common_prefix(items: list[str]) -> str:
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


def _loc_group_prefix(loc_suffix: str | None) -> str | None:
    if not loc_suffix:
        return None
    if "." not in loc_suffix:
        return None
    return loc_suffix.rsplit(".", 1)[0] or None


def _loc_parts(loc_suffix: str | None) -> list[str]:
    if not loc_suffix:
        return []
    return [p for p in loc_suffix.split(".") if p]


def _loc_group_chain(loc_suffix: str | None) -> list[str]:
    parts = _loc_parts(loc_suffix)
    if not parts:
        return []
    prefixes: list[str] = []
    for idx in range(1, len(parts) + 1):
        prefixes.append(".".join(parts[:idx]))
    return prefixes


def _parse_semaphore_count(value: str | None) -> int | None:
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


def _strip_group_prefix(loc_suffix: str | None, prefix: str) -> str | None:
    if not loc_suffix:
        return None
    if loc_suffix == prefix:
        return ""
    if loc_suffix.startswith(prefix + "."):
        return loc_suffix[len(prefix) + 1 :]
    return loc_suffix


def _strip_loc_op_suffix(
    loc_suffix: str | None,
    inst_name: str | None,
    inst_base_name: str | None = None,
) -> str | None:
    if not loc_suffix:
        return loc_suffix
    loc_parts = _loc_parts(loc_suffix)
    if not loc_parts:
        return loc_suffix
    loc_last = loc_parts[-1]
    inst_last = inst_name.split(".")[-1] if inst_name else None
    base_last = inst_base_name.split(".")[-1] if inst_base_name else None
    if loc_last not in {inst_last, base_last}:
        return loc_suffix
    trimmed = loc_parts[:-1]
    return ".".join(trimmed) if trimmed else ""


def _group_loc_suffix(
    suffix: str | None,
    base_prefix: str | None,
    inst_name: str | None,
    inst_base_name: str | None = None,
) -> str | None:
    if suffix is None:
        if base_prefix:
            return base_prefix
        return None
    stripped = _strip_loc_op_suffix(suffix, inst_name, inst_base_name)
    if stripped is None:
        return None
    parts = _loc_parts(stripped)
    base = base_prefix or ""
    if base and len(parts) <= 1:
        return f"{base}.{stripped}" if stripped else base
    return stripped


def _axis_from_attrs(attrs: str | None) -> str | None:
    if not attrs:
        return None
    has_x = re.search(r"\bonX\b", attrs) is not None
    has_y = re.search(r"\bonY\b", attrs) is not None
    if has_x and not has_y:
        return "X"
    if has_y and not has_x:
        return "Y"
    return None


def _parse_attrs_flat(attrs: str) -> dict[str, str | None]:
    return _parse_attrs_nested(attrs)


def _parse_attrs_nested(attrs: str, prefix: str = "") -> dict[str, str | None]:
    text = attrs.strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()
    parts = [p.strip() for p in _split_top_level(text, ",") if p.strip()]
    result: dict[str, str | None] = {}
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            full_key = f"{prefix}.{key}" if prefix else key
            norm_value = _strip_type_annotations(value)
            if norm_value.startswith('"') and norm_value.endswith('"') and len(norm_value) >= 2:
                norm_value = norm_value[1:-1]
            result[full_key] = norm_value
            if value.startswith("{") and value.endswith("}"):
                result.update(_parse_attrs_nested(value, full_key))
        else:
            key = part.strip()
            full_key = f"{prefix}.{key}" if prefix else key
            result[full_key] = None
    return result


def _fuzzy_match(query: str, text: str) -> bool:
    q = "".join(query.lower().split())
    t = "".join(text.lower().split())
    if not q:
        return True
    it = iter(t)
    return all(ch in it for ch in q)


def _filter_nodes_by_range(nodes: list[Node], start: int, end: int) -> list[Node]:
    filtered: list[Node] = []
    for node in nodes:
        child_filtered = _filter_nodes_by_range(node.children, start, end) if node.children else []
        in_range = node.source_index is not None and start <= node.source_index <= end
        if in_range or child_filtered:
            filtered.append(
                Node(
                    label=node.label,
                    kind=node.kind,
                    source_index=node.source_index,
                    children=child_filtered,
                    attrs=node.attrs,
                    value=node.value,
                    layout_index=node.layout_index,
                )
            )
    return filtered


def _clamp_window_start(index: int, total: int, window_size: int) -> int:
    if total <= window_size:
        return 0
    half = window_size // 2
    return max(0, min(index - half, total - window_size))


def _onxy_prefix(attrs: str | None) -> str:
    if not attrs:
        return ""
    prefixes: list[str] = []
    if re.search(r"\bonX\b", attrs):
        prefixes.append("onX")
    if re.search(r"\bonY\b", attrs):
        prefixes.append("onY")
    return " ".join(prefixes) + " " if prefixes else ""


def _semaphore_value(attrs: str | None) -> str | None:
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


def _is_zero_literal(value: str) -> bool:
    return re.fullmatch(r"0+(?:\.0+)?(?:e[+-]?0+)?", value, re.IGNORECASE) is not None


def _default_ir_path() -> Path:
    return Path.cwd() / "ir.mlir"


def _ir_not_found_message(path: Path, requested: str | None = None) -> str:
    resolved = path.expanduser().resolve(strict=False)
    lines = [
        f"IR file not found: {path}",
        f"Resolved path: {resolved}",
        f"Current working directory: {Path.cwd()}",
        f"Requested argument: {requested or str(path)}",
    ]

    seen: set[Path] = set()
    search_dirs: list[Path] = [Path.cwd(), path.parent, _default_ir_path().parent]
    for directory in search_dirs:
        directory = directory.resolve(strict=False)
        if directory in seen or not directory.exists() or not directory.is_dir():
            continue
        seen.add(directory)
        mlir_files = sorted(directory.glob("*.mlir"))
        if not mlir_files:
            continue
        preview = ", ".join(candidate.name for candidate in mlir_files[:8])
        if len(mlir_files) > 8:
            preview += f", ... (+{len(mlir_files) - 8} more)"
        lines.append(f"Available .mlir files in {directory}: {preview}")

    return "\n".join(lines)


def _cursor_row(cursor_location) -> int:
    if hasattr(cursor_location, "row"):
        return cursor_location.row
    if isinstance(cursor_location, tuple) and cursor_location:
        return int(cursor_location[0])
    return 0


def _instruction_histogram_data(path: Path) -> tuple[int, list[tuple[str, dict[str, int]]]]:
    doc = Document.from_path(path)
    view = DocumentView(doc)
    view.render()

    def _valid_inst_name(inst_name: str) -> bool:
        name = inst_name.strip()
        if "." not in name:
            return False
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name):
            return False
        return True

    def _valid_op_name(op_name: str) -> bool:
        return re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", op_name) is not None

    counts: dict[str, dict[str, int]] = {}
    included_count = 0
    for source_index in sorted(view.instructions.keys()):
        instruction = view.instructions[source_index]
        if not _valid_inst_name(instruction.inst):
            continue
        op_value = _op_attr_value(instruction.attrs)
        key = f"{instruction.inst}.{op_value}" if op_value and _valid_op_name(op_value) else instruction.inst
        axis = _axis_from_attrs(instruction.attrs)
        axis_key = "onX" if axis == "X" else "onY" if axis == "Y" else "others"
        row = counts.setdefault(key, {"onX": 0, "onY": 0, "others": 0, "total": 0})
        row[axis_key] += 1
        row["total"] += 1
        included_count += 1

    ordered = sorted(counts.items(), key=lambda item: (-item[1]["total"], item[0]))
    return included_count, ordered


def _instruction_histogram_report(path: Path) -> str:
    included_count, ordered = _instruction_histogram_data(path)
    name_width = max([len("Instruction"), *(len(name) for name, _ in ordered)] or [11])
    lines = [
        f"Instruction histogram for {path}",
        f"Total instructions: {included_count}",
        f"Unique inst/op keys: {len(ordered)}",
        "",
        f"{'Instruction'.ljust(name_width)}  {'onX':>6}  {'onY':>6}  {'others':>8}  {'total':>6}",
        f"{'-' * name_width}  {'-' * 6}  {'-' * 6}  {'-' * 8}  {'-' * 6}",
    ]
    for name, row in ordered:
        lines.append(
            f"{name.ljust(name_width)}  {row['onX']:6d}  {row['onY']:6d}  {row['others']:8d}  {row['total']:6d}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="IR Viewer")
    parser.add_argument("path", nargs="?", default=None, help="Path to IR file")
    parser.add_argument("--profile", action="store_true", help="Run in headless profile mode")
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Print instruction histogram (inst.<op>) by axis and exit",
    )
    parser.add_argument("--show-alloc-free", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-full-prefix", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-line-numbers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-alloc-sizes", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-source-vars", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-full-source-vars", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-left-types", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wrap-left-panel", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--group-loc-prefixes", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--align-left-panel", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-left-loc", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--highlight-non-simple-srctgt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Highlight layout SrcTgts lines where source and target sizes differ",
    )
    parsed = parser.parse_args()

    profile_mode = parsed.profile
    path = Path(parsed.path) if parsed.path else _default_ir_path()
    file_choices: list[Path] = []
    file_root: Path | None = None
    if path.exists() and path.is_dir():
        file_root = path
        file_choices = sorted(path.glob("*.mlir"))
        if not file_choices:
            raise SystemExit(
                f"No .mlir files found in directory: {path}\n"
                f"Current working directory: {Path.cwd()}"
            )
        path = file_choices[0]
    if not path.exists():
        raise SystemExit(_ir_not_found_message(path, parsed.path))
    options = RenderOptions()
    if parsed.show_alloc_free is not None:
        options.show_alloc_free = parsed.show_alloc_free
    if parsed.show_full_prefix is not None:
        options.show_full_prefix = parsed.show_full_prefix
    if parsed.show_line_numbers is not None:
        options.show_line_numbers = parsed.show_line_numbers
    if parsed.show_alloc_sizes is not None:
        options.show_alloc_sizes = parsed.show_alloc_sizes
    if parsed.show_source_vars is not None:
        options.show_source_vars = parsed.show_source_vars
    if parsed.show_full_source_vars is not None:
        options.show_full_source_vars = parsed.show_full_source_vars
    if parsed.show_left_types is not None:
        options.show_left_types = parsed.show_left_types
    if parsed.wrap_left_panel is not None:
        options.wrap_left_panel = parsed.wrap_left_panel
    if parsed.group_loc_prefixes is not None:
        options.group_loc_prefixes = parsed.group_loc_prefixes
    if parsed.align_left_panel is not None:
        options.align_left_panel = parsed.align_left_panel
    if parsed.show_left_loc is not None:
        options.show_left_loc = parsed.show_left_loc
    if parsed.highlight_non_simple_srctgt is not None:
        options.highlight_non_simple_srctgt = parsed.highlight_non_simple_srctgt
    if parsed.histogram:
        print(_instruction_histogram_report(path))
        return
    app = IRViewerApp(
        path,
        profile_mode=profile_mode,
        options=options,
        file_choices=file_choices,
        file_root=file_root,
    )
    app.run(headless=profile_mode, mouse=True)


def _looks_like_type(token: str) -> bool:
    if "<" in token or ">" in token or "!" in token:
        return True
    if re.match(r"^(ui|i|f|bf)\d+$", token):
        return True
    if re.match(r"^(?:\d+x)+(?:ui|i|f|bf)\d+$", token):
        return True
    if re.match(r"^(?:\?x)+(?:ui|i|f|bf)\d+$", token):
        return True
    if token in {"index"}:
        return True
    return False


if __name__ == "__main__":
    main()
