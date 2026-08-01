"""Activity analysis plots: actograms, LD/DD profiles, and phase totals."""

from __future__ import annotations

import html
import io
import math
import re
from dataclasses import dataclass, field, replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from mosquito_lab.paths import mosquito_project_dir

EXPERIMENTS_DIR = mosquito_project_dir() / "experiments"

MOSQUITO_KINDS = [
    "Female sg (WT)",
    "Male sg (WT)",
    "Female KO",
    "Male KO",
]

DEFAULT_GROUP_LAYOUT = pd.DataFrame(
    [
        {"group": "Female sg (WT)", "index": "0-5", "color": "#e63946"},
        {"group": "Male sg (WT)", "index": "6-11", "color": "#457b9d"},
        {"group": "Female KO", "index": "12-17", "color": "#f4a261"},
        {"group": "Male KO", "index": "18-23", "color": "#2a9d8f"},
    ]
)

MOSQUITO_NUMBERS = list(range(1, 7))

BAR_COLOR = "#ef3c26"

FONT_CHOICES = [
    "DejaVu Sans",
    "Arial",
    "Helvetica",
    "Times New Roman",
    "Courier New",
    "Comic Sans MS",
]


@dataclass
class PlotStyle:
    """User-tunable figure titles and typography."""

    graph_title: str = ""
    group_titles: dict[str, str] = field(default_factory=dict)
    xlabel: str = ""
    ylabel: str = ""
    width_scale: float = 1.0
    height_scale: float = 1.0
    font_family: str = "DejaVu Sans"
    font_size: float = 10.0
    title_size: float = 14.0
    label_size: float = 11.0
    tick_size: float = 9.0

    def display_group(self, name: str) -> str:
        override = self.group_titles.get(name, "").strip()
        return override or name

    def figsize(self, width: float, height: float) -> tuple[float, float]:
        return (max(1.0, width * self.width_scale), max(1.0, height * self.height_scale))

    def compose_title(self, default: str) -> str:
        custom = self.graph_title.strip()
        if not custom:
            return default
        if not default:
            return custom
        return f"{custom} — {default}"

    def x_label(self, default: str) -> str:
        return self.xlabel.strip() or default

    def y_label(self, default: str) -> str:
        return self.ylabel.strip() or default

    def rc(self) -> dict:
        return {
            "font.family": self.font_family,
            "font.size": self.font_size,
            "axes.titlesize": self.title_size,
            "axes.labelsize": self.label_size,
            "xtick.labelsize": self.tick_size,
            "ytick.labelsize": self.tick_size,
            "figure.titlesize": self.title_size + 2,
            "legend.fontsize": max(7.0, self.font_size - 1),
        }

    def overridden(
        self,
        *,
        graph_title: str | None = None,
        width_scale: float | None = None,
        height_scale: float | None = None,
        font_size: float | None = None,
        title_size: float | None = None,
        label_size: float | None = None,
        tick_size: float | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ) -> "PlotStyle":
        """Return a copy with selected fields replaced (None = keep)."""
        updates = {}
        if graph_title is not None and str(graph_title).strip():
            updates["graph_title"] = str(graph_title).strip()
        if width_scale is not None:
            updates["width_scale"] = float(width_scale)
        if height_scale is not None:
            updates["height_scale"] = float(height_scale)
        if font_size is not None:
            updates["font_size"] = float(font_size)
        if title_size is not None:
            updates["title_size"] = float(title_size)
        if label_size is not None:
            updates["label_size"] = float(label_size)
        if tick_size is not None:
            updates["tick_size"] = float(tick_size)
        if xlabel is not None and str(xlabel).strip():
            updates["xlabel"] = str(xlabel).strip()
        if ylabel is not None and str(ylabel).strip():
            updates["ylabel"] = str(ylabel).strip()
        return replace(self, **updates) if updates else self


def default_plot_style() -> PlotStyle:
    return PlotStyle()


def _md_inline_to_html(text: str) -> str:
    """Minimal markdown → HTML for hover tips (bold, code, bullets, paragraphs)."""
    text = text.strip()
    if not text:
        return ""
    blocks: list[str] = []
    list_items: list[str] = []

    def flush_list() -> None:
        nonlocal list_items
        if list_items:
            blocks.append(
                "<ul>" + "".join(f"<li>{li}</li>" for li in list_items) + "</ul>"
            )
            list_items = []

    def inline(s: str) -> str:
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
        return s

    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            flush_list()
            continue
        if stripped.startswith("- "):
            list_items.append(inline(stripped[2:]))
        else:
            flush_list()
            blocks.append(f"<p>{inline(stripped)}</p>")
    flush_list()
    return "".join(blocks)


def _help_bubble_body(what: str = "", how: str = "", ask: str | None = None) -> str:
    del ask  # "Ask yourself" prompts removed per request
    parts: list[str] = []
    if what:
        parts.append(_md_inline_to_html(what))
    if how:
        parts.append(_md_inline_to_html(how))
    return "".join(parts)


def _inject_fig_header_css() -> None:
    """Streamlit-like hover '?' help icons + quiet Style controls."""
    if st.session_state.get("_fig_header_css"):
        return
    st.session_state["_fig_header_css"] = True
    st.markdown(
        """
<style>
.ml-fig-head {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  margin: 0.15rem 0 0.35rem 0;
  flex-wrap: wrap;
}
.ml-fig-head h3, .ml-fig-head h4 {
  margin: 0 !important;
  padding: 0 !important;
  line-height: 1.35;
  display: inline;
}
.ml-fig-head.h3 { font-size: 1.25rem; font-weight: 600; }
.ml-fig-head.h4 { font-size: 1.05rem; font-weight: 600; }

.ml-help {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.05rem;
  height: 1.05rem;
  border-radius: 999px;
  border: 1px solid rgba(49, 51, 63, 0.35);
  color: rgba(49, 51, 63, 0.55);
  font-size: 0.72rem;
  font-weight: 600;
  line-height: 1;
  cursor: help;
  user-select: none;
  vertical-align: middle;
  flex: 0 0 auto;
  background: transparent;
}
.ml-help:hover, .ml-help:focus-within {
  color: rgba(49, 51, 63, 0.85);
  border-color: rgba(49, 51, 63, 0.55);
}
.ml-help-tip {
  visibility: hidden;
  opacity: 0;
  position: absolute;
  left: 50%;
  bottom: calc(100% + 0.45rem);
  transform: translateX(-50%);
  z-index: 1000;
  width: min(22rem, 70vw);
  max-height: 16rem;
  overflow: auto;
  padding: 0.65rem 0.75rem;
  border-radius: 0.5rem;
  background: #262730;
  color: #fafafa;
  font-size: 0.8rem;
  font-weight: 400;
  line-height: 1.45;
  text-align: left;
  box-shadow: 0 6px 20px rgba(0,0,0,0.25);
  pointer-events: none;
  transition: opacity 0.12s ease, visibility 0.12s ease;
}
.ml-help-tip p { margin: 0 0 0.45rem 0; }
.ml-help-tip p:last-child { margin-bottom: 0; }
.ml-help-tip ul { margin: 0.2rem 0 0.45rem 1.1rem; padding: 0; }
.ml-help-tip li { margin: 0.15rem 0; }
.ml-help-tip code {
  background: rgba(255,255,255,0.12);
  padding: 0.05rem 0.25rem;
  border-radius: 0.2rem;
  font-size: 0.78em;
}
.ml-help:hover .ml-help-tip,
.ml-help:focus-within .ml-help-tip {
  visibility: visible;
  opacity: 1;
}

div[data-testid="stPopover"] > button p {
  margin: 0 !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _title_with_help(
    title: str,
    *,
    what: str = "",
    how: str = "",
    ask: str | None = None,
    level: str = "###",
) -> None:
    """Render a section title with a Streamlit-style hover '?' tip."""
    _inject_fig_header_css()
    tip = _help_bubble_body(what, how, ask)
    level_cls = (
        "h3"
        if level.strip().startswith("###") and not level.strip().startswith("####")
        else "h4"
    )
    tag = "h3" if level_cls == "h3" else "h4"
    help_html = ""
    if tip:
        help_html = (
            '<span class="ml-help" tabindex="0" aria-label="Help">'
            f'?<span class="ml-help-tip">{tip}</span></span>'
        )
    st.markdown(
        f'<div class="ml-fig-head {level_cls}">'
        f"<{tag}>{html.escape(title)}</{tag}>{help_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def help_popover(body_md: str, *, label: str = "?") -> None:
    """A click-to-expand '?' button that reveals an explanation.

    Use this to tuck away wordy inline captions: the text is one click away
    instead of always on screen.
    """
    with st.popover(label):
        st.markdown(body_md)


def header_with_help(title: str, body_md: str, *, level: str = "###") -> None:
    """Section title with a click-to-expand '?' popover holding the explanation."""
    head_col, help_col = st.columns([0.85, 0.15])
    with head_col:
        st.markdown(f"{level} {title}")
    with help_col:
        help_popover(body_md)


_FIG_CUSTOM_PREFIX = "figfmt::"


def _fig_key(fig_id: str, name: str) -> str:
    return f"{_FIG_CUSTOM_PREFIX}{fig_id}::{name}"


def read_fig_style(fig_id: str, base: PlotStyle) -> PlotStyle:
    """Effective style for one figure: sidebar ``base`` + per-figure overrides.

    Pure read from ``st.session_state`` (renders nothing), so both ``fig_header``
    and the Section E compare view resolve the same per-figure customization.
    """
    if not fig_id:
        return base
    ss = st.session_state
    if not ss.get(_fig_key(fig_id, "override"), False):
        return base
    return base.overridden(
        graph_title=ss.get(_fig_key(fig_id, "title"), ""),
        xlabel=ss.get(_fig_key(fig_id, "xlabel"), ""),
        ylabel=ss.get(_fig_key(fig_id, "ylabel"), ""),
        width_scale=ss.get(_fig_key(fig_id, "width"), base.width_scale),
        height_scale=ss.get(_fig_key(fig_id, "height"), base.height_scale),
        font_size=ss.get(_fig_key(fig_id, "font"), base.font_size),
        title_size=ss.get(_fig_key(fig_id, "title_size"), base.title_size),
        label_size=ss.get(_fig_key(fig_id, "label_size"), base.label_size),
        tick_size=ss.get(_fig_key(fig_id, "tick_size"), base.tick_size),
    )


def _render_fig_customizer(fig_id: str, base: PlotStyle) -> PlotStyle:
    """Per-figure 'Customize' popover. Returns the effective style for this figure."""
    with st.popover("Customize", use_container_width=True):
        st.caption(
            "Overrides the sidebar **Plot style** for *this figure only*. "
            "Leave off to follow the sidebar."
        )
        on = st.checkbox("Customize this figure", key=_fig_key(fig_id, "override"))
        if on:
            st.text_input(
                "Title (prefix)",
                key=_fig_key(fig_id, "title"),
                placeholder="blank = inherit sidebar title",
            )
            lc, rc = st.columns(2)
            with lc:
                st.text_input(
                    "Rename X-axis",
                    key=_fig_key(fig_id, "xlabel"),
                    placeholder="blank = default",
                )
            with rc:
                st.text_input(
                    "Rename Y-axis",
                    key=_fig_key(fig_id, "ylabel"),
                    placeholder="blank = default",
                )
            wc, hc = st.columns(2)
            with wc:
                st.slider(
                    "Width", 0.5, 2.5, float(base.width_scale), 0.05,
                    key=_fig_key(fig_id, "width"),
                )
            with hc:
                st.slider(
                    "Height", 0.5, 2.5, float(base.height_scale), 0.05,
                    key=_fig_key(fig_id, "height"),
                )
            sc1, sc2 = st.columns(2)
            with sc1:
                st.slider(
                    "Body text", 6, 20, int(base.font_size),
                    key=_fig_key(fig_id, "font"),
                )
                st.slider(
                    "Axis label size", 6, 22, int(base.label_size),
                    key=_fig_key(fig_id, "label_size"),
                )
            with sc2:
                st.slider(
                    "Title size", 8, 28, int(base.title_size),
                    key=_fig_key(fig_id, "title_size"),
                )
                st.slider(
                    "Tick size", 5, 18, int(base.tick_size),
                    key=_fig_key(fig_id, "tick_size"),
                )
            if st.button("Reset to sidebar", key=_fig_key(fig_id, "reset")):
                prefix = f"{_FIG_CUSTOM_PREFIX}{fig_id}::"
                for k in [k for k in st.session_state if k.startswith(prefix)]:
                    del st.session_state[k]
                st.rerun()
    return read_fig_style(fig_id, base)


def fig_header(
    title: str,
    *,
    fig_id: str = "",
    base_style: PlotStyle | None = None,
    what: str = "",
    how: str = "",
    ask: str | None = None,
    level: str = "###",
    customize: bool = False,
) -> PlotStyle:
    """Figure title with hover '?' help and (optional) per-figure Customize popover.

    Returns the effective :class:`PlotStyle` for this figure — the sidebar style,
    plus any per-figure overrides when ``customize=True``. Pass the returned style
    to the plotting function so overrides actually take effect.
    """
    base = base_style or default_plot_style()
    if customize and fig_id:
        head_col, ctrl_col = st.columns([0.82, 0.18])
        with head_col:
            _title_with_help(title, what=what, how=how, ask=ask, level=level)
        with ctrl_col:
            return _render_fig_customizer(fig_id, base)
    _title_with_help(title, what=what, how=how, ask=ask, level=level)
    return base


def bin_sum(size: int, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    return np.array([np.sum(arr[i : i + size]) for i in range(0, len(arr), size)])


def load_activity(csv_source) -> pd.DataFrame:
    raw = pd.read_csv(csv_source)
    mosquito_cols = [c for c in raw.columns if c.startswith("mosquito_")]
    if not mosquito_cols:
        raise ValueError("No columns starting with 'mosquito_' were found in this CSV.")
    return pd.DataFrame(
        {
            "Mosquito": mosquito_cols,
            "frames": [raw[c].fillna(0).to_numpy(dtype=float) for c in mosquito_cols],
        }
    )


def build_counts(data: pd.DataFrame, bin_size: int) -> list[np.ndarray]:
    return [bin_sum(bin_size, f) for f in data["frames"]]


def groups_from_editor(groups_df: pd.DataFrame, n_mosq: int) -> dict[str, list[int]]:
    """Legacy start/end table → group → sorted mosquito indices."""
    groups: dict[str, list[int]] = {}
    for _, row in groups_df.iterrows():
        name = str(row["name"]).strip()
        if not name:
            continue
        start = max(0, int(row["start"]))
        end = min(n_mosq, int(row["end"]))
        if end <= start:
            continue
        groups[name] = list(range(start, end))
    return groups


def parse_index_spec(spec: str, n_mosq: int) -> list[int]:
    """Parse index column: '0-5', '6,7,8', or '0-5,10,12-14' (ranges inclusive)."""
    text = str(spec).strip()
    if not text:
        return []
    indices: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, end = int(left.strip()), int(right.strip())
            if start <= end:
                indices.update(range(start, end + 1))
            else:
                indices.update(range(end, start + 1))
        else:
            indices.add(int(part))
    return sorted(i for i in indices if 0 <= i < n_mosq)


def groups_from_layout_table(
    layout_df: pd.DataFrame, n_mosq: int
) -> tuple[dict[str, list[int]], dict[str, str]]:
    groups: dict[str, list[int]] = {}
    colors: dict[str, str] = {}
    for _, row in layout_df.iterrows():
        name = str(row["group"]).strip()
        if not name:
            continue
        idxs = parse_index_spec(str(row["index"]), n_mosq)
        if not idxs:
            continue
        groups.setdefault(name, []).extend(idxs)
        colors[name] = str(row["color"]).strip() or BAR_COLOR
    return (
        {name: sorted(set(idxs)) for name, idxs in groups.items()},
        colors,
    )


def colors_from_editor(groups_df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for _, row in groups_df.iterrows():
        name = str(row["name"]).strip()
        if name:
            out[name] = str(row["color"]).strip() or BAR_COLOR
    return out


def build_idx_to_label(groups: dict[str, list[int]]) -> dict[int, tuple[str, int]]:
    idx_to_label: dict[int, tuple[str, int]] = {}
    for name, indices in groups.items():
        for j, idx in enumerate(indices):
            idx_to_label[idx] = (name, j + 1)
    return idx_to_label


def apply_mosquito_exclusions(
    groups: dict[str, list[int]],
    group_colors: dict[str, str],
    excluded: set[int],
) -> tuple[dict[str, list[int]], dict[str, str]]:
    """Drop excluded mosquito indices from every group.

    Groups left with no mosquitoes are removed entirely (so plots don't try to
    average an empty set). Returns filtered ``(groups, group_colors)``; the
    originals are left untouched.
    """
    if not excluded:
        return groups, group_colors
    filtered = {
        name: kept
        for name, idxs in groups.items()
        if (kept := [i for i in idxs if i not in excluded])
    }
    colors = {name: c for name, c in group_colors.items() if name in filtered}
    return filtered, colors


def render_exclusion_control(
    groups: dict[str, list[int]],
    *,
    key: str,
    label_fn=None,
) -> set[int]:
    """Multiselect to drop specific mosquitoes from every figure and statistic.

    Options are every mosquito currently assigned to a group. ``label_fn(idx)``
    customizes each option's display (e.g. to add experiment names in the
    combined view); the default shows ``mosquito_<idx> · <group> #<n>``.
    Returns the set of excluded indices.
    """
    idx_to_label = build_idx_to_label(groups)
    options = sorted(idx_to_label)

    def _fmt(i: int) -> str:
        if label_fn is not None:
            return label_fn(i)
        grp, num = idx_to_label[i]
        return f"mosquito_{i} · {grp} #{num}"

    chosen = st.multiselect(
        "Exclude mosquitoes (optional)",
        options=options,
        format_func=_fmt,
        key=key,
        help="Pick any mosquitoes to remove from every graph and statistic. "
        "Leave empty to keep all.",
    )
    if chosen:
        st.caption(f"Excluding {len(chosen)} of {len(options)} mosquitoes — {len(options) - len(chosen)} remain.")
    return set(chosen)


def death_bin_by_idx(
    deaths: list[dict],
    groups: dict[str, list[int]],
    bin_size: int,
) -> dict[int, int]:
    out: dict[int, int] = {}
    for entry in deaths:
        group = str(entry.get("group", "")).strip()
        if group not in groups:
            continue
        try:
            num = int(entry["mosquito_num"])
            frame = int(entry["death_frame"])
        except (ValueError, TypeError, KeyError):
            continue
        indices = groups[group]
        if not (1 <= num <= len(indices)):
            continue
        out[indices[num - 1]] = frame // bin_size
    return out


def resolve_group_name(value, groups: dict[str, list[int]]) -> str | None:
    """Map CSV/manual group value to a group name in the current layout."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in groups:
        return text
    lower = text.lower()
    for name in groups:
        if name.lower() == lower:
            return name
    for kind in MOSQUITO_KINDS:
        if kind.lower() == lower and kind in groups:
            return kind
    if text.isdigit():
        num = int(text)
        if 1 <= num <= len(MOSQUITO_KINDS):
            kind = MOSQUITO_KINDS[num - 1]
            return kind if kind in groups else None
    return None


def parse_deaths_from_csv(
    source,
    max_frames: int,
    groups: dict[str, list[int]],
) -> tuple[list[dict], list[str]]:
    """
    Parse death-calls CSV.

    Expected columns (header row):
      group, mosquito # (in group), death frame
    """
    raw = pd.read_csv(source)
    if raw.empty:
        return [], ["CSV is empty."]

    def norm(name: str) -> str:
        return (
            str(name)
            .strip()
            .lower()
            .replace("#", "")
            .replace("(", "")
            .replace(")", "")
        )

    col_lookup = {norm(c): c for c in raw.columns}

    def pick(*candidates: str) -> str | None:
        for cand in candidates:
            key = norm(cand)
            if key in col_lookup:
                return col_lookup[key]
        return None

    group_col = pick("group", "type")
    mosq_col = pick(
        "mosquito in group",
        "mosquito # in group",
        "mosquito num",
        "mosquito number",
        "index number",
        "index",
    )
    frame_col = pick("death frame", "frame", "death_frame")

    missing = [
        label
        for label, col in (
            ("group", group_col),
            ("mosquito # (in group)", mosq_col),
            ("death frame", frame_col),
        )
        if col is None
    ]
    if missing:
        return [], [
            f"Missing column(s): {', '.join(missing)}. "
            f"Found: {', '.join(str(c) for c in raw.columns)}"
        ]

    entries: list[dict] = []
    warnings: list[str] = []
    seen: set[tuple[str, int]] = set()

    for row_num, row in raw.iterrows():
        line = int(row_num) + 2  # 1-based + header
        group = resolve_group_name(row[group_col], groups)
        if group is None:
            warnings.append(f"Row {line}: could not parse group {row[group_col]!r}")
            continue
        try:
            mosq_num = int(row[mosq_col])
            frame = int(row[frame_col])
        except (ValueError, TypeError):
            warnings.append(f"Row {line}: invalid mosquito # or death frame")
            continue
        if mosq_num not in MOSQUITO_NUMBERS:
            warnings.append(f"Row {line}: mosquito # must be 1–6 (got {mosq_num})")
            continue
        if frame < 0 or frame >= max_frames:
            warnings.append(
                f"Row {line}: death frame {frame} out of range 0–{max_frames - 1}"
            )
            continue
        key = (group, mosq_num)
        if key in seen:
            warnings.append(f"Row {line}: duplicate group/mosquito — keeping latest")
            entries = [e for e in entries if (e["group"], e["mosquito_num"]) != key]
        seen.add(key)
        entries.append(
            {
                "group": group,
                "mosquito_num": mosq_num,
                "death_frame": frame,
            }
        )

    if not entries and not warnings:
        warnings.append("No valid death calls found in CSV.")
    return entries, warnings


def deaths_csv_template() -> str:
    return (
        "group,mosquito # (in group),death frame\n"
        "Female sg (WT),1,5000\n"
        "Male sg (WT),3,7200\n"
    )


def render_group_layout_table(n_mosq: int) -> pd.DataFrame:
    """4-row group layout table: group kind, index spec, hex color."""
    if "group_layout_initialized" not in st.session_state:
        for i, row in DEFAULT_GROUP_LAYOUT.iterrows():
            st.session_state[f"layout_group_{i}"] = str(row["group"])
            st.session_state[f"layout_index_{i}"] = str(row["index"])
            st.session_state[f"layout_color_{i}"] = str(row["color"])
        st.session_state["group_layout_initialized"] = True

    hdr = st.columns([2.2, 2.2, 1.2])
    hdr[0].markdown("**Group**")
    hdr[1].markdown("**Index**")
    hdr[2].markdown("**Color**")

    rows = []
    for i in range(len(DEFAULT_GROUP_LAYOUT)):
        c1, c2, c3 = st.columns([2.2, 2.2, 1.2])
        with c1:
            st.selectbox(
                f"Group row {i + 1}",
                MOSQUITO_KINDS,
                key=f"layout_group_{i}",
                label_visibility="collapsed",
            )
        with c2:
            st.text_input(
                f"Index row {i + 1}",
                key=f"layout_index_{i}",
                label_visibility="collapsed",
                placeholder="0-5 or 0,2,4",
            )
        with c3:
            color_key = f"layout_color_{i}"
            cur = str(st.session_state.get(color_key, DEFAULT_GROUP_LAYOUT.iloc[i]["color"]))
            if not (cur.startswith("#") and len(cur) in (4, 7)):
                st.session_state[color_key] = str(DEFAULT_GROUP_LAYOUT.iloc[i]["color"])
            st.color_picker(
                f"Color row {i + 1}",
                key=color_key,
                label_visibility="collapsed",
            )
        rows.append(
            {
                "group": st.session_state[f"layout_group_{i}"],
                "index": st.session_state[f"layout_index_{i}"],
                "color": st.session_state[f"layout_color_{i}"],
            }
        )

    layout_df = pd.DataFrame(rows)
    groups, group_colors = groups_from_layout_table(layout_df, n_mosq)
    if not groups:
        st.warning("No valid index ranges — check the Index column (e.g. `0-5`).")
    return layout_df, groups, group_colors


def render_group_and_death_controls(
    n_mosq: int,
    max_frames: int,
) -> tuple[dict[str, list[int]], dict[str, str], list[dict], set[int]]:
    """UI for group layout table, mosquito exclusions, and death calls."""
    st.markdown("#### 3. Mosquito-kind layout")
    st.caption(
        "Four rows — pick the mosquito kind, which CSV indices belong to it "
        "(`0-5` or `0,2,4`), and a color."
    )

    _, groups, group_colors = render_group_layout_table(n_mosq)

    st.markdown("#### 3b. Exclude mosquitoes (optional)")
    st.caption("Drop specific wells (e.g. dead-on-arrival or noisy) from every graph and stat.")
    excluded = render_exclusion_control(groups, key="exclude_mosq_single")

    st.markdown("#### 4. Death calls (optional)")
    st.caption(
        f"Upload a CSV or add manually. Death frame is the raw frame index (0–{max_frames - 1})."
    )

    deaths_key = "death_calls_list"
    if deaths_key not in st.session_state:
        st.session_state[deaths_key] = []

    upload_col, template_col = st.columns([3, 1])
    with upload_col:
        death_csv = st.file_uploader(
            "Upload death calls CSV",
            type=["csv"],
            key="death_calls_csv_upload",
            help='Columns: group, mosquito # (in group), death frame',
        )
    with template_col:
        st.download_button(
            "Download template",
            data=deaths_csv_template(),
            file_name="death_calls_template.csv",
            mime="text/csv",
            key="death_csv_template",
        )

    load_col, replace_col, _ = st.columns([1, 1, 2])
    with load_col:
        load_csv = st.button(
            "Load from CSV",
            key="death_load_csv",
            disabled=death_csv is None,
            use_container_width=True,
        )
    with replace_col:
        merge_csv = st.button(
            "Merge with existing",
            key="death_merge_csv",
            disabled=death_csv is None,
            use_container_width=True,
        )

    if load_csv or merge_csv:
        parsed, parse_msgs = parse_deaths_from_csv(death_csv, max_frames, groups)
        if parse_msgs and not parsed:
            for msg in parse_msgs:
                st.error(msg)
        else:
            if merge_csv:
                merged = {(e["group"], e["mosquito_num"]): e for e in st.session_state[deaths_key]}
                for e in parsed:
                    merged[(e["group"], e["mosquito_num"])] = e
                st.session_state[deaths_key] = list(merged.values())
            else:
                st.session_state[deaths_key] = parsed
            if parsed:
                st.success(f"Loaded {len(parsed)} death call(s) from CSV.")
            for msg in parse_msgs:
                st.warning(msg)
            st.rerun()

    st.caption(
        "CSV format — header row: `group`, `mosquito # (in group)`, `death frame`. "
        "Group is a mosquito kind name (e.g. `Female sg (WT)`)."
    )

    active_groups = list(groups.keys())
    if not active_groups:
        return groups, group_colors, st.session_state[deaths_key], excluded

    st.markdown("**Or add one manually**")
    hdr1, hdr2, hdr3, _ = st.columns([2, 1, 2, 1])
    hdr1.markdown("Group")
    hdr2.markdown("Mosquito #")
    hdr3.markdown(f"Death frame (0–{max_frames - 1})")

    add_col1, add_col2, add_col3, add_col4 = st.columns([2, 1, 2, 1])
    with add_col1:
        add_group = st.selectbox(
            "Group",
            active_groups,
            key="death_add_group",
            label_visibility="collapsed",
        )
    with add_col2:
        group_indices = groups.get(add_group, [])
        max_mosq_pick = min(len(group_indices), len(MOSQUITO_NUMBERS))
        mosq_options = list(range(1, max_mosq_pick + 1)) if max_mosq_pick else [1]

        def _mosq_label(n: int) -> str:
            if n <= len(group_indices):
                idx = group_indices[n - 1]
                return f"#{n} (mosquito_{idx})"
            return f"#{n}"

        add_mosq_num = st.selectbox(
            "Mosquito #",
            mosq_options,
            format_func=_mosq_label,
            key="death_add_mosq",
            label_visibility="collapsed",
        )
    with add_col3:
        add_frame = st.number_input(
            "Death frame",
            min_value=0,
            max_value=max(0, max_frames - 1),
            value=0,
            step=1,
            key="death_add_frame",
            label_visibility="collapsed",
        )
    with add_col4:
        if st.button("Add", key="death_add_btn", use_container_width=True):
            entry = {
                "group": add_group,
                "mosquito_num": int(add_mosq_num),
                "death_frame": int(add_frame),
            }
            st.session_state[deaths_key] = [
                e
                for e in st.session_state[deaths_key]
                if not (
                    e["group"] == entry["group"]
                    and e["mosquito_num"] == entry["mosquito_num"]
                )
            ] + [entry]
            st.rerun()

    deaths = st.session_state[deaths_key]
    if deaths:
        with st.expander(f"Current death calls ({len(deaths)})", expanded=False):
            for i, entry in enumerate(deaths):
                c1, c2, c3, c4 = st.columns([2, 1, 2, 1])
                c1.write(entry["group"])
                c2.write(f"#{entry['mosquito_num']}")
                c3.write(str(entry["death_frame"]))
                if c4.button("Remove", key=f"death_rm_{i}"):
                    st.session_state[deaths_key] = [
                        e for j, e in enumerate(deaths) if j != i
                    ]
                    st.rerun()
            if st.button("Clear all death calls", key="death_clear_all"):
                st.session_state[deaths_key] = []
                st.rerun()
    else:
        st.caption("No death calls added yet.")

    return groups, group_colors, deaths, excluded


def apply_death_cut(
    trace: np.ndarray,
    idx: int,
    death_bins: dict[int, int],
    start_zt: float,
    exclude_hours: float = 0.0,
) -> np.ndarray:
    # ``death_bins[idx]`` is already a bin index (death_frame // bin_size) on the
    # same axis as ``trace``/``counts`` (measured from frame 0). ``start_zt`` only
    # shifts the x-axis *labels*, not the data, so it must NOT be subtracted here —
    # doing so cut every trace ~start_zt bins too early and made these figures
    # disagree with the Section D phase totals (which cut at ``i >= death_bin``).
    del start_zt  # kept for call-site compatibility; intentionally unused
    trace = np.asarray(trace, dtype=float).copy()
    if idx not in death_bins:
        return trace
    cut = int(np.floor(death_bins[idx] - exclude_hours))
    cut = int(np.clip(cut, 0, len(trace)))
    trace[cut:] = np.nan
    return trace


def group_mean_trace(
    counts: list[np.ndarray],
    indices: range,
    death_bins: dict[int, int],
    start_zt: float,
    exclude_hours: float = 0.0,
) -> np.ndarray:
    traces = [
        apply_death_cut(counts[i], i, death_bins, start_zt, exclude_hours)
        for i in indices
    ]
    if not traces:
        return np.array([])
    length = max(len(t) for t in traces)
    padded = [np.concatenate([t, np.full(length - len(t), np.nan)]) for t in traces]
    return np.nanmean(padded, axis=0)


def fold_24h(
    counts: list[np.ndarray],
    indices: range,
    start_zt: float,
    period: int,
    lo: int,
    hi: int | None,
    death_bins: dict[int, int],
) -> np.ndarray:
    rows = []
    for m in indices:
        acc = np.zeros(period)
        wgt = np.zeros(period)
        trace = counts[m]
        end = len(trace) if hi is None else min(hi, len(trace))
        death = death_bins.get(m, math.inf)
        for i in range(lo, end):
            if i >= death:
                continue
            val = trace[i]
            if not np.isfinite(val):  # skip pre-start / gap / post-death NaNs
                continue
            zt = (start_zt + i) % period
            low = int(np.floor(zt)) % period
            high = (low + 1) % period
            frac = zt - np.floor(zt)
            acc[low] += val * (1 - frac)
            acc[high] += val * frac
            wgt[low] += 1 - frac
            wgt[high] += frac
        rows.append(np.divide(acc, wgt, out=np.full(period, np.nan), where=wgt > 0))
    return np.array(rows)


def fold_mean_bar(
    counts: list[np.ndarray],
    indices: range,
    start_zt: float,
    period: int,
    lo: int,
    hi: int | None,
    death_bins: dict[int, int],
) -> np.ndarray:
    width = (len(counts[indices[0]]) if hi is None else hi) - lo
    width = max(width, 0)
    if width == 0:
        return np.zeros(period)
    traces = np.array(
        [
            apply_death_cut(
                np.asarray(counts[i], dtype=float), i, death_bins, start_zt
            )[lo : lo + width]
            for i in indices
        ]
    )
    t = np.arange(lo, lo + traces.shape[1])
    zt_hour = (t + start_zt) % period
    out = np.zeros(period)
    for h in range(period):
        mask = (zt_hour >= h) & (zt_hour < h + 1)
        if np.any(mask):
            out[h] = np.nanmean(traces[:, mask])
    return out


def shade_dark_phases(ax, ld_end: int, x_end: float, period: int) -> None:
    """Shade ZT night halves on a ZT-offset x-axis (x ≈ start_zt + bin index).

    Night = [period/2, period) within each circadian day. Continues through DD as
    *subjective* night (same ZT halves) — does **not** paint all of DD solid grey.

    ``ld_end`` is kept for call-site compatibility; shading is driven by ZT only.
    """
    del ld_end  # boundary is encoded in the plot data / phase totals, not shading
    half = period // 2
    start = float(half)
    while start < x_end:
        ax.axvspan(start, min(start + half, x_end), color="grey", alpha=0.3)
        start += period


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    return buf.read()


def show_and_offer(fig, filename: str, key: str, *, show: bool = True):
    """Display + download a figure, or return it for compare/stitch mode."""
    if not show:
        return fig
    st.pyplot(fig)
    st.download_button(
        "Download PNG",
        data=fig_to_png_bytes(fig),
        file_name=filename,
        mime="image/png",
        key=key,
    )
    plt.close(fig)
    return None


def actogram_grid(
    counts, groups, group_colors, idx_to_label, start_zt, period, ld_end,
    death_bins, apply_deaths: bool, title: str, key: str,
    style: PlotStyle | None = None,
) -> None:
    style = style or default_plot_style()
    order = list(groups.keys())
    x_end = start_zt + max((len(c) for c in counts), default=0)

    def trace_for(i):
        if apply_deaths:
            return apply_death_cut(counts[i], i, death_bins, start_zt)
        return np.asarray(counts[i], dtype=float)

    group_ymax = {
        g: max((np.nanmax(trace_for(i)) for i in idxs), default=1.0)
        for g, idxs in groups.items()
    }
    with plt.rc_context(style.rc()):
        for g in order:
            g_label = style.display_group(g)
            st.markdown(f"**{g_label}**")
            idxs = list(groups[g])
            ncol = min(3, len(idxs)) or 1
            nrow = math.ceil(len(idxs) / ncol)
            fig, axes = plt.subplots(
                nrow, ncol, figsize=style.figsize(6 * ncol, 3.2 * nrow), squeeze=False
            )
            for k, idx in enumerate(idxs):
                ax = axes[k // ncol][k % ncol]
                y = trace_for(idx)
                x = start_zt + np.arange(len(y))
                ax.bar(x, y, color=group_colors.get(g, BAR_COLOR), width=1)
                _, sub = idx_to_label[idx]
                ax.set_title(f"{g_label} #{sub}")
                ax.set_ylim(0, group_ymax[g] or 1.0)
                ax.set_xlim(0, x_end)
                ax.axvline(start_zt, color="blue", linestyle="--", linewidth=1.5)
                shade_dark_phases(ax, ld_end, x_end, period)
                ax.set_xticks(np.arange(0, x_end + 1, max(period, 12)))
                ax.set_xlabel(style.x_label("Experimental hour"))
                ax.set_ylabel(style.y_label("Distance moved"))
            for k in range(len(idxs), nrow * ncol):
                axes[k // ncol][k % ncol].axis("off")
            fig.suptitle(style.compose_title(f"{title} — {g_label}"))
            fig.tight_layout()
            show_and_offer(fig, f"{key}_{g}.png", key=f"{key}_{g}")


def death_comparison(
    counts, groups, group_colors, start_zt, period, ld_end, death_bins, key: str,
    style: PlotStyle | None = None,
    show: bool = True,
):
    style = style or default_plot_style()
    order = list(groups.keys())
    means_death = {
        g: group_mean_trace(counts, groups[g], death_bins, start_zt, 0.0)
        for g in order
    }
    means_24h = {
        g: group_mean_trace(counts, groups[g], death_bins, start_zt, 24.0)
        for g in order
    }
    gymax = max(
        (np.nanmax(m) for m in list(means_death.values()) + list(means_24h.values())
         if m.size and np.isfinite(np.nanmax(m))),
        default=1.0,
    )
    x_end = start_zt + max((len(c) for c in counts), default=0)

    with plt.rc_context(style.rc()):
        fig, axes = plt.subplots(
            len(order), 2, figsize=style.figsize(18, 3.4 * len(order)), squeeze=False
        )
        for row, g in enumerate(order):
            g_label = style.display_group(g)
            for col, (means, tag) in enumerate(
                ((means_death, "cut at death"), (means_24h, "cut 24 h before"))
            ):
                ax = axes[row][col]
                y = means[g]
                x = start_zt + np.arange(len(y))
                ax.bar(x, y, color=group_colors.get(g, BAR_COLOR), width=1)
                ax.set_title(f"{g_label}\n({tag})")
                ax.set_ylim(0, gymax or 1.0)
                ax.set_xlim(0, x_end)
                ax.axvline(start_zt, color="blue", linestyle="--", linewidth=1.5)
                shade_dark_phases(ax, ld_end, x_end, period)
                ax.set_xticks(np.arange(0, x_end + 1, max(period, 12)))
                ax.set_xlabel(style.x_label("Experimental hour"))
                ax.set_ylabel(style.y_label("Distance moved"))
        fig.suptitle(style.compose_title("Death comparison (cut at death vs 24 h before)"))
        fig.tight_layout()
        out = show_and_offer(fig, "fig3_death_comparison.png", key=key, show=show)
    if show:
        return means_death
    return means_death, out


def full_period_bar(
    counts, groups, group_colors, start_zt, period, lo, hi, ld_end,
    title: str, xlabel: str, key: str, means_override=None,
    death_bins: dict[int, int] | None = None,
    style: PlotStyle | None = None,
    show: bool = True,
):
    style = style or default_plot_style()
    order = list(groups.keys())
    ncol = min(2, len(order)) or 1
    nrow = math.ceil(len(order) / ncol)
    x_end = start_zt + max((len(c) for c in counts), default=0)
    death_bins = death_bins or {}

    def mean_for(g):
        if means_override is not None:
            return means_override[g]
        traces = np.array(
            [
                apply_death_cut(
                    np.asarray(counts[i], dtype=float), i, death_bins, start_zt
                )
                for i in groups[g]
            ]
        )
        return np.nanmean(traces, axis=0)

    means = {g: mean_for(g) for g in order}
    gymax = max((np.nanmax(m) for m in means.values() if m.size), default=1.0)

    with plt.rc_context(style.rc()):
        fig, axes = plt.subplots(
            nrow, ncol, figsize=style.figsize(11 * ncol, 3.8 * nrow), squeeze=False
        )
        for k, g in enumerate(order):
            ax = axes[k // ncol][k % ncol]
            y_full = means[g]
            x_full = start_zt + np.arange(len(y_full))
            hi_eff = len(y_full) if hi is None else hi
            mask = (np.arange(len(y_full)) >= lo) & (np.arange(len(y_full)) < hi_eff)
            ax.bar(x_full[mask], np.asarray(y_full)[mask],
                   color=group_colors.get(g, BAR_COLOR), width=1)
            ax.set_title(style.display_group(g))
            ax.set_ylim(0, gymax or 1.0)
            left = start_zt + lo if lo > 0 else 0
            right = start_zt + hi_eff
            ax.set_xlim(left, right)
            shade_dark_phases(ax, ld_end, x_end, period)
            ax.set_xticks(np.arange(math.floor(left), right + 1, max(period, 12)))
            ax.set_xlabel(style.x_label(xlabel))
            ax.set_ylabel(style.y_label("Distance moved"))
        for k in range(len(order), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.suptitle(style.compose_title(title))
        fig.tight_layout()
        return show_and_offer(fig, f"{key}.png", key=key, show=show)


def folded_bar(
    counts, groups, group_colors, start_zt, period, lo, hi, death_bins, title: str,
    key: str,
    style: PlotStyle | None = None,
    show: bool = True,
):
    style = style or default_plot_style()
    order = list(groups.keys())
    ncol = min(2, len(order)) or 1
    nrow = math.ceil(len(order) / ncol)
    folded = {
        g: fold_mean_bar(
            counts, groups[g], start_zt, period, lo, hi, death_bins
        )
        for g in order
    }
    gymax = max((f.max() for f in folded.values() if f.size), default=1.0)

    with plt.rc_context(style.rc()):
        fig, axes = plt.subplots(
            nrow, ncol, figsize=style.figsize(10 * ncol, 3.8 * nrow), squeeze=False
        )
        for k, g in enumerate(order):
            ax = axes[k // ncol][k % ncol]
            ax.bar(np.arange(period), folded[g], color=group_colors.get(g, BAR_COLOR))
            ax.set_title(f"{style.display_group(g)} (24 h-folded)")
            ax.set_xlim(0, period)
            ax.set_ylim(0, gymax or 1.0)
            ax.set_xticks(np.arange(0, period + 1, max(period // 4, 1)))
            ax.axvspan(period // 2, period, color="grey", alpha=0.3)
            ax.set_xlabel(style.x_label("ZT (hours)"))
            ax.set_ylabel(style.y_label("Activity"))
        for k in range(len(order), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.suptitle(style.compose_title(title))
        fig.tight_layout()
        return show_and_offer(fig, f"{key}.png", key=key, show=show)


def folded_line(
    counts, groups, group_colors, start_zt, period, lo, hi, death_bins, title: str,
    key: str,
    style: PlotStyle | None = None,
    show: bool = True,
):
    style = style or default_plot_style()
    order = list(groups.keys())
    folded = {
        g: fold_24h(counts, groups[g], start_zt, period, lo, hi, death_bins)
        for g in order
    }
    gymax = 1.0
    for arr in folded.values():
        if arr.size:
            top = np.nanmax(np.nanmean(arr, axis=0) + np.nanstd(arr, axis=0))
            if np.isfinite(top):
                gymax = max(gymax, top)

    ncol = min(2, len(order)) or 1
    nrow = math.ceil(len(order) / ncol)
    zt = np.arange(period)
    with plt.rc_context(style.rc()):
        fig, axes = plt.subplots(
            nrow, ncol, figsize=style.figsize(9 * ncol, 4 * nrow), squeeze=False
        )
        for k, g in enumerate(order):
            ax = axes[k // ncol][k % ncol]
            arr = folded[g]
            if arr.size:
                mean = np.nanmean(arr, axis=0)
                std = np.nanstd(arr, axis=0)
                c = group_colors.get(g, BAR_COLOR)
                ax.plot(zt, mean, color=c, linewidth=2, label="Mean")
                ax.fill_between(
                    zt,
                    np.maximum(mean - std, 0),
                    mean + std,
                    color=c,
                    alpha=0.3,
                    label="±1 SD",
                )
            ax.axvspan(period // 2, period, color="grey", alpha=0.3)
            ax.set_title(style.display_group(g))
            ax.set_xlim(0, period - 1)
            ax.set_ylim(0, gymax)
            ax.set_xticks(np.arange(0, period, max(2, period // 12)))
            ax.set_xlabel(style.x_label("ZT"))
            ax.set_ylabel(style.y_label("Distance moved"))
            ax.legend(loc="upper right")
        for k in range(len(order), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.suptitle(style.compose_title(title))
        fig.tight_layout()
        return show_and_offer(fig, f"{key}.png", key=key, show=show)


def find_default_csv() -> str:
    if EXPERIMENTS_DIR.exists():
        hits = sorted(EXPERIMENTS_DIR.glob("27 *box1/*activity*.csv"))
        if hits:
            return str(hits[-1])
        any_hits = sorted(EXPERIMENTS_DIR.glob("**/*activity*.csv"))
        if any_hits:
            return str(any_hits[0])
    return ""


def render_graphs_sidebar() -> dict:
    """Sidebar controls for activity graphs."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 1. Data source")
    st.sidebar.caption("Upload the activity CSV from inference (`frame, mosquito_0, …`).")
    uploaded = st.sidebar.file_uploader(
        "Upload activity CSV",
        type=["csv"],
        key="graphs_csv_upload",
    )

    st.sidebar.markdown("#### 2. Timing parameters")
    st.sidebar.caption("How to bin frames and align to ZT / LD-DD.")
    start_zt = st.sidebar.number_input(
        "ZT offset (ZT of first bin)",
        value=9.54,
        step=0.01,
        format="%.2f",
        key="g_start_zt",
    )
    bin_size = st.sidebar.number_input(
        "Bin size (frames per bin)",
        value=60,
        min_value=1,
        step=1,
        key="g_bin_size",
    )
    period = st.sidebar.number_input(
        "Circadian period (h)",
        value=24,
        min_value=1,
        step=1,
        key="g_period",
    )
    ld_end = st.sidebar.number_input(
        "LD → DD switch (experimental hour)",
        value=96,
        min_value=0,
        step=1,
        key="g_ld_end",
    )

    st.sidebar.markdown("#### Plot style")
    st.sidebar.caption("Applies to every figure on this page.")
    style = render_plot_style_controls()

    return {
        "uploaded": uploaded,
        "start_zt": start_zt,
        "bin_size": bin_size,
        "period": period,
        "ld_end": ld_end,
        "style": style,
    }


def render_plot_style_controls() -> PlotStyle:
    """Sidebar controls for titles, fonts, and figure size."""
    graph_title = st.sidebar.text_input(
        "Graph title",
        value="",
        placeholder="e.g. ClockKO rep6 box1",
        key="g_graph_title",
    )
    xlabel = st.sidebar.text_input(
        "X-axis label (optional)",
        value="",
        placeholder="leave blank for figure default",
        key="g_xlabel",
    )
    ylabel = st.sidebar.text_input(
        "Y-axis label (optional)",
        value="",
        placeholder="leave blank for figure default",
        key="g_ylabel",
    )
    font_family = st.sidebar.selectbox(
        "Font",
        FONT_CHOICES,
        index=0,
        key="g_font_family",
    )
    width_scale = st.sidebar.slider(
        "Figure width",
        min_value=0.5,
        max_value=2.5,
        value=1.0,
        step=0.05,
        key="g_width_scale",
    )
    height_scale = st.sidebar.slider(
        "Figure height",
        min_value=0.5,
        max_value=2.5,
        value=1.0,
        step=0.05,
        key="g_height_scale",
    )
    font_size = st.sidebar.slider(
        "Body text size",
        min_value=6,
        max_value=20,
        value=10,
        key="g_font_size",
    )
    title_size = st.sidebar.slider(
        "Title size",
        min_value=8,
        max_value=28,
        value=14,
        key="g_title_size",
    )
    label_size = st.sidebar.slider(
        "Axis label size",
        min_value=6,
        max_value=22,
        value=11,
        key="g_label_size",
    )
    tick_size = st.sidebar.slider(
        "Tick number size",
        min_value=5,
        max_value=18,
        value=9,
        key="g_tick_size",
    )

    with st.sidebar.expander("Rename groups on plots", expanded=False):
        st.caption("Blank keeps the names from the mosquito-kind layout.")
        group_title_overrides: dict[str, str] = {}
        for kind in MOSQUITO_KINDS:
            group_title_overrides[kind] = st.text_input(
                kind,
                value="",
                key=f"g_title_{kind}",
                placeholder=kind,
            )

    return PlotStyle(
        graph_title=graph_title,
        group_titles={k: v for k, v in group_title_overrides.items() if str(v).strip()},
        xlabel=xlabel,
        ylabel=ylabel,
        width_scale=float(width_scale),
        height_scale=float(height_scale),
        font_family=str(font_family),
        font_size=float(font_size),
        title_size=float(title_size),
        label_size=float(label_size),
        tick_size=float(tick_size),
    )


# Compare-view figure id -> the fig_header fig_id whose per-figure overrides apply.
def render_activity_graphs_body(settings: dict) -> None:
    """Main panel for all activity figures."""
    uploaded = settings["uploaded"]
    start_zt = settings["start_zt"]
    bin_size = settings["bin_size"]
    period = settings["period"]
    ld_end = settings["ld_end"]

    if uploaded is None:
        st.info("Upload an activity CSV in the sidebar to generate graphs.")
        return

    try:
        data = load_activity(uploaded)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        return

    counts = build_counts(data, int(bin_size))
    n_mosq = len(counts)
    trace_len = max((len(c) for c in counts), default=0)
    label = getattr(uploaded, "name", "uploaded file")
    st.success(
        f"Loaded **{n_mosq} mosquitoes · {trace_len} bins** "
        f"(≈ {trace_len} h at bin size {int(bin_size)}) from `{label}`."
    )

    style = settings.get("style") or default_plot_style()
    if style.graph_title.strip():
        st.markdown(f"## {style.graph_title.strip()}")

    period_i = int(period)
    ld_end_i = int(ld_end)
    max_frames = max((len(f) for f in data["frames"]), default=0)

    groups, group_colors, deaths, excluded = render_group_and_death_controls(
        n_mosq, max_frames
    )
    if not groups:
        st.warning("Define at least one valid group row to see plots.")
        return
    # Death calls map (group, mosquito #) → index, so resolve them against the
    # full groups before exclusions shift within-group positions.
    death_bins = death_bin_by_idx(deaths, groups, int(bin_size))
    groups, group_colors = apply_mosquito_exclusions(groups, group_colors, excluded)
    if not groups:
        st.warning("Every mosquito is excluded — clear some exclusions to see plots.")
        return
    idx_to_label = build_idx_to_label(groups)

    sec_a, sec_b, sec_c = st.tabs(
        [
            "Section A — General",
            "Section B — LD",
            "Section C — DD",
        ]
    )

    means_death: dict = {}

    with sec_a:
        s_fig1 = fig_header(
            "Fig 1 — Individual actograms (pre-death-cut)",
            fig_id="fig1",
            base_style=style,
            customize=True,
            what="One subplot per mosquito: hourly distance over the full experiment (no death trimming).",
            how="""
- **Bars** = distance moved in each time bin (default ~1 hour).
- **Grey shading** = night / subjective-night halves (ZT period/2–period), including in DD.
- **Blue dashed line** = ZT offset / start of the first bin.
- Use this to spot noisy wells, long gaps, or mosquitoes that look dead early.
""",
            ask="Does each mosquito show clear day–night structure before any death cut?",
        )
        actogram_grid(
            counts, groups, group_colors, idx_to_label, start_zt, period_i, ld_end_i,
            death_bins, apply_deaths=False,
            title="Individual actograms (pre-death-cut)", key="fig1",
            style=s_fig1,
        )
        st.divider()
        s_fig2 = fig_header(
            "Fig 2 — Individual actograms (death-cut)",
            fig_id="fig2",
            base_style=style,
            customize=True,
            what="Same as Fig 1, but activity after each mosquito’s death frame is removed (set to missing).",
            how="""
- Requires death calls in section 4 above.
- After the death bin, bars disappear (NaN) so late “noise” doesn’t inflate totals.
- Compare to Fig 1 to see what the cut removed.
""",
            ask="Did the death cut remove only post-death junk, or also useful activity?",
        )
        actogram_grid(
            counts, groups, group_colors, idx_to_label, start_zt, period_i, ld_end_i,
            death_bins, apply_deaths=True,
            title="Individual actograms (death-cut)", key="fig2",
            style=s_fig2,
        )
        st.divider()
        s_fig3 = fig_header(
            "Fig 3 — Death comparison",
            fig_id="fig3",
            base_style=style,
            customize=True,
            what="Group-average traces with two cut rules: at death vs 24 h before death.",
            how="""
- **Left column:** cut exactly at the recorded death time.
- **Right column:** cut 24 hours earlier (stricter — excludes the last day of life).
- Useful to check whether “death” effects are driven by the final day.
""",
            ask="Do group patterns change a lot if you cut 24 h earlier?",
        )
        means_death = death_comparison(
            counts, groups, group_colors, start_zt, period_i, ld_end_i,
            death_bins, key="fig3", style=s_fig3,
        )

    with sec_b:
        help_popover(
            "**LD** = bins before the LD→DD switch.\n\n"
            "- **Day** = unshaded ZT half (ZT 0 – period/2)\n"
            "- **Night** = grey band (ZT period/2 – period)",
            label="❓ About LD (light–dark)",
        )
        s_fig4 = fig_header(
            "Fig 4 — Full LD period",
            fig_id="fig4",
            base_style=style,
            customize=True,
            what="Group-mean activity across the LD portion of the experiment (timeline, not folded).",
            how="""
- Each panel is one group’s average across mosquitoes.
- Activity after each mosquito’s death call is excluded (same rule as Fig 7).
- X-axis is experimental / ZT-aligned time within LD only.
- Grey bands mark night halves of each day.
""",
            ask="Where is activity highest in LD — day, night, or dawn/dusk transitions?",
        )
        full_period_bar(
            counts, groups, group_colors, start_zt, period_i,
            lo=0, hi=ld_end_i, ld_end=ld_end_i,
            title="Full LD period, averaged across mosquitoes",
            xlabel="ZT / experimental hour", key="fig4", style=s_fig4,
            death_bins=death_bins,
        )
        st.divider()
        s_fig5 = fig_header(
            "Fig 5 — LD, 24 h-folded",
            fig_id="fig5",
            base_style=style,
            customize=True,
            what="All LD days stacked into one average 24 h (ZT 0–24) profile per group.",
            how="""
- Collapses multi-day LD into a single circadian day.
- Bins after each mosquito’s death call are excluded.
- Grey = subjective / scheduled night (ZT period/2–period).
- Good for comparing waveform shape across genotypes/sexes.
""",
            ask="Do WT and KO share the same peak timing in LD?",
        )
        folded_bar(
            counts, groups, group_colors, start_zt, period_i,
            lo=0, hi=ld_end_i, death_bins=death_bins,
            title="LD, 24 h-folded", key="fig5", style=s_fig5,
        )
        st.divider()
        s_fig6 = fig_header(
            "Fig 6 — LD mean ± 1 SD",
            fig_id="fig6",
            base_style=style,
            customize=True,
            what="Same 24 h-folded LD profile as Fig 5, shown as mean ± 1 SD across mosquitoes.",
            how="""
- Line = group mean; ribbon = ± 1 SD between mosquitoes.
- Only live (pre-death) bins contribute per mosquito.
- Wide ribbons mean mosquitoes disagree; tight ribbons mean a consistent group pattern.
""",
            ask="Is the LD rhythm consistent within each group, or very mosquito-to-mosquito?",
        )
        folded_line(
            counts, groups, group_colors, start_zt, period_i,
            lo=0, hi=ld_end_i, death_bins=death_bins,
            title="LD (24 h-folded) mean ± 1 SD", key="fig6",
            style=s_fig6,
        )

    with sec_c:
        if not means_death:
            means_death = {
                g: group_mean_trace(counts, groups[g], death_bins, start_zt, 0.0)
                for g in groups
            }
        help_popover(
            "**DD** = bins after the LD→DD switch (lights stay off).\n\n"
            "- **DD subjective day / night** use the same ZT halves as the grey bands "
            "(not actual light)\n"
            "- Subjective day = unshaded, subjective night = grey",
            label="❓ About DD (dark–dark)",
        )
        s_fig7 = fig_header(
            "Fig 7 — Full DD period",
            fig_id="fig7",
            base_style=style,
            customize=True,
            what="Group-mean activity after the LD→DD switch (death-cut means).",
            how="""
- Timeline of free-running / constant-dark portion only.
- Uses death-cut group means so late deaths don’t dominate.
- Grey bands continue as **subjective night** (same ZT halves as LD), not solid grey for all of DD.
""",
            ask="Does rhythmic activity persist in DD, or flatten out?",
        )
        full_period_bar(
            counts, groups, group_colors, start_zt, period_i,
            lo=ld_end_i, hi=None, ld_end=ld_end_i,
            title="Full DD period, averaged across mosquitoes",
            xlabel="Experimental hour", key="fig7", means_override=means_death,
            style=s_fig7,
        )
        st.divider()
        s_fig8 = fig_header(
            "Fig 8 — DD, 24 h-folded",
            fig_id="fig8",
            base_style=style,
            customize=True,
            what="DD days folded into one average 24 h ZT profile per group.",
            how="""
- Same folding idea as Fig 5, but only using post-switch bins.
- Bins after each mosquito’s death call are excluded.
- Grey = ZT night half (even though lights are off in DD).
""",
            ask="Is the DD peak still aligned with the LD night, or did it drift?",
        )
        folded_bar(
            counts, groups, group_colors, start_zt, period_i,
            lo=ld_end_i, hi=None, death_bins=death_bins,
            title="DD, 24 h-folded", key="fig8", style=s_fig8,
        )
        st.divider()
        s_fig9 = fig_header(
            "Fig 9 — DD mean ± 1 SD",
            fig_id="fig9",
            base_style=style,
            customize=True,
            what="DD 24 h-folded mean ± 1 SD across mosquitoes.",
            how="""
- Compare ribbon width to Fig 6: DD is often noisier.
- Only live (pre-death) bins contribute per mosquito.
- Useful for seeing whether a genotype keeps a coherent free-running rhythm.
""",
            ask="Which groups still look rhythmic in DD?",
        )
        folded_line(
            counts, groups, group_colors, start_zt, period_i,
            lo=ld_end_i, hi=None, death_bins=death_bins,
            title="DD (24 h-folded) mean ± 1 SD", key="fig9",
            style=s_fig9,
        )
