"""Activity analysis plots (copy of activity_explorer.py logic for cloud_inference)."""

from __future__ import annotations

import html
import io
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy import stats

from mosquito_lab.paths import mosquito_project_dir

EXPERIMENTS_DIR = mosquito_project_dir() / "experiments"

DEFAULT_GROUPS = pd.DataFrame(
    [
        {"name": "Female sg (WT)", "start": 0, "end": 6, "color": "#e63946"},
        {"name": "Male sg (WT)", "start": 6, "end": 12, "color": "#457b9d"},
        {"name": "Female KO", "start": 12, "end": 18, "color": "#f4a261"},
        {"name": "Male KO", "start": 18, "end": 24, "color": "#2a9d8f"},
    ]
)

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

PHASE_ORDER = ("LD day", "LD night", "DD day", "DD night")
PHASE_COLORS = {
    "LD day": "#f4d35e",
    "LD night": "#335c67",
    "DD day": "#e9c46a",
    "DD night": "#1b263b",
}
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


def fig_guide(what: str, how: str, ask: str | None = None) -> None:
    """Deprecated — prefer fig_header / _title_with_help."""
    _title_with_help("", what=what, how=how, ask=ask)


def stats_glossary_help() -> str:
    return """
**Stars (`* / ** / *** / ns`)**
- `ns` = not significant (p ≥ 0.05)
- `*` = p < 0.05
- `**` = p < 0.01
- `***` = p < 0.001

**Wilcoxon signed-rank (within-group day vs night)**
- Compares **paired** values from the **same** mosquitoes (e.g. each mosquito’s LD-day total vs its LD-night total).
- Non-parametric (does not assume normality).
- Used here when a group has **n ≥ 6**.
- Small p → day and night totals differ systematically for that group.

**Paired t-test**
- Same idea as Wilcoxon (paired day vs night), but assumes roughly normal differences.
- Used here as a fallback when **n < 6**.

**Kruskal–Wallis (across all groups)**
- Asks: “Do **any** of these groups differ?” for one phase (e.g. LD night).
- Non-parametric analog of one-way ANOVA.
- Small p → at least one group’s distribution differs from another’s — not which pair.

**Mann–Whitney U (pairwise between groups)**
- Compares **two** independent groups (different mosquitoes), e.g. Female WT vs Female KO.
- Non-parametric analog of a two-sample t-test.
- Small p → the two groups’ totals for that phase differ.
- The heatmap shows these pairwise p-values (green = more significant / smaller p on the color scale).

**Caveats**
- These are exploratory summaries; they do not correct for testing many phases/pairs at once.
- Small n and death cuts reduce power. Interpret stars as guides, not definitive biology.
"""


def stats_glossary() -> None:
    """Plain-language explainers for the Section D tests (hover help)."""
    _title_with_help(
        "Stats guide",
        level="####",
        what="What the significance tests and stars mean.",
        how=stats_glossary_help(),
    )


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
) -> tuple[dict[str, list[int]], dict[str, str], list[dict]]:
    """UI for group layout table and death calls."""
    st.markdown("#### 3. Mosquito-kind layout")
    st.caption(
        "Four rows — pick the mosquito kind, which CSV indices belong to it "
        "(`0-5` or `0,2,4`), and a color."
    )

    _, groups, group_colors = render_group_layout_table(n_mosq)

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
        return groups, group_colors, st.session_state[deaths_key]

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

    return groups, group_colors, deaths


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
) -> np.ndarray:
    width = (len(counts[indices[0]]) if hi is None else hi) - lo
    width = max(width, 0)
    if width == 0:
        return np.zeros(period)
    traces = np.array(
        [np.asarray(counts[i], dtype=float)[lo : lo + width] for i in indices]
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


def classify_bin_phase(bin_idx: int, start_zt: float, period: int, ld_end: int) -> str:
    """Classify a bin as LD/DD day or night using the same ZT bands as grey shading.

    - **Day** (unshaded): ``ZT < period/2``
    - **Night** (grey band): ``ZT >= period/2``
    - **LD vs DD**: bin index vs ``ld_end``
    """
    zt = (start_zt + bin_idx) % period
    is_day = zt < (period / 2)
    if bin_idx >= ld_end:
        return "DD day" if is_day else "DD night"
    return "LD day" if is_day else "LD night"


def phase_totals_table(
    counts: list[np.ndarray],
    groups: dict[str, list[int]],
    death_bins: dict[int, int],
    start_zt: float,
    period: int,
    ld_end: int,
) -> pd.DataFrame:
    """Per-mosquito summed pixel distance by LD/DD × day/night (ZT bands)."""
    rows: list[dict] = []
    for group_name, indices in groups.items():
        for j, idx in enumerate(indices):
            totals = {phase: 0.0 for phase in PHASE_ORDER}
            death = death_bins.get(idx, math.inf)
            for i, val in enumerate(counts[idx]):
                if i >= death:
                    break
                if not np.isfinite(val):
                    continue
                phase = classify_bin_phase(i, start_zt, period, ld_end)
                totals[phase] += float(val)
            rows.append(
                {
                    "group": group_name,
                    "mosquito": j + 1,
                    "mosquito_idx": idx,
                    **totals,
                    "LD total": totals["LD day"] + totals["LD night"],
                    "DD total": totals["DD day"] + totals["DD night"],
                }
            )
    return pd.DataFrame(rows)


def _p_stars(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def phase_summary_by_group(totals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name, sub in totals.groupby("group", sort=False):
        row: dict = {"group": group_name, "n": len(sub)}
        for col in list(PHASE_ORDER) + ["LD total", "DD total"]:
            vals = sub[col].to_numpy(dtype=float)
            row[f"{col} mean"] = float(np.nanmean(vals))
            row[f"{col} SD"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def within_group_day_night_tests(totals: pd.DataFrame) -> pd.DataFrame:
    """Paired day vs night tests within each group for LD and DD."""
    rows = []
    comparisons = (
        ("LD", "LD day", "LD night"),
        ("DD", "DD day", "DD night"),
    )
    for group_name, sub in totals.groupby("group", sort=False):
        for condition, day_col, night_col in comparisons:
            day = sub[day_col].to_numpy(dtype=float)
            night = sub[night_col].to_numpy(dtype=float)
            n = len(sub)
            p = float("nan")
            test = "n/a"
            note = ""
            if n < 2:
                note = "need n≥2"
            elif np.allclose(day, night):
                p = 1.0
                test = "identical"
            else:
                try:
                    if n >= 6:
                        stat = stats.wilcoxon(
                            day, night, zero_method="wilcox", alternative="two-sided"
                        )
                        p = float(stat.pvalue)
                        test = "Wilcoxon signed-rank"
                    else:
                        stat = stats.ttest_rel(day, night, nan_policy="omit")
                        p = float(stat.pvalue)
                        test = "paired t-test (n<6)"
                except Exception as exc:
                    note = str(exc)
            rows.append(
                {
                    "group": group_name,
                    "condition": condition,
                    "comparison": f"{day_col} vs {night_col}",
                    "n": n,
                    "day mean": float(np.nanmean(day)) if n else float("nan"),
                    "night mean": float(np.nanmean(night)) if n else float("nan"),
                    "test": test,
                    "p": p,
                    "sig": _p_stars(p),
                    "note": note,
                }
            )
    return pd.DataFrame(rows)


# Back-compat alias
within_group_light_dark_tests = within_group_day_night_tests


def between_group_phase_tests(totals: pd.DataFrame) -> pd.DataFrame:
    """Kruskal–Wallis across groups for each phase, plus pairwise Mann–Whitney."""
    group_names = list(dict.fromkeys(totals["group"].tolist()))
    rows = []
    for phase in list(PHASE_ORDER) + ["LD total", "DD total"]:
        samples = [
            totals.loc[totals["group"] == g, phase].to_numpy(dtype=float)
            for g in group_names
        ]
        valid = [s for s in samples if len(s) >= 1]
        if len(valid) < 2:
            rows.append(
                {
                    "phase": phase,
                    "comparison": "all groups",
                    "test": "n/a",
                    "p": float("nan"),
                    "sig": "n/a",
                    "note": "need ≥2 groups",
                }
            )
            continue
        try:
            if all(len(s) >= 1 for s in samples) and len(group_names) >= 2:
                h = stats.kruskal(*samples)
                rows.append(
                    {
                        "phase": phase,
                        "comparison": "all groups",
                        "test": "Kruskal–Wallis",
                        "p": float(h.pvalue),
                        "sig": _p_stars(float(h.pvalue)),
                        "note": "",
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "phase": phase,
                    "comparison": "all groups",
                    "test": "Kruskal–Wallis",
                    "p": float("nan"),
                    "sig": "n/a",
                    "note": str(exc),
                }
            )
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                a = samples[i]
                b = samples[j]
                if len(a) < 1 or len(b) < 1:
                    continue
                try:
                    u = stats.mannwhitneyu(a, b, alternative="two-sided")
                    p = float(u.pvalue)
                    rows.append(
                        {
                            "phase": phase,
                            "comparison": f"{group_names[i]} vs {group_names[j]}",
                            "test": "Mann–Whitney U",
                            "p": p,
                            "sig": _p_stars(p),
                            "note": "",
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "phase": phase,
                            "comparison": f"{group_names[i]} vs {group_names[j]}",
                            "test": "Mann–Whitney U",
                            "p": float("nan"),
                            "sig": "n/a",
                            "note": str(exc),
                        }
                    )
    return pd.DataFrame(rows)


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


def stitch_figures(
    fig_a,
    fig_b,
    *,
    layout: str = "side-by-side",
) -> bytes:
    """Combine two matplotlib figures into one PNG (side-by-side or stacked)."""
    img_a = Image.open(io.BytesIO(fig_to_png_bytes(fig_a))).convert("RGB")
    img_b = Image.open(io.BytesIO(fig_to_png_bytes(fig_b))).convert("RGB")
    if layout == "stacked":
        width = max(img_a.width, img_b.width)
        height = img_a.height + img_b.height + 16
        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        canvas.paste(img_a, ((width - img_a.width) // 2, 0))
        canvas.paste(img_b, ((width - img_b.width) // 2, img_a.height + 16))
    else:
        height = max(img_a.height, img_b.height)
        width = img_a.width + img_b.width + 16
        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        canvas.paste(img_a, (0, (height - img_a.height) // 2))
        canvas.paste(img_b, (img_a.width + 16, (height - img_b.height) // 2))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


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
    style: PlotStyle | None = None,
    show: bool = True,
):
    style = style or default_plot_style()
    order = list(groups.keys())
    ncol = min(2, len(order)) or 1
    nrow = math.ceil(len(order) / ncol)
    x_end = start_zt + max((len(c) for c in counts), default=0)

    def mean_for(g):
        if means_override is not None:
            return means_override[g]
        traces = np.array([np.asarray(counts[i], dtype=float) for i in groups[g]])
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
    counts, groups, group_colors, start_zt, period, lo, hi, title: str, key: str,
    style: PlotStyle | None = None,
    show: bool = True,
):
    style = style or default_plot_style()
    order = list(groups.keys())
    ncol = min(2, len(order)) or 1
    nrow = math.ceil(len(order) / ncol)
    folded = {g: fold_mean_bar(counts, groups[g], start_zt, period, lo, hi) for g in order}
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
    counts, groups, group_colors, start_zt, period, lo, hi, title: str, key: str,
    style: PlotStyle | None = None,
    show: bool = True,
):
    style = style or default_plot_style()
    order = list(groups.keys())
    folded = {
        g: fold_24h(counts, groups[g], start_zt, period, lo, hi, {}) for g in order
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


def phase_totals_figure(
    totals: pd.DataFrame,
    groups: dict[str, list[int]],
    group_colors: dict[str, str],
    style: PlotStyle,
    key: str,
    show: bool = True,
):
    """Grouped bar chart: mean ± 1 SD total distance by phase for each group."""
    order = list(groups.keys())
    phases = list(PHASE_ORDER)
    x = np.arange(len(order))
    width = 0.22
    with plt.rc_context(style.rc()):
        fig, ax = plt.subplots(figsize=style.figsize(10, 5.5))
        for i, phase in enumerate(phases):
            means = []
            err_lo = []
            err_hi = []
            for g in order:
                vals = totals.loc[totals["group"] == g, phase].to_numpy(dtype=float)
                mean = float(np.nanmean(vals)) if len(vals) else 0.0
                sd = (
                    float(np.nanstd(vals, ddof=1))
                    if len(vals) > 1
                    else 0.0
                )
                means.append(mean)
                # Clip lower whisker at 0 — distance totals can't be negative
                err_lo.append(min(sd, mean))
                err_hi.append(sd)
            offset = (i - (len(phases) - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                means,
                width,
                yerr=np.array([err_lo, err_hi]),
                capsize=3,
                label=phase,
                color=PHASE_COLORS.get(phase, BAR_COLOR),
                edgecolor="white",
                linewidth=0.6,
            )
            for bar, mean in zip(bars, means):
                if mean > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{mean:.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=max(7, style.tick_size - 1),
                    )
        ax.set_xticks(x)
        ax.set_xticklabels([style.display_group(g) for g in order])
        ax.set_ylabel(style.y_label("Mean total distance ± 1 SD (pixel units)"))
        ax.set_xlabel(style.x_label("Group"))
        ax.set_ylim(bottom=0)
        ax.legend(title="Phase", framealpha=0.9)
        ax.set_title(style.compose_title("Mean ± 1 SD — total distance by day/night × LD/DD"))
        ax.text(
            0.99,
            0.98,
            "Error bars = ± 1 SD (floored at 0)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=max(8, style.tick_size),
            style="italic",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        return show_and_offer(fig, f"{key}_phase_totals.png", key=key, show=show)


def phase_totals_box_figure(
    totals: pd.DataFrame,
    groups: dict[str, list[int]],
    style: PlotStyle,
    key: str,
    show: bool = True,
):
    """Box + strip plot of per-mosquito totals by phase (alternative to tables)."""
    order = list(groups.keys())
    phases = list(PHASE_ORDER)
    n_phases = len(phases)
    with plt.rc_context(style.rc()):
        fig, axes = plt.subplots(
            1, n_phases, figsize=style.figsize(3.2 * n_phases, 4.8), squeeze=False
        )
        for col, phase in enumerate(phases):
            ax = axes[0][col]
            data = [
                totals.loc[totals["group"] == g, phase].to_numpy(dtype=float)
                for g in order
            ]
            bp = ax.boxplot(
                data,
                patch_artist=True,
                showfliers=False,
            )
            ax.set_xticks(range(1, len(order) + 1))
            ax.set_xticklabels([style.display_group(g) for g in order])
            for patch, g in zip(bp["boxes"], order):
                patch.set_facecolor(PHASE_COLORS.get(phase, BAR_COLOR))
                patch.set_alpha(0.35)
            for i, (g, vals) in enumerate(zip(order, data), start=1):
                if len(vals) == 0:
                    continue
                jitter = np.random.default_rng(i * 17 + col).uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(
                    np.full(len(vals), i) + jitter,
                    vals,
                    s=28,
                    color=PHASE_COLORS.get(phase, BAR_COLOR),
                    alpha=0.85,
                    zorder=3,
                    edgecolors="white",
                    linewidths=0.4,
                )
            ax.set_title(phase)
            ax.tick_params(axis="x", rotation=25)
            if col == 0:
                ax.set_ylabel(style.y_label("Total distance"))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.suptitle(style.compose_title("Per-mosquito totals by phase"))
        fig.tight_layout()
        return show_and_offer(fig, f"{key}_phase_box.png", key=key, show=show)


def stats_heatmap_figure(
    between: pd.DataFrame,
    style: PlotStyle,
    key: str,
) -> None:
    """Heatmap of pairwise Mann–Whitney p-values by phase."""
    pairwise = between[
        (between["test"] == "Mann–Whitney U") & between["p"].notna()
    ].copy()
    if pairwise.empty:
        st.info("Not enough groups for a pairwise stats heatmap.")
        return

    phases = [p for p in list(PHASE_ORDER) + ["LD total", "DD total"] if p in set(pairwise["phase"])]
    # Collect unique group names from comparison strings
    names: list[str] = []
    for comp in pairwise["comparison"]:
        left, right = [s.strip() for s in str(comp).split(" vs ", 1)]
        for n in (left, right):
            if n not in names:
                names.append(n)

    n = len(names)
    n_ph = len(phases)
    with plt.rc_context(style.rc()):
        fig, axes = plt.subplots(
            1, n_ph, figsize=style.figsize(3.4 * n_ph, 3.6), squeeze=False
        )
        for col, phase in enumerate(phases):
            ax = axes[0][col]
            mat = np.full((n, n), np.nan)
            sub = pairwise[pairwise["phase"] == phase]
            for _, row in sub.iterrows():
                left, right = [s.strip() for s in str(row["comparison"]).split(" vs ", 1)]
                if left in names and right in names:
                    i, j = names.index(left), names.index(right)
                    mat[i, j] = row["p"]
                    mat[j, i] = row["p"]
            np.fill_diagonal(mat, 1.0)
            im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=0.1, aspect="equal")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            labels = [style.display_group(nm) for nm in names]
            ax.set_xticklabels(labels, rotation=40, ha="right")
            ax.set_yticklabels(labels)
            ax.set_title(phase)
            for i in range(n):
                for j in range(n):
                    p = mat[i, j]
                    if not np.isfinite(p):
                        continue
                    ax.text(
                        j,
                        i,
                        _p_stars(p) if i != j else "—",
                        ha="center",
                        va="center",
                        fontsize=max(7, style.tick_size - 1),
                        color="black",
                    )
            if col == n_ph - 1:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="p-value")
        fig.suptitle(style.compose_title("Pairwise significance (Mann–Whitney)"))
        fig.tight_layout()
        show_and_offer(fig, f"{key}_stats_heatmap.png", key=key)


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
_COMPARE_TO_FIGID = {
    "fig3": "fig3",
    "fig4": "fig4",
    "fig5": "fig5",
    "fig6": "fig6",
    "fig7": "fig7",
    "fig8": "fig8",
    "fig9": "fig9",
    "fig10_bars": "fig10_phase",
    "fig10_box": "fig10_box",
}


def build_selected_figure(
    choice: str,
    *,
    counts,
    groups,
    group_colors,
    start_zt,
    period_i,
    ld_end_i,
    death_bins,
    means_death,
    totals,
    style: PlotStyle,
):
    """Build one comparable figure (no Streamlit display). Returns matplotlib Figure."""
    key = f"compare_build_{choice}"
    # Honour each figure's own per-graph customization in the compare/mix view.
    style = read_fig_style(_COMPARE_TO_FIGID.get(choice, ""), style)
    if choice == "fig3":
        _means, fig = death_comparison(
            counts, groups, group_colors, start_zt, period_i, ld_end_i,
            death_bins, key=key, style=style, show=False,
        )
        return fig
    if choice == "fig4":
        return full_period_bar(
            counts, groups, group_colors, start_zt, period_i,
            lo=0, hi=ld_end_i, ld_end=ld_end_i,
            title="Full LD period, averaged across mosquitoes",
            xlabel="ZT / experimental hour", key=key, style=style, show=False,
        )
    if choice == "fig5":
        return folded_bar(
            counts, groups, group_colors, start_zt, period_i,
            lo=0, hi=ld_end_i, title="LD, 24 h-folded", key=key, style=style, show=False,
        )
    if choice == "fig6":
        return folded_line(
            counts, groups, group_colors, start_zt, period_i,
            lo=0, hi=ld_end_i, title="LD (24 h-folded) mean ± 1 SD",
            key=key, style=style, show=False,
        )
    if choice == "fig7":
        return full_period_bar(
            counts, groups, group_colors, start_zt, period_i,
            lo=ld_end_i, hi=None, ld_end=ld_end_i,
            title="Full DD period, averaged across mosquitoes",
            xlabel="Experimental hour", key=key, means_override=means_death,
            style=style, show=False,
        )
    if choice == "fig8":
        return folded_bar(
            counts, groups, group_colors, start_zt, period_i,
            lo=ld_end_i, hi=None, title="DD, 24 h-folded", key=key, style=style, show=False,
        )
    if choice == "fig9":
        return folded_line(
            counts, groups, group_colors, start_zt, period_i,
            lo=ld_end_i, hi=None, title="DD (24 h-folded) mean ± 1 SD",
            key=key, style=style, show=False,
        )
    if choice == "fig10_bars":
        return phase_totals_figure(
            totals, groups, group_colors, style, key=key, show=False,
        )
    if choice == "fig10_box":
        return phase_totals_box_figure(totals, groups, style, key=key, show=False)
    raise ValueError(f"Unknown compare choice: {choice}")


COMPARE_LABELS = {
    "fig3": "Fig 3 — Death comparison",
    "fig4": "Fig 4 — Full LD period",
    "fig5": "Fig 5 — LD, 24 h-folded",
    "fig6": "Fig 6 — LD mean ± 1 SD",
    "fig7": "Fig 7 — Full DD period",
    "fig8": "Fig 8 — DD, 24 h-folded",
    "fig9": "Fig 9 — DD mean ± 1 SD",
    "fig10_bars": "Section D — Mean ± 1 SD by phase",
    "fig10_box": "Section D — Per-mosquito boxplots",
}


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

    groups, group_colors, deaths = render_group_and_death_controls(
        n_mosq, max_frames
    )
    if not groups:
        st.warning("Define at least one valid group row to see plots.")
        return
    idx_to_label = build_idx_to_label(groups)
    death_bins = death_bin_by_idx(deaths, groups, int(bin_size))

    sec_a, sec_b, sec_c, sec_d, sec_e = st.tabs(
        [
            "Section A — General",
            "Section B — LD",
            "Section C — DD",
            "Section D — Day/night totals",
            "Section E — Compare / mix",
        ]
    )

    means_death: dict = {}
    # Precompute totals for Section D + Compare (cheap relative to plotting)
    totals = phase_totals_table(
        counts, groups, death_bins, start_zt, period_i, ld_end_i
    )

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
        with st.expander("About LD (light–dark)", expanded=False):
            st.markdown(
                "**LD** = bins before the LD→DD switch.\n\n"
                "- **Day** = unshaded ZT half (ZT 0 – period/2)\n"
                "- **Night** = grey band (ZT period/2 – period)"
            )
        s_fig4 = fig_header(
            "Fig 4 — Full LD period",
            fig_id="fig4",
            base_style=style,
            customize=True,
            what="Group-mean activity across the LD portion of the experiment (timeline, not folded).",
            how="""
- Each panel is one group’s average across mosquitoes.
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
- Grey = subjective / scheduled night (ZT period/2–period).
- Good for comparing waveform shape across genotypes/sexes.
""",
            ask="Do WT and KO share the same peak timing in LD?",
        )
        folded_bar(
            counts, groups, group_colors, start_zt, period_i,
            lo=0, hi=ld_end_i, title="LD, 24 h-folded", key="fig5", style=s_fig5,
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
- Wide ribbons mean mosquitoes disagree; tight ribbons mean a consistent group pattern.
""",
            ask="Is the LD rhythm consistent within each group, or very mosquito-to-mosquito?",
        )
        folded_line(
            counts, groups, group_colors, start_zt, period_i,
            lo=0, hi=ld_end_i, title="LD (24 h-folded) mean ± 1 SD", key="fig6",
            style=s_fig6,
        )

    with sec_c:
        if not means_death:
            means_death = {
                g: group_mean_trace(counts, groups[g], death_bins, start_zt, 0.0)
                for g in groups
            }
        with st.expander("About DD (dark–dark)", expanded=False):
            st.markdown(
                "**DD** = bins after the LD→DD switch (incubator stays dark).\n\n"
                "- **DD day / DD night** still follow the same ZT bands as the grey shading\n"
                "- Subjective day = unshaded, subjective night = grey"
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
- Grey = ZT night half (even though lights are off in DD).
""",
            ask="Is the DD peak still aligned with the LD night, or did it drift?",
        )
        folded_bar(
            counts, groups, group_colors, start_zt, period_i,
            lo=ld_end_i, hi=None, title="DD, 24 h-folded", key="fig8", style=s_fig8,
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
- Useful for seeing whether a genotype keeps a coherent free-running rhythm.
""",
            ask="Which groups still look rhythmic in DD?",
        )
        folded_line(
            counts, groups, group_colors, start_zt, period_i,
            lo=ld_end_i, hi=None, title="DD (24 h-folded) mean ± 1 SD", key="fig9",
            style=s_fig9,
        )

    with sec_d:
        st.markdown("### Total pixel distance — day vs night × LD / DD")
        ld_bins = max(0, min(ld_end_i, trace_len))
        dd_bins = max(0, trace_len - ld_end_i)
        if dd_bins and ld_bins:
            st.caption(
                f"These are **summed** totals. This experiment has ~**{ld_bins} h of LD** "
                f"vs ~**{dd_bins} h of DD**, so LD totals are larger mostly because the LD "
                "window is longer (and death cuts trim late DD). Compare shapes/ratios, not "
                "raw LD-vs-DD heights."
            )
        with st.expander("How phases & stats are computed", expanded=False):
            st.markdown(
                """
**How phases are defined** (same rule as the grey bands on the graphs)
- **LD vs DD:** bin index vs the LD→DD switch.
- **Day** = unshaded ZT half (`ZT < period/2`).
- **Night** = grey band (`ZT ≥ period/2`).
- Applies in **both** LD and DD (subjective day/night after lights stay off).
- Death cuts stop summing after a mosquito’s death bin.

**What the plots show**
- **Mean ± 1 SD bars:** group average total distance in each phase (error bars floored at 0).
- **Box + points:** every mosquito’s total (spread / outliers).
- **Stats heatmap:** pairwise group differences (Mann–Whitney p-values).
"""
                + "\n\n---\n\n"
                + stats_glossary_help()
            )

        totals_display = totals.copy()
        totals_display["group"] = totals_display["group"].map(style.display_group)

        s_fig10_phase = fig_header(
            "Mean ± 1 SD by phase",
            fig_id="fig10_phase",
            base_style=style,
            level="####",
            customize=True,
            what="Grouped bars: average total distance (± 1 SD) per group for LD day, LD night, DD day, DD night.",
            how="""
- Tall bar = more cumulative movement in that phase.
- Error bars = ± 1 SD across mosquitoes (lower whisker clipped at 0).
- Compare day vs night *within* a group, and the same phase *across* groups.
""",
            ask="Is night activity higher than day activity in every genotype?",
        )
        phase_totals_figure(totals, groups, group_colors, s_fig10_phase, key="fig10_phase")

        s_fig10_box = fig_header(
            "Per-mosquito distribution (box + points)",
            fig_id="fig10_box",
            base_style=style,
            level="####",
            customize=True,
            what="Boxplots with individual mosquito points for each phase.",
            how="""
- Box = middle 50% of mosquitoes; line inside ≈ median.
- Dots = individual mosquitoes (jittered).
- Use this to see if a high mean is driven by one hyperactive mosquito.
""",
            ask="Are group differences driven by most mosquitoes, or one outlier?",
        )
        phase_totals_box_figure(totals, groups, s_fig10_box, key="fig10_box")

        with st.expander("Per-mosquito totals table", expanded=False):
            show_cols = [c for c in totals_display.columns if c != "mosquito_idx"]
            st.dataframe(
                totals_display[show_cols].round(1),
                use_container_width=True,
            )
            st.download_button(
                "Download per-mosquito totals CSV",
                data=totals_display[show_cols].to_csv(index=False),
                file_name="phase_totals_per_mosquito.csv",
                mime="text/csv",
                key="dl_phase_totals",
            )

        summary = phase_summary_by_group(totals)
        summary_display = summary.copy()
        summary_display["group"] = summary_display["group"].map(style.display_group)
        with st.expander("Group means ± 1 SD table", expanded=False):
            st.dataframe(summary_display.round(2), use_container_width=True)

        fig_header(
            "Within-group: day vs night (LD and DD)",
            fig_id="fig10_within",
            base_style=style,
            level="####",
            customize=False,
            what="For each group, test whether day totals differ from night totals (paired per mosquito).",
            how="""
- Uses **Wilcoxon signed-rank** (n≥6) or **paired t-test** (n<6).
- Separate rows for LD and for DD.
- Open the section **?** tip above for plain-language definitions.
""",
            ask="Within a genotype, is night activity significantly higher than day?",
        )
        within = within_group_day_night_tests(totals)
        within_display = within.copy()
        within_display["group"] = within_display["group"].map(style.display_group)
        within_display["p"] = within_display["p"].map(
            lambda p: f"{p:.4g}" if np.isfinite(p) else "n/a"
        )
        st.dataframe(within_display, use_container_width=True)

        s_fig10_heat = fig_header(
            "Between-group comparisons",
            fig_id="fig10_heat",
            base_style=style,
            level="####",
            customize=True,
            what="First Kruskal–Wallis across all groups, then pairwise Mann–Whitney U (heatmap).",
            how="""
- **Kruskal–Wallis** (in the table): any overall group difference for that phase?
- **Mann–Whitney heatmap:** which specific pairs differ? Cells show significance stars.
- Diagonal is blank (a group vs itself).
- Open the section **?** tip for plain-language definitions.
""",
            ask="Which pairwise genotype/sex contrasts are significant in LD night vs DD night?",
        )
        between = between_group_phase_tests(totals)
        stats_heatmap_figure(between, s_fig10_heat, key="fig10_heat")
        with st.expander("Full between-group stats table", expanded=False):
            between_display = between.copy()
            between_display["p"] = between_display["p"].map(
                lambda p: f"{p:.4g}" if np.isfinite(p) else "n/a"
            )
            st.dataframe(between_display, use_container_width=True)
        st.download_button(
            "Download stats CSV",
            data=pd.concat(
                [
                    within.assign(kind="within day vs night"),
                    between.assign(kind="between groups"),
                ],
                ignore_index=True,
            ).to_csv(index=False),
            file_name="phase_totals_stats.csv",
            mime="text/csv",
            key="dl_phase_stats",
        )

    with sec_e:
        fig_header(
            "Compare / mix two graphs",
            fig_id="compare_mix",
            base_style=style,
            customize=False,
            what="Pick any two figures to view side-by-side or stacked, then download as one PNG.",
            how="""
- Uses the **Plot style** controls in the sidebar.
- Best for related plots (e.g. LD folded vs DD folded, or mean bars vs boxplots).
- Individual actograms (Figs 1–2) are omitted here because they are multi-page grids.
""",
            ask="Which two views tell the cleanest LD vs DD story for this experiment?",
        )
        if not means_death:
            means_death = {
                g: group_mean_trace(counts, groups[g], death_bins, start_zt, 0.0)
                for g in groups
            }

        choice_ids = list(COMPARE_LABELS.keys())
        labels = [COMPARE_LABELS[k] for k in choice_ids]
        c1, c2, c3 = st.columns([2, 2, 1.2])
        with c1:
            left_label = st.selectbox(
                "Left / top graph",
                labels,
                index=labels.index(COMPARE_LABELS["fig5"]),
                key="compare_left",
            )
        with c2:
            right_label = st.selectbox(
                "Right / bottom graph",
                labels,
                index=labels.index(COMPARE_LABELS["fig8"]),
                key="compare_right",
            )
        with c3:
            layout = st.radio(
                "Layout",
                ["side-by-side", "stacked"],
                horizontal=False,
                key="compare_layout",
            )

        left_id = choice_ids[labels.index(left_label)]
        right_id = choice_ids[labels.index(right_label)]

        try:
            fig_left = build_selected_figure(
                left_id,
                counts=counts,
                groups=groups,
                group_colors=group_colors,
                start_zt=start_zt,
                period_i=period_i,
                ld_end_i=ld_end_i,
                death_bins=death_bins,
                means_death=means_death,
                totals=totals,
                style=style,
            )
            fig_right = build_selected_figure(
                right_id,
                counts=counts,
                groups=groups,
                group_colors=group_colors,
                start_zt=start_zt,
                period_i=period_i,
                ld_end_i=ld_end_i,
                death_bins=death_bins,
                means_death=means_death,
                totals=totals,
                style=style,
            )
        except Exception as exc:
            st.error(f"Could not build compare view: {exc}")
        else:
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(f"**{COMPARE_LABELS[left_id]}**")
                st.pyplot(fig_left)
            with col_r:
                st.markdown(f"**{COMPARE_LABELS[right_id]}**")
                st.pyplot(fig_right)

            combined = stitch_figures(fig_left, fig_right, layout=layout)
            st.download_button(
                "Download combined PNG",
                data=combined,
                file_name=f"compare_{left_id}_{right_id}.png",
                mime="image/png",
                key="compare_download",
            )
            plt.close(fig_left)
            plt.close(fig_right)


def render_activity_graphs() -> None:
    """Backward-compatible wrapper."""
    settings = render_graphs_sidebar()
    render_activity_graphs_body(settings)
