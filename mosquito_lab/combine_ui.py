"""Streamlit page: combine several experiments onto one ZT timeline.

Upload one activity CSV per experiment, set each one's ZT start, LD->DD hour,
optional data gap, mosquito-kind layout, and (optionally) death calls. The page
then aligns everything onto a shared ZT grid — preserving each experiment's true
ZT start and its own LD/DD schedule — regroups mosquitoes by kind, and renders
the same Figures 3-9 + day/night stats as the single-experiment page.

The heavy lifting lives in :mod:`mosquito_lab.combine`; plotting reuses the
helpers in :mod:`mosquito_lab.activity_plots`.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from mosquito_lab import combine as cmb
from mosquito_lab.activity_plots import (
    MOSQUITO_KINDS,
    apply_mosquito_exclusions,
    build_counts,
    death_bin_by_idx,
    death_comparison,
    default_plot_style,
    fig_header,
    folded_bar,
    folded_line,
    full_period_bar,
    load_activity,
    parse_deaths_from_csv,
    parse_index_spec,
    render_exclusion_control,
    render_plot_style_controls,
)

# Sensible default column layout (4 kinds x 6 mosquitoes = 24 wells).
_DEFAULT_LAYOUT = pd.DataFrame(
    {
        "type": MOSQUITO_KINDS,
        "index": ["0-5", "6-11", "12-17", "18-23"],
    }
)


def render_combine_sidebar() -> dict:
    """Sidebar controls for the combine page (binning + plot style)."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Binning")
    st.sidebar.caption("Applied to every uploaded experiment.")
    bin_size = st.sidebar.number_input(
        "Bin size (frames per bin)",
        value=60,
        min_value=1,
        step=1,
        key="cmb_bin_size",
        help="60 frames/bin -> 1-hour bins (assuming 1 frame per minute).",
    )
    period = st.sidebar.number_input(
        "Circadian period (h)",
        value=24,
        min_value=1,
        step=1,
        key="cmb_period",
    )

    st.sidebar.markdown("#### Plot style")
    st.sidebar.caption("Applies to every combined figure.")
    style = render_plot_style_controls()

    return {"bin_size": int(bin_size), "period": int(period), "style": style}


def _configure_experiment(file, bin_size: int) -> cmb.ExperimentSpec | None:
    """Render one experiment's config block and return its spec (or None)."""
    name = getattr(file, "name", "experiment")
    try:
        data = load_activity(file)
    except Exception as exc:  # noqa: BLE001
        st.error(f"`{name}`: could not read CSV — {exc}")
        return None

    counts = build_counts(data, bin_size)
    n = len(counts)
    max_frames = max((len(f) for f in data["frames"]), default=0)

    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    with c1:
        start_zt = st.number_input(
            "ZT of first bin (start_zt)",
            value=9.54,
            step=0.01,
            format="%.2f",
            key=f"cmb_zt_{name}",
            help="ZT hour when this experiment's recording began.",
        )
    with c2:
        ld_end = st.number_input(
            "LD -> DD switch (ZT hour)",
            value=72,
            min_value=0,
            step=1,
            key=f"cmb_ld_{name}",
            help="Absolute ZT hour the lights go off (e.g. 72 for 3 LD days, 96 for 4).",
        )
    with c3:
        has_gap = st.checkbox(
            "Has data gap",
            key=f"cmb_hasgap_{name}",
            help="Tick if there was a camera outage / missing window.",
        )

    pause_hour: int | None = None
    gap = 0.0
    if has_gap:
        g1, g2 = st.columns(2)
        with g1:
            pause_hour = int(
                st.number_input(
                    "Last hour before gap (pause_hour)",
                    value=0,
                    min_value=0,
                    step=1,
                    key=f"cmb_pause_{name}",
                )
            )
        with g2:
            gap = float(
                st.number_input(
                    "Gap length (hours)",
                    value=0.0,
                    min_value=0.0,
                    step=0.1,
                    key=f"cmb_gap_{name}",
                    help="Inserted as empty (NaN) time so ZT stays aligned.",
                )
            )

    st.caption(
        f"**Mosquito-kind layout** — which of this CSV's {n} columns belong to each "
        "kind (`0-5` or `0,2,4`). Add/remove rows as needed."
    )
    edited = st.data_editor(
        _DEFAULT_LAYOUT,
        key=f"cmb_layout_{name}",
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "type": st.column_config.SelectboxColumn(
                "Kind", options=MOSQUITO_KINDS, required=True, width="medium"
            ),
            "index": st.column_config.TextColumn(
                "CSV columns", help="e.g. 0-5 or 0,2,4", width="medium"
            ),
        },
    )

    groups_local: dict[str, list[int]] = {}
    for _, row in edited.iterrows():
        gtype = str(row["type"]).strip()
        if not gtype:
            continue
        idxs = parse_index_spec(str(row["index"]), n)
        if idxs:
            groups_local.setdefault(gtype, []).extend(idxs)
    groups_local = {k: sorted(set(v)) for k, v in groups_local.items()}

    if not groups_local:
        st.warning(f"`{name}`: no valid mosquito-kind rows yet.")

    # Optional per-experiment death calls.
    deaths_local: dict[int, int] = {}
    dfile = st.file_uploader(
        "Death calls CSV (optional)",
        type=["csv"],
        key=f"cmb_death_{name}",
        help="Columns: group, mosquito # (in group), death frame.",
    )
    if dfile is not None and groups_local:
        entries, msgs = parse_deaths_from_csv(dfile, max_frames, groups_local)
        if entries:
            deaths_local = death_bin_by_idx(entries, groups_local, bin_size)
            st.success(f"`{name}`: loaded {len(entries)} death call(s).")
        for m in msgs:
            st.warning(f"`{name}`: {m}")

    st.caption(
        f"Loaded **{n} mosquitoes**, ~{max_frames} frames "
        f"(≈ {max_frames // max(bin_size, 1)} h at bin size {bin_size})."
    )

    return cmb.ExperimentSpec(
        name=name,
        counts=counts,
        start_zt=float(start_zt),
        ld_end=int(ld_end),
        groups=groups_local,
        deaths=deaths_local,
        pause_hour=pause_hour,
        gap=gap,
    )


def _render_combined_figures(cd: cmb.CombinedData, style, period: int) -> None:
    """Figures 3-9 on the combined ZT grid (start_zt anchored at 0)."""
    counts = cd.counts
    counts_ld = cd.counts_ld
    counts_dd = cd.counts_dd
    groups = cd.groups
    gc = cd.group_colors
    death_bins = cd.death_bins

    sec_a, sec_b, sec_c = st.tabs(
        [
            "Section A — General",
            "Section B — LD",
            "Section C — DD",
        ]
    )

    with sec_a:
        s3 = fig_header(
            "Fig 3 — Death comparison",
            fig_id="cmb_fig3",
            base_style=style,
            customize=True,
            what="Combined group-average traces with two cut rules: at death vs 24 h before.",
            how="""
- Traces are averaged across **all** experiments after ZT alignment.
- Each mosquito is cut at its own death call (left) or 24 h earlier (right).
- Hours before an experiment's ZT start stay blank (NaN).
""",
        )
        death_comparison(
            counts, groups, gc, 0.0, period, cd.ld_max, death_bins,
            key="cmb_fig3", style=s3,
        )

    with sec_b:
        st.caption(
            "LD = each mosquito's hours **before** its own LD→DD switch "
            f"(switches range ZT {cd.dd_min}–{cd.ld_max})."
        )
        s4 = fig_header(
            "Fig 4 — Full LD period",
            fig_id="cmb_fig4",
            base_style=style,
            customize=True,
            what="Combined group-mean activity across the LD portion (ZT timeline).",
            how="Each mosquito contributes only its own LD hours; death calls excluded.",
        )
        full_period_bar(
            counts_ld, groups, gc, 0.0, period,
            lo=0, hi=cd.ld_max, ld_end=cd.ld_max,
            title="Full LD period, combined across experiments",
            xlabel="ZT (hour)", key="cmb_fig4", death_bins=death_bins, style=s4,
        )
        st.divider()
        s5 = fig_header(
            "Fig 5 — LD, 24 h-folded",
            fig_id="cmb_fig5",
            base_style=style,
            customize=True,
            what="All LD days folded into one average 24 h (ZT) profile per kind.",
            how="Grey = ZT night half (ZT period/2–period).",
        )
        folded_bar(
            counts_ld, groups, gc, 0.0, period, 0, None, death_bins,
            title="LD, 24 h-folded (combined)", key="cmb_fig5", style=s5,
        )
        st.divider()
        s6 = fig_header(
            "Fig 6 — LD mean ± 1 SD",
            fig_id="cmb_fig6",
            base_style=style,
            customize=True,
            what="24 h-folded LD profile as mean ± 1 SD across all mosquitoes.",
            how="Ribbon width shows mosquito-to-mosquito spread within each kind.",
        )
        folded_line(
            counts_ld, groups, gc, 0.0, period, 0, None, death_bins,
            title="LD (24 h-folded) mean ± 1 SD (combined)", key="cmb_fig6", style=s6,
        )

    with sec_c:
        st.caption(
            "DD = each mosquito's hours **after** its own LD→DD switch. "
            "Grey bands continue as subjective night (same ZT halves)."
        )
        s7 = fig_header(
            "Fig 7 — Full DD period",
            fig_id="cmb_fig7",
            base_style=style,
            customize=True,
            what="Combined group-mean activity after the LD→DD switch (DD-only means).",
            how="Only each mosquito's post-switch hours contribute; death calls excluded.",
        )
        full_period_bar(
            counts_dd, groups, gc, 0.0, period,
            lo=cd.dd_min, hi=None, ld_end=cd.dd_min,
            title="Full DD period, combined across experiments",
            xlabel="ZT (hour)", key="cmb_fig7", death_bins=death_bins, style=s7,
        )
        st.divider()
        s8 = fig_header(
            "Fig 8 — DD, 24 h-folded",
            fig_id="cmb_fig8",
            base_style=style,
            customize=True,
            what="DD days folded into one average 24 h ZT profile per kind.",
            how="Grey = ZT night half.",
        )
        folded_bar(
            counts_dd, groups, gc, 0.0, period, 0, None, death_bins,
            title="DD, 24 h-folded (combined)", key="cmb_fig8", style=s8,
        )
        st.divider()
        s9 = fig_header(
            "Fig 9 — DD mean ± 1 SD",
            fig_id="cmb_fig9",
            base_style=style,
            customize=True,
            what="DD 24 h-folded mean ± 1 SD across all mosquitoes.",
            how="Compare ribbon width to Fig 6 — DD is often noisier.",
        )
        folded_line(
            counts_dd, groups, gc, 0.0, period, 0, None, death_bins,
            title="DD (24 h-folded) mean ± 1 SD (combined)", key="cmb_fig9", style=s9,
        )


def render_combine_body(settings: dict) -> None:
    """Main panel: upload experiments, configure each, then plot combined figures."""
    bin_size = settings["bin_size"]
    period = settings["period"]
    style = settings.get("style") or default_plot_style()

    st.markdown(
        "Upload **one activity CSV per experiment**, configure each below, then "
        "combine them onto a shared ZT timeline."
    )
    with st.expander("How combining works", expanded=False):
        st.markdown(
            """
- **True ZT is preserved:** an experiment recorded from ZT 3.5 starts at ~hour 3.5
  on the shared axis; one from ZT 10 starts at ~hour 10. Hours before each
  experiment's start stay blank (NaN) — nothing is re-anchored to 0.
- **Kinds line up:** mosquitoes are regrouped by kind (Female/Male × WT/KO)
  across experiments, regardless of their column order in each CSV.
- **Per-experiment LD/DD:** each experiment keeps its own LD→DD switch (ZT hour),
  so 3-day and 4-day LD schedules aren't blended.
- **Gaps → NaN:** a data gap (camera outage) is inserted as empty time so
  everything downstream of it stays ZT-aligned.
"""
        )

    files = st.file_uploader(
        "Upload activity CSVs (one per experiment)",
        type=["csv"],
        accept_multiple_files=True,
        key="cmb_uploads",
    )
    if not files:
        st.info("Upload two or more activity CSVs to begin.")
        return

    if style.graph_title.strip():
        st.markdown(f"## {style.graph_title.strip()}")

    specs: list[cmb.ExperimentSpec] = []
    for file in files:
        name = getattr(file, "name", "experiment")
        with st.expander(f"⚙️ Configure — {name}", expanded=len(files) <= 2):
            spec = _configure_experiment(file, bin_size)
        if spec is not None:
            specs.append(spec)

    valid = [s for s in specs if s.groups]
    if not valid:
        st.warning("Define at least one mosquito-kind row in each experiment to combine.")
        return

    try:
        cd = cmb.combine_experiments(valid, period)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not combine experiments: {exc}")
        return

    st.success(
        f"Combined **{cd.n_mosq} mosquitoes** from **{len(valid)} experiments** "
        f"onto a ZT 0–{cd.T} h timeline."
    )

    # Compact per-kind × per-experiment count table.
    exp_names = [s.name for s in valid]
    summary_rows = []
    for kind, per_exp in cd.per_type_counts.items():
        row = {"kind": kind}
        for e in exp_names:
            row[e] = per_exp.get(e, 0)
        row["total"] = sum(per_exp.values())
        summary_rows.append(row)
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    st.markdown("#### Exclude mosquitoes (optional)")
    st.caption("Drop specific wells (e.g. dead-on-arrival or noisy) from every combined graph and stat.")

    def _combined_label(i: int) -> str:
        kind, num = cd.labels.get(i, ("?", 0))
        exp = cd.exp_by_idx.get(i, "?")
        return f"{kind} #{num} · {exp} (idx {i})"

    excluded = render_exclusion_control(
        cd.groups, key="exclude_mosq_combined", label_fn=_combined_label
    )
    cd.groups, cd.group_colors = apply_mosquito_exclusions(
        cd.groups, cd.group_colors, excluded
    )
    if not cd.groups:
        st.warning("Every mosquito is excluded — clear some exclusions to see plots.")
        return

    _render_combined_figures(cd, style, period)
