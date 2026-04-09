"""
LatentSelf — Interactive 3D Visualization with Interpretability Microscope

Features:
  - 3D constellation map (Anthropic interpretability style)
  - Cluster detail cards, comparison, dimension heatmap
  - Time-axis animation: opacity-encoded temporal evolution + timeline chart
  - Session dialogue browser: full conversation viewer
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
SESSION_PATH = ROOT / "data" / "processed" / "sessionized_data.json"
INTERP_PATH = ROOT / "data" / "processed" / "interpretations.json"

# ── 颜色系统 ──
BG_COLOR = "#0a0e27"
PAPER_COLOR = "#0a0e27"
TEXT_COLOR = "#c8cdd3"
ACCENT_COLOR = "#00d9ff"
NOISE_COLOR = "#555555"
GRID_COLOR = "#1a1f3a"
CARD_BG = "linear-gradient(135deg, #111640, #1a1f3a)"
CARD_BORDER = "#2a3050"

CLUSTER_PALETTE = [
    "#6e40aa", "#7b3fb5", "#8b3ec0", "#a33db2", "#b93da4",
    "#cf3e96", "#e04186", "#ee4977", "#f65568", "#fb6a5a",
    "#fd804e", "#fb9747", "#f5af46", "#ebc74b", "#dfdf55",
    "#d0f264", "#b5fc8a", "#72f79e", "#3de8b0", "#1ad5c0",
    "#22bdc8", "#35a4cc", "#448bce", "#5072cc", "#5c58c5",
    "#6846ba", "#7040b0", "#a855f7", "#c084fc", "#e879f9",
    "#f0abfc", "#fbbf24", "#f59e0b", "#ef4444", "#f97316",
    "#84cc16", "#22c55e", "#06b6d4", "#3b82f6", "#8b5cf6",
    "#ec4899", "#14b8a6",
]


# ═══════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════

@st.cache_data
def load_data():
    with open(SESSION_PATH, "r", encoding="utf-8") as f:
        sessions = json.load(f)

    rows = []
    dialogues_map = {}
    for s in sessions:
        coords = s["engine_data"]["umap_coords"]
        target = s["engine_data"]["embedding_target"]
        sid = s["session_id"]

        rows.append({
            "session_id": sid,
            "start_time": s["metadata"]["start_time"],
            "turn_count": s["metadata"]["turn_count"],
            "cluster_id": s["engine_data"]["cluster_id"],
            "cluster_name": s["engine_data"].get("cluster_name", ""),
            "x": coords[0],
            "y": coords[1],
            "z": coords[2],
            "hover_text": target[:120] + ("..." if len(target) > 120 else ""),
            "full_text": target,
        })
        dialogues_map[sid] = s["ui_data"]["dialogues"]

    df = pd.DataFrame(rows)
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["date"] = df["start_time"].dt.date
    return df, dialogues_map


@st.cache_data
def load_interpretations():
    if not INTERP_PATH.exists():
        return None
    with open(INTERP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════
# 3D Figure
# ═══════════════════════════════════════════

def build_figure(df: pd.DataFrame, highlight_cluster=None, time_anchor=None) -> go.Figure:
    """Build 3D scatter. If time_anchor is set, encode opacity by recency."""
    fig = go.Figure()

    # 时间 opacity 编码
    if time_anchor is not None and len(df) > 0:
        days_diff = (time_anchor - df["start_time"].dt.date).apply(lambda d: d.days)
        max_days = max(days_diff.max(), 1)
        # recency: 0 (oldest) → 1 (newest/at anchor)
        recency = 1.0 - (days_diff / max_days).clip(0, 1)
        # opacity: 0.08 (oldest) → 0.95 (newest)
        opacity_arr = 0.08 + recency * 0.87
    else:
        opacity_arr = None

    # ── 噪声点 ──
    noise = df[df["cluster_id"] == -1]
    if len(noise) > 0:
        n_opacity = opacity_arr[noise.index].values * 0.4 if opacity_arr is not None else 0.25
        fig.add_trace(go.Scatter3d(
            x=noise["x"], y=noise["y"], z=noise["z"],
            mode="markers",
            marker=dict(size=3, color=NOISE_COLOR,
                        opacity=float(np.mean(n_opacity)) if opacity_arr is not None else 0.25,
                        line=dict(width=0)),
            text=noise["hover_text"],
            customdata=np.stack([
                noise["start_time"].dt.strftime("%Y-%m-%d %H:%M"),
                noise["turn_count"].astype(str),
            ], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>Turns: %{customdata[1]}<br><i>%{text}</i><extra>noise</extra>",
            name="noise", showlegend=False,
        ))

    # ── 聚类点 ──
    clustered = df[df["cluster_id"] != -1]
    for cid in sorted(clustered["cluster_id"].unique()):
        subset = clustered[clustered["cluster_id"] == cid]
        color = CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)]
        cname = subset["cluster_name"].iloc[0] if len(subset) > 0 else ""

        is_highlighted = highlight_cluster is not None and cid == highlight_cluster
        is_dimmed = highlight_cluster is not None and cid != highlight_cluster

        if opacity_arr is not None:
            c_opacity = opacity_arr[subset.index].values
            if is_dimmed:
                c_opacity = c_opacity * 0.2
        else:
            c_opacity = 0.95 if is_highlighted else (0.15 if is_dimmed else 0.85)

        # For per-point opacity, use color with alpha
        if isinstance(c_opacity, np.ndarray):
            # Plotly Scatter3d doesn't support per-point opacity directly
            # Workaround: use mean opacity per cluster trace
            mean_op = float(np.mean(c_opacity))
        else:
            mean_op = c_opacity

        fig.add_trace(go.Scatter3d(
            x=subset["x"], y=subset["y"], z=subset["z"],
            mode="markers",
            marker=dict(
                size=7 if is_highlighted else (3 if is_dimmed else 5),
                color=color,
                opacity=mean_op,
                line=dict(width=0.8 if is_highlighted else 0.3,
                          color="rgba(255,255,255,0.3)" if is_highlighted else "rgba(255,255,255,0.1)"),
            ),
            text=subset["hover_text"],
            customdata=np.stack([
                subset["start_time"].dt.strftime("%Y-%m-%d %H:%M"),
                subset["turn_count"].astype(str),
                [cname] * len(subset),
                subset["cluster_id"].astype(str),
            ], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[2]} (Cluster %{customdata[3]})<br>"
                "Turns: %{customdata[1]}<br>"
                "<i>%{text}</i><extra></extra>"
            ),
            name=f"{cname}", showlegend=False,
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PAPER_COLOR, plot_bgcolor=BG_COLOR,
        font=dict(family="Inter, -apple-system, sans-serif", color=TEXT_COLOR),
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, showticklabels=False, title="", zeroline=False, showspikes=False, showbackground=False),
            yaxis=dict(showgrid=True, gridcolor=GRID_COLOR, showticklabels=False, title="", zeroline=False, showspikes=False, showbackground=False),
            zaxis=dict(showgrid=True, gridcolor=GRID_COLOR, showticklabels=False, title="", zeroline=False, showspikes=False, showbackground=False),
            bgcolor=BG_COLOR,
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2), up=dict(x=0, y=0, z=1)),
        ),
        hoverlabel=dict(bgcolor="#1a1f3a", bordercolor="#2a3050",
                        font=dict(size=12, color=TEXT_COLOR, family="monospace")),
        height=700,
    )
    return fig


# ═══════════════════════════════════════════
# Timeline Chart
# ═══════════════════════════════════════════

def render_timeline(df: pd.DataFrame, labels: dict):
    """Stacked area chart: cluster activity over time."""
    if len(df) == 0:
        return

    # 按周分桶
    df_t = df[df["cluster_id"] != -1].copy()
    df_t["week"] = df_t["start_time"].dt.to_period("W").apply(lambda r: r.start_time)

    # 取 top-8 最大的 cluster，其余归入 "Other"
    top_clusters = df_t["cluster_id"].value_counts().head(8).index.tolist()
    df_t["group"] = df_t["cluster_id"].apply(
        lambda c: labels.get(str(c), {}).get("name", f"C{c}") if c in top_clusters else "Other"
    )

    pivot = df_t.groupby(["week", "group"]).size().unstack(fill_value=0)

    fig = go.Figure()
    # "Other" 先画（底层）
    if "Other" in pivot.columns:
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot["Other"],
            mode="lines", stackgroup="one", name="Other",
            line=dict(width=0), fillcolor="rgba(80,80,80,0.3)",
        ))

    # Top clusters
    for cid in top_clusters:
        name = labels.get(str(cid), {}).get("name", f"C{cid}")
        if name not in pivot.columns:
            continue
        color = CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)]
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot[name],
            mode="lines", stackgroup="one", name=name,
            line=dict(width=0.5, color=color),
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, size=11),
        margin=dict(l=0, r=0, t=30, b=0), height=250,
        title=dict(text="Cluster Activity Over Time (weekly)", font=dict(size=13)),
        xaxis=dict(gridcolor=GRID_COLOR, title=""),
        yaxis=dict(gridcolor=GRID_COLOR, title="Sessions / week"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=10)),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════
# Dialogue Browser
# ═══════════════════════════════════════════

def render_dialogue(session_id, dialogues_map, df):
    """Render full multi-turn conversation for a session."""
    dialogues = dialogues_map.get(session_id, [])
    if not dialogues:
        st.warning("No dialogue data found.")
        return

    row = df[df["session_id"] == session_id].iloc[0]
    cname = row["cluster_name"]
    cid = row["cluster_id"]
    color = CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)] if cid >= 0 else NOISE_COLOR

    # Session header
    st.markdown(
        f'<div style="background:{CARD_BG}; border:1px solid {CARD_BORDER}; '
        f'border-left:4px solid {color}; border-radius:10px; padding:14px 18px; margin-bottom:16px;">'
        f'<span style="color:{color}; font-weight:600; font-size:1rem;">{cname}</span>'
        f'<span style="color:#666; font-size:0.8rem;"> &middot; {session_id} &middot; '
        f'{row["start_time"].strftime("%Y-%m-%d %H:%M")} &middot; {row["turn_count"]} turns</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Render each turn
    for d in dialogues:
        turn = d["turn"]
        ts = d["timestamp"]
        prompt = d["user_prompt"]
        response = d["ai_response"]

        # User message
        st.markdown(
            f'<div style="background:#1a1f3a; border-radius:8px; padding:12px 16px; margin-bottom:4px;">'
            f'<div style="color:{ACCENT_COLOR}; font-size:0.75rem; margin-bottom:4px;">'
            f'User &middot; Turn {turn} &middot; {ts}</div>'
            f'<div style="color:#e0e4ea; font-size:0.9rem; white-space:pre-wrap;">{_escape_html(prompt)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # AI response
        if response:
            # Truncate very long responses for display
            display_resp = response[:2000] + ("..." if len(response) > 2000 else "")
            st.markdown(
                f'<div style="background:#0d1230; border-left:2px solid #2a3050; '
                f'border-radius:8px; padding:12px 16px; margin-bottom:12px;">'
                f'<div style="color:#8890a4; font-size:0.75rem; margin-bottom:4px;">AI</div>'
                f'<div style="color:#aab0be; font-size:0.85rem; white-space:pre-wrap;">{_escape_html(display_resp)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _escape_html(text: str) -> str:
    """Basic HTML escaping for display in markdown."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


# ═══════════════════════════════════════════
# Cluster Panels (existing)
# ═══════════════════════════════════════════

def render_cluster_card(cid, interp_data, df, color):
    labels = interp_data.get("cluster_labels", {})
    profiles = interp_data.get("cluster_profiles", {})
    label = labels.get(str(cid), {})
    profile = profiles.get(str(cid), {})

    name = label.get("name", f"Cluster {cid}")
    interpretation = label.get("interpretation", "")
    size = profile.get("size", 0)
    top_dims = profile.get("top_dimensions", [])[:5]

    st.markdown(
        f'<div style="background:{CARD_BG}; border:1px solid {CARD_BORDER}; '
        f'border-left:4px solid {color}; border-radius:10px; padding:16px; margin-bottom:12px;">'
        f'<div style="display:flex; justify-content:space-between; align-items:center;">'
        f'<span style="font-size:1.1rem; font-weight:600; color:{color};">{name}</span>'
        f'<span style="color:#8890a4; font-size:0.8rem;">Cluster {cid} &middot; {size} sessions</span>'
        f'</div>'
        f'<p style="color:{TEXT_COLOR}; font-size:0.85rem; margin:8px 0 0 0;">{interpretation}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if top_dims:
        dims_data = []
        for d in top_dims:
            dims_data.append({
                "Dim": d["dim"],
                "Z-Score": f"{d['z_score']:+.2f}",
                "High Example": d["high_samples"][0]["text"][:80] if d["high_samples"] else "",
                "Low Example": d["low_samples"][0]["text"][:80] if d["low_samples"] else "",
            })
        st.dataframe(pd.DataFrame(dims_data), use_container_width=True, height=200)


def render_comparison(interp_data, cid_a, cid_b):
    comparisons = interp_data.get("comparisons", [])
    labels = interp_data.get("cluster_labels", {})

    comp = None
    for c in comparisons:
        if (c["cluster_a"] == cid_a and c["cluster_b"] == cid_b) or \
           (c["cluster_a"] == cid_b and c["cluster_b"] == cid_a):
            comp = c
            break

    name_a = labels.get(str(cid_a), {}).get("name", f"Cluster {cid_a}")
    name_b = labels.get(str(cid_b), {}).get("name", f"Cluster {cid_b}")
    color_a = CLUSTER_PALETTE[cid_a % len(CLUSTER_PALETTE)]
    color_b = CLUSTER_PALETTE[cid_b % len(CLUSTER_PALETTE)]

    if comp is None:
        st.info(f"No pre-computed comparison for {name_a} vs {name_b}. Only top-3 largest clusters have comparisons.")
        return

    dims = comp["distinguishing_dims"]
    fig = go.Figure(go.Bar(
        x=[d["diff"] for d in dims],
        y=[f"Dim {d['dim']}" for d in dims],
        orientation="h",
        marker=dict(color=[color_a if d["diff"] > 0 else color_b for d in dims]),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, size=11),
        margin=dict(l=0, r=0, t=30, b=0), height=250,
        title=dict(text=f"{name_a} vs {name_b}: Distinguishing Dimensions", font=dict(size=13)),
        xaxis=dict(title="Direction Difference", gridcolor=GRID_COLOR),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{name_a} side:**")
        for s in comp["a_side_samples"][:3]:
            st.markdown(f"<p style='color:#aaa; font-size:0.8rem; margin:4px 0;'>- {s['text'][:120]}</p>",
                        unsafe_allow_html=True)
    with col2:
        st.markdown(f"**{name_b} side:**")
        for s in comp["b_side_samples"][:3]:
            st.markdown(f"<p style='color:#aaa; font-size:0.8rem; margin:4px 0;'>- {s['text'][:120]}</p>",
                        unsafe_allow_html=True)


def render_heatmap(interp_data):
    profiles = interp_data.get("cluster_profiles", {})
    labels = interp_data.get("cluster_labels", {})

    all_dims = set()
    for prof in profiles.values():
        for d in prof.get("top_dimensions", [])[:3]:
            all_dims.add(d["dim"])
    all_dims = sorted(all_dims)[:25]

    cluster_ids = sorted(int(k) for k in profiles.keys())
    z_matrix, y_labels = [], []
    for cid in cluster_ids:
        prof = profiles[str(cid)]
        name = labels.get(str(cid), {}).get("name", f"C{cid}")
        y_labels.append(f"{name} ({cid})")
        dim_map = {d["dim"]: d["z_score"] for d in prof.get("top_dimensions", [])}
        z_matrix.append([dim_map.get(dim, 0.0) for dim in all_dims])

    fig = go.Figure(go.Heatmap(
        z=z_matrix, x=[f"D{d}" for d in all_dims], y=y_labels,
        colorscale="RdBu_r", zmid=0,
        colorbar=dict(title=dict(text="Z-Score", font=dict(color=TEXT_COLOR)), tickfont=dict(color=TEXT_COLOR)),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, size=10),
        margin=dict(l=0, r=0, t=30, b=0), height=max(400, len(cluster_ids) * 18),
        title=dict(text="Cluster x Dimension Activation (Z-Score)", font=dict(size=13)),
        xaxis=dict(tickangle=45),
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def main():
    st.set_page_config(page_title="LatentSelf", layout="wide", initial_sidebar_state="expanded")

    st.markdown(f"""
    <style>
        .stApp {{ background-color: {BG_COLOR}; }}
        .stSidebar {{ background-color: #0d1230; }}
        header[data-testid="stHeader"] {{ background-color: {BG_COLOR}; }}
        .block-container {{ padding-top: 1.5rem; }}
        h1, h2, h3 {{ color: #e0e4ea !important; }}
        .stat-card {{
            background: {CARD_BG};
            border: 1px solid {CARD_BORDER};
            border-radius: 12px;
            padding: 16px 20px;
            text-align: center;
        }}
        .stat-number {{ font-size: 2rem; font-weight: 700; color: {ACCENT_COLOR}; line-height: 1.2; }}
        .stat-label {{ font-size: 0.75rem; color: #8890a4; text-transform: uppercase; letter-spacing: 0.08em; }}
    </style>
    """, unsafe_allow_html=True)

    df, dialogues_map = load_data()
    interp_data = load_interpretations()
    labels = interp_data.get("cluster_labels", {}) if interp_data else {}

    # ── Sidebar ──
    with st.sidebar:
        st.markdown(
            f"<h2 style='color:{ACCENT_COLOR}; margin-bottom:0;'>LatentSelf</h2>"
            "<p style='color:#8890a4; font-size:0.85rem; margin-top:0;'>Latent Space of Your Mind</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        min_date, max_date = df["date"].min(), df["date"].max()

        # Animation mode
        animation_mode = st.toggle("Animation Mode", value=False)

        if animation_mode:
            time_mode = st.radio("Mode", ["Cumulative", "Window"], horizontal=True)
            anchor_date = st.slider("Time Cursor", min_value=min_date, max_value=max_date,
                                    value=max_date, format="YYYY-MM-DD")
            if time_mode == "Window":
                window_days = st.slider("Window (days)", 7, 90, 30)
        else:
            date_range = st.slider("Time Range", min_value=min_date, max_value=max_date,
                                   value=(min_date, max_date), format="YYYY-MM-DD")

        show_noise = st.checkbox("Show noise points", value=True)

        st.markdown("---")

        cluster_options = {-1: "All Clusters"}
        for cid in sorted(df[df["cluster_id"] != -1]["cluster_id"].unique()):
            name = labels.get(str(cid), {}).get("name", f"Cluster {cid}")
            cluster_options[cid] = f"{cid}: {name}"
        selected_cluster = st.selectbox("Focus Cluster", options=list(cluster_options.keys()),
                                        format_func=lambda x: cluster_options[x])

        st.markdown("---")
        st.markdown("<p style='color:#555; font-size:0.7rem;'>Qwen3-Embedding-8B + UMAP + HDBSCAN</p>",
                    unsafe_allow_html=True)

    # ── 过滤 ──
    time_anchor = None
    if animation_mode:
        if time_mode == "Cumulative":
            mask = df["date"] <= anchor_date
            time_anchor = anchor_date
        else:
            from datetime import timedelta
            window_start = anchor_date - timedelta(days=window_days)
            mask = (df["date"] >= window_start) & (df["date"] <= anchor_date)
            time_anchor = anchor_date
    else:
        mask = (df["date"] >= date_range[0]) & (df["date"] <= date_range[1])

    if not show_noise:
        mask = mask & (df["cluster_id"] != -1)
    filtered = df[mask]

    # ── 统计卡片 ──
    n_sessions = len(filtered)
    n_clusters = filtered[filtered["cluster_id"] != -1]["cluster_id"].nunique()
    n_noise = int((filtered["cluster_id"] == -1).sum())
    n_turns = int(filtered["turn_count"].sum())

    cols = st.columns(4)
    for col, (num, lbl) in zip(cols, [
        (str(n_sessions), "Sessions"), (str(n_clusters), "Clusters"),
        (str(n_noise), "Noise"), (str(n_turns), "Turns"),
    ]):
        col.markdown(f'<div class="stat-card"><div class="stat-number">{num}</div>'
                     f'<div class="stat-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── 3D 图 ──
    highlight = selected_cluster if selected_cluster != -1 else None
    fig = build_figure(filtered, highlight_cluster=highlight, time_anchor=time_anchor)
    st.plotly_chart(fig, use_container_width=True, config={
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["toImage", "resetCameraLastSave3d"],
        "displaylogo": False,
    })

    # ── Timeline chart (always show, useful context) ──
    render_timeline(df, labels)

    # ── Tab 面板 ──
    if interp_data:
        tab1, tab2, tab3, tab4 = st.tabs(["Cluster Details", "Comparison", "Heatmap", "Dialogue Browser"])

        with tab1:
            if selected_cluster != -1 and selected_cluster is not None:
                color = CLUSTER_PALETTE[selected_cluster % len(CLUSTER_PALETTE)]
                render_cluster_card(selected_cluster, interp_data, filtered, color)
            else:
                for cid in sorted(labels.keys(), key=int):
                    cid_int = int(cid)
                    color = CLUSTER_PALETTE[cid_int % len(CLUSTER_PALETTE)]
                    name = labels[cid].get("name", "")
                    interpretation = labels[cid].get("interpretation", "")
                    size = interp_data.get("cluster_profiles", {}).get(cid, {}).get("size", 0)
                    st.markdown(
                        f'<div style="background:{CARD_BG}; border:1px solid {CARD_BORDER}; '
                        f'border-left:4px solid {color}; border-radius:8px; padding:12px 16px; margin-bottom:8px;">'
                        f'<span style="color:{color}; font-weight:600;">{name}</span>'
                        f'<span style="color:#666; font-size:0.8rem;"> &middot; C{cid} &middot; {size}s</span>'
                        f'<br><span style="color:#999; font-size:0.8rem;">{interpretation}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        with tab2:
            valid_clusters = sorted(int(k) for k in labels.keys())
            col1, col2 = st.columns(2)
            with col1:
                cid_a = st.selectbox("Cluster A", valid_clusters,
                                     format_func=lambda x: f"{x}: {labels.get(str(x), {}).get('name', '')}")
            with col2:
                cid_b = st.selectbox("Cluster B", [c for c in valid_clusters if c != cid_a],
                                     format_func=lambda x: f"{x}: {labels.get(str(x), {}).get('name', '')}")
            render_comparison(interp_data, cid_a, cid_b)

        with tab3:
            render_heatmap(interp_data)

        with tab4:
            # ── Dialogue Browser ──
            st.markdown(f"<p style='color:#8890a4; font-size:0.85rem;'>Select a session to view the full conversation.</p>",
                        unsafe_allow_html=True)

            # Session selector
            session_list = filtered.sort_values("start_time", ascending=False)
            if len(session_list) > 0:
                options = session_list["session_id"].tolist()
                display_labels = {
                    sid: f"{row['start_time'].strftime('%Y-%m-%d %H:%M')} | {row['cluster_name']} | {row['hover_text'][:60]}"
                    for sid, row in session_list.set_index("session_id").iterrows()
                }
                selected_session = st.selectbox(
                    "Session",
                    options=options,
                    format_func=lambda x: display_labels.get(x, x),
                )
                render_dialogue(selected_session, dialogues_map, df)
            else:
                st.info("No sessions in current filter range.")


if __name__ == "__main__":
    main()
