"""
LatentSelf — Dash Frontend with Click-to-Dialogue

Three-panel layout:
  Left sidebar: Controls (time range, cluster focus, animation mode)
  Center: 3D constellation + timeline + cluster analysis tabs
  Right panel: Dialogue viewer (click a point → see full conversation)
"""

import json
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback, no_update, ALL, ctx

ROOT = Path(__file__).resolve().parent.parent.parent
SESSION_PATH = ROOT / "data" / "processed" / "sessionized_data.json"
INTERP_PATH = ROOT / "data" / "processed" / "interpretations.json"

# ── Colors ──
BG = "#0a0e27"
SIDEBAR_BG = "#0d1230"
PANEL_BG = "#111640"
CARD_BG = "#141938"
BORDER = "#2a3050"
TEXT = "#c8cdd3"
TEXT_DIM = "#8890a4"
ACCENT = "#00d9ff"
NOISE_COLOR = "#555555"

PALETTE = [
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
# Data
# ═══════════════════════════════════════════

def load_all():
    with open(SESSION_PATH, "r", encoding="utf-8") as f:
        sessions = json.load(f)

    rows, dialogues_map = [], {}
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
            "cluster_interpretation": s["engine_data"].get("cluster_interpretation", ""),
            "x": coords[0], "y": coords[1], "z": coords[2],
            "hover_text": target[:120] + ("..." if len(target) > 120 else ""),
            "full_text": target,
        })
        dialogues_map[sid] = s["ui_data"]["dialogues"]

    df = pd.DataFrame(rows)
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["date"] = df["start_time"].dt.date

    interp = None
    if INTERP_PATH.exists():
        with open(INTERP_PATH, "r", encoding="utf-8") as f:
            interp = json.load(f)

    return df, dialogues_map, interp


DF, DIALOGUES, INTERP = load_all()
LABELS = INTERP.get("cluster_labels", {}) if INTERP else {}


# ═══════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════

def build_3d(df, highlight_cluster=None):
    fig = go.Figure()

    # Noise
    noise = df[df["cluster_id"] == -1]
    if len(noise) > 0:
        fig.add_trace(go.Scatter3d(
            x=noise["x"], y=noise["y"], z=noise["z"], mode="markers",
            marker=dict(size=3, color=NOISE_COLOR, opacity=0.25),
            text=noise["hover_text"],
            customdata=noise["session_id"].values,
            hovertemplate="<b>%{text}</b><extra>noise</extra>",
            name="noise", showlegend=False,
        ))

    # Clusters
    clustered = df[df["cluster_id"] != -1]
    for cid in sorted(clustered["cluster_id"].unique()):
        sub = clustered[clustered["cluster_id"] == cid]
        color = PALETTE[cid % len(PALETTE)]
        cname = sub["cluster_name"].iloc[0]
        hl = highlight_cluster is not None and cid == highlight_cluster
        dim = highlight_cluster is not None and cid != highlight_cluster

        fig.add_trace(go.Scatter3d(
            x=sub["x"], y=sub["y"], z=sub["z"], mode="markers",
            marker=dict(
                size=7 if hl else (3 if dim else 5),
                color=color,
                opacity=0.95 if hl else (0.12 if dim else 0.85),
                line=dict(width=0.5 if hl else 0, color="rgba(255,255,255,0.3)"),
            ),
            text=sub["hover_text"],
            customdata=sub["session_id"].values,
            hovertemplate=f"<b>{cname}</b> (C{cid})<br>" + "%{text}<extra></extra>",
            name=cname, showlegend=False,
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Inter, sans-serif", color=TEXT),
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(showgrid=True, gridcolor="#1a1f3a", showticklabels=False, title="", zeroline=False, showspikes=False, showbackground=False),
            yaxis=dict(showgrid=True, gridcolor="#1a1f3a", showticklabels=False, title="", zeroline=False, showspikes=False, showbackground=False),
            zaxis=dict(showgrid=True, gridcolor="#1a1f3a", showticklabels=False, title="", zeroline=False, showspikes=False, showbackground=False),
            bgcolor=BG,
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
        ),
        hoverlabel=dict(bgcolor="#1a1f3a", bordercolor=BORDER, font=dict(size=12, color=TEXT, family="monospace")),
        height=620,
    )
    return fig


def build_timeline(df):
    df_t = df[df["cluster_id"] != -1].copy()
    if len(df_t) == 0:
        return go.Figure()
    df_t["week"] = df_t["start_time"].dt.to_period("W").apply(lambda r: r.start_time)
    top = df_t["cluster_id"].value_counts().head(8).index.tolist()
    df_t["group"] = df_t["cluster_id"].apply(
        lambda c: LABELS.get(str(c), {}).get("name", f"C{c}") if c in top else "Other"
    )
    pivot = df_t.groupby(["week", "group"]).size().unstack(fill_value=0)

    fig = go.Figure()
    if "Other" in pivot.columns:
        fig.add_trace(go.Scatter(x=pivot.index, y=pivot["Other"], mode="lines",
                                 stackgroup="one", name="Other", line=dict(width=0),
                                 fillcolor="rgba(80,80,80,0.3)"))
    for cid in top:
        name = LABELS.get(str(cid), {}).get("name", f"C{cid}")
        if name not in pivot.columns:
            continue
        color = PALETTE[cid % len(PALETTE)]
        fig.add_trace(go.Scatter(x=pivot.index, y=pivot[name], mode="lines",
                                 stackgroup="one", name=name, line=dict(width=0.5, color=color)))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, size=11), margin=dict(l=0, r=0, t=10, b=0), height=200,
        xaxis=dict(gridcolor="#1a1f3a"), yaxis=dict(gridcolor="#1a1f3a", title="sessions/week"),
        legend=dict(orientation="h", y=1.15, font=dict(size=9)), hovermode="x unified",
    )
    return fig


# ═══════════════════════════════════════════
# Dialogue renderer
# ═══════════════════════════════════════════

def render_dialogue_html(session_id):
    if not session_id or session_id not in DIALOGUES:
        return html.Div("Click a point to view its conversation.",
                        style={"color": TEXT_DIM, "padding": "20px", "textAlign": "center"})

    dialogues = DIALOGUES[session_id]
    row = DF[DF["session_id"] == session_id].iloc[0]
    cid = row["cluster_id"]
    color = PALETTE[cid % len(PALETTE)] if cid >= 0 else NOISE_COLOR
    cname = row["cluster_name"]

    children = [
        # Header
        html.Div([
            html.Span(cname, style={"color": color, "fontWeight": "600", "fontSize": "1rem"}),
            html.Span(f"  {row['start_time'].strftime('%Y-%m-%d %H:%M')}  {row['turn_count']} turns",
                      style={"color": TEXT_DIM, "fontSize": "0.75rem", "marginLeft": "8px"}),
        ], style={"borderLeft": f"4px solid {color}", "padding": "10px 14px",
                  "background": CARD_BG, "borderRadius": "8px", "marginBottom": "12px"}),
    ]

    for d in dialogues:
        # User
        children.append(html.Div([
            html.Div(f"User  Turn {d['turn']}  {d['timestamp']}",
                     style={"color": ACCENT, "fontSize": "0.7rem", "marginBottom": "4px"}),
            html.Div(d["user_prompt"],
                     style={"color": "#e0e4ea", "fontSize": "0.85rem", "whiteSpace": "pre-wrap"}),
        ], style={"background": "#1a1f3a", "borderRadius": "8px", "padding": "10px 14px", "marginBottom": "4px"}))

        # AI
        if d.get("ai_response"):
            resp = d["ai_response"][:2000] + ("..." if len(d["ai_response"]) > 2000 else "")
            children.append(html.Div([
                html.Div("AI", style={"color": TEXT_DIM, "fontSize": "0.7rem", "marginBottom": "4px"}),
                html.Div(resp, style={"color": "#aab0be", "fontSize": "0.8rem", "whiteSpace": "pre-wrap"}),
            ], style={"background": "#0d1230", "borderLeft": f"2px solid {BORDER}",
                      "borderRadius": "8px", "padding": "10px 14px", "marginBottom": "12px"}))

    return html.Div(children)


# ═══════════════════════════════════════════
# Cluster cards
# ═══════════════════════════════════════════

def render_all_clusters_html(selected_cid=None):
    children = []
    for cid_str in sorted(LABELS.keys(), key=int):
        cid = int(cid_str)
        color = PALETTE[cid % len(PALETTE)]
        label = LABELS[cid_str]
        size = INTERP.get("cluster_profiles", {}).get(cid_str, {}).get("size", 0)
        is_selected = selected_cid == cid
        children.append(html.Div([
            html.Span(label.get("name", ""), style={"color": color, "fontWeight": "600"}),
            html.Span(f"  C{cid}  {size}s", style={"color": "#666", "fontSize": "0.8rem"}),
            html.Br(),
            html.Span(label.get("interpretation", ""), style={"color": "#999", "fontSize": "0.8rem"}),
        ], id={"type": "cluster-card", "index": cid}, n_clicks=0,
           style={"background": CARD_BG, "border": f"2px solid {color}" if is_selected else f"1px solid {BORDER}",
                  "borderLeft": f"4px solid {color}", "borderRadius": "8px",
                  "padding": "10px 14px", "marginBottom": "6px", "cursor": "pointer",
                  "opacity": "1" if is_selected or selected_cid is None else "0.5"}))
    return children


# ═══════════════════════════════════════════
# App Layout
# ═══════════════════════════════════════════

app = Dash(__name__)
app.title = "LatentSelf"

# Kill default white border/margin
app.index_string = '''<!DOCTYPE html>
<html>
<head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>html,body{margin:0;padding:0;background:''' + BG + ''';overflow:hidden;}</style>
</head>
<body>{%app_entry%}{%config%}{%scripts%}{%renderer%}</body>
</html>'''

min_date = DF["date"].min()
max_date = DF["date"].max()
date_range_days = (max_date - min_date).days


app.layout = html.Div(style={"backgroundColor": BG, "color": TEXT, "fontFamily": "Inter, sans-serif",
                              "display": "flex", "height": "100vh", "overflow": "hidden"}, children=[

    # ── Left Sidebar ──
    html.Div(style={"width": "260px", "minWidth": "260px", "backgroundColor": SIDEBAR_BG,
                     "padding": "16px", "overflowY": "auto", "borderRight": f"1px solid {BORDER}"}, children=[
        html.H2("LatentSelf", style={"color": ACCENT, "margin": "0 0 2px 0", "fontSize": "1.3rem"}),
        html.P("Latent Space of Your Mind", style={"color": TEXT_DIM, "fontSize": "0.75rem", "margin": "0 0 16px 0"}),
        html.Hr(style={"borderColor": BORDER}),

        html.Label("Time Range", style={"fontSize": "0.8rem", "color": TEXT_DIM}),
        dcc.RangeSlider(
            id="time-slider",
            min=0, max=date_range_days, step=1,
            value=[0, date_range_days],
            marks={0: str(min_date), date_range_days: str(max_date)},
            tooltip={"placement": "bottom"},
        ),

        html.Div(style={"height": "12px"}),

        dcc.Checklist(
            id="show-noise",
            options=[{"label": " Show noise", "value": "yes"}],
            value=["yes"],
            style={"fontSize": "0.85rem"},
        ),

        html.Hr(style={"borderColor": BORDER, "marginTop": "12px"}),

        # Hidden store for selected cluster
        dcc.Store(id="cluster-focus", data=-1),

        html.P("Cluster Details", style={"color": ACCENT, "fontWeight": "600",
                                          "fontSize": "0.85rem", "margin": "0 0 8px 0"}),
        html.Div(id="cluster-cards", style={"fontSize": "0.8rem"}),

        html.Hr(style={"borderColor": BORDER, "marginTop": "12px"}),
        html.P("Qwen3-Embedding-8B + UMAP + HDBSCAN",
               style={"color": "#444", "fontSize": "0.65rem"}),
    ]),

    # ── Center: 3D + Timeline ──
    html.Div(style={"flex": "1", "overflowY": "auto", "padding": "12px 16px"}, children=[

        # Stats
        html.Div(id="stats-bar", style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),

        # 3D graph
        dcc.Graph(id="scatter-3d", config={"displayModeBar": True, "displaylogo": False,
                                            "modeBarButtonsToRemove": ["toImage"]},
                  style={"height": "620px"}),

        # Timeline
        dcc.Graph(id="timeline", config={"displayModeBar": False}, style={"height": "200px"}),
    ]),

    # ── Right Panel: Dialogue ──
    html.Div(style={"width": "380px", "minWidth": "380px", "backgroundColor": SIDEBAR_BG,
                     "borderLeft": f"1px solid {BORDER}", "overflowY": "auto",
                     "padding": "16px"}, children=[
        html.P("Dialogue Viewer", style={"color": ACCENT, "fontWeight": "600",
                                          "fontSize": "0.95rem", "margin": "0 0 4px 0"}),
        html.P("Click a point in the 3D plot", style={"color": TEXT_DIM, "fontSize": "0.75rem",
                                                       "margin": "0 0 12px 0"}),
        html.Div(id="dialogue-panel"),
    ]),
])


# ═══════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════

@callback(
    Output("cluster-focus", "data"),
    Input({"type": "cluster-card", "index": ALL}, "n_clicks"),
    State("cluster-focus", "data"),
    prevent_initial_call=True,
)
def on_cluster_card_click(n_clicks_list, current_focus):
    if not ctx.triggered_id or not any(n_clicks_list):
        return no_update
    clicked_cid = ctx.triggered_id["index"]
    # Toggle: click same cluster again → deselect
    if clicked_cid == current_focus:
        return -1
    return clicked_cid


@callback(
    Output("scatter-3d", "figure"),
    Output("timeline", "figure"),
    Output("stats-bar", "children"),
    Output("cluster-cards", "children"),
    Input("time-slider", "value"),
    Input("show-noise", "value"),
    Input("cluster-focus", "data"),
)
def update_main(time_range, show_noise, focus_cluster):
    d0 = min_date + timedelta(days=time_range[0])
    d1 = min_date + timedelta(days=time_range[1])
    mask = (DF["date"] >= d0) & (DF["date"] <= d1)
    if "yes" not in (show_noise or []):
        mask = mask & (DF["cluster_id"] != -1)
    filtered = DF[mask]

    hl = focus_cluster if focus_cluster != -1 else None
    fig3d = build_3d(filtered, highlight_cluster=hl)
    timeline = build_timeline(filtered)

    n_sess = len(filtered)
    n_clust = filtered[filtered["cluster_id"] != -1]["cluster_id"].nunique()
    n_noise = int((filtered["cluster_id"] == -1).sum())
    n_turns = int(filtered["turn_count"].sum())

    def stat_card(num, label):
        return html.Div([
            html.Div(str(num), style={"fontSize": "1.8rem", "fontWeight": "700", "color": ACCENT}),
            html.Div(label, style={"fontSize": "0.7rem", "color": TEXT_DIM, "textTransform": "uppercase",
                                   "letterSpacing": "0.08em"}),
        ], style={"background": CARD_BG, "border": f"1px solid {BORDER}", "borderRadius": "12px",
                  "padding": "12px 20px", "textAlign": "center", "flex": "1"})

    stats = [stat_card(n_sess, "Sessions"), stat_card(n_clust, "Clusters"),
             stat_card(n_noise, "Noise"), stat_card(n_turns, "Turns")]

    clusters = render_all_clusters_html(selected_cid=hl)
    return fig3d, timeline, stats, clusters

@callback(
    Output("dialogue-panel", "children"),
    Input("scatter-3d", "clickData"),
)
def on_point_click(click_data):
    if not click_data:
        return html.Div("Click a point to view its conversation.",
                        style={"color": TEXT_DIM, "padding": "20px", "textAlign": "center"})
    point = click_data["points"][0]
    session_id = point.get("customdata")
    return render_dialogue_html(session_id)


# ═══════════════════════════════════════════
# Run
# ═══════════════════════════════════════════

def run():
    app.run(host="127.0.0.1", port=8501, debug=False)


if __name__ == "__main__":
    run()
