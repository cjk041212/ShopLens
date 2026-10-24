"""Views of the Studio's observed lists, with no inferred missing ranks."""

from __future__ import annotations

import html
from collections import Counter

import plotly.graph_objects as go

from .visual_charts import COLORS, LABELS, MODES, _rank_maps


def arena_data(results: dict, errors: dict | None = None) -> dict:
    """Keep failed lists distinct from legitimate empty retrieval results."""
    errors = errors or {}
    available = {mode: results[mode] for mode in MODES if mode in results and mode not in errors}
    _, ranks = _rank_maps(available)
    item_ids = list(dict.fromkeys(item for mode in available for item in ranks[mode]))
    memberships = {item: tuple(mode for mode in available if item in ranks[mode]) for item in item_ids}
    complete = len(available) == len(MODES)
    counts = Counter(len(modes) for modes in memberships.values())
    return {"modes": list(available), "ranks": ranks, "item_ids": item_ids,
            "membership": memberships, "complete": complete, "union": len(item_ids),
            "all_three": counts[3] if complete else None,
            "two_models": counts[2] if complete else None,
            "exclusive": {mode: sum(modes == (mode,) for modes in memberships.values())
                          for mode in MODES} if complete else None}


def rank_story_figure(data: dict, catalog: dict, selected: str) -> go.Figure:
    """Show the selected product against faint actual Top-12 trajectories."""
    if selected not in data["item_ids"]:
        raise ValueError("Select an item in the current observed recommendation lists.")
    modes = data["modes"]
    figure = go.Figure()
    # Draw focus last, so its line and nodes stay above the contextual products.
    order = [item for item in data["item_ids"] if item != selected] + [selected]
    for item in order:
        focus = item == selected
        ranks = [data["ranks"][mode].get(item) for mode in modes]
        title = str(catalog.get(item, {}).get("title") or item)
        title = title[:110] + ("…" if len(title) > 110 else "")
        figure.add_trace(go.Scatter(
            x=list(range(len(modes))), y=ranks, mode="lines+markers+text" if focus else "lines+markers",
            text=[f"#{rank}" if rank is not None else "" for rank in ranks] if focus else None,
            textposition="middle center", textfont={"color": "white", "size": 12},
            connectgaps=False, showlegend=False, name=item,
            line={"color": "#263E64" if focus else "rgba(150,170,197,.23)", "width": 4 if focus else 1.3},
            marker={"color": [COLORS[mode] for mode in modes] if focus else "rgba(150,170,197,.26)",
                    "size": 30 if focus else 7, "line": {"color": "white", "width": 2 if focus else .5}},
            customdata=[[html.escape(title), html.escape(item), LABELS[mode]] for mode in modes],
            hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>%{customdata[2]} · rank #%{y}<extra></extra>",
        ))
    maximum = max(rank for ranks in data["ranks"].values() for rank in ranks.values())
    figure.update_layout(
        template="plotly_white", height=400, paper_bgcolor="white", plot_bgcolor="#F7F9FD",
        font={"family": "Avenir Next, sans-serif", "size": 12, "color": "#526984"},
        margin={"l": 45, "r": 30, "t": 25, "b": 45}, hovermode="closest",
    )
    figure.update_xaxes(tickvals=list(range(len(modes))), ticktext=[LABELS[mode] for mode in modes],
                        range=[-.15, max(.15, len(modes) - .85)], showgrid=True,
                        gridcolor="#DDE6F2", zeroline=False, fixedrange=True)
    figure.update_yaxes(title="Observed rank", range=[max(2, maximum) + .5, .4],
                        tickmode="linear", dtick=1, gridcolor="#E9EEF6", zeroline=False, fixedrange=True)
    return figure
