"""Pure Plotly views of recorded ShopLens ranks, vectors and test metrics.

The caller supplies actual recommendations or evaluation records. These helpers
never infer missing ranks, compare raw scores across models, or invent samples.
"""

from __future__ import annotations

import html
import json
import math
from collections import defaultdict
from statistics import mean

import plotly.graph_objects as go


MODES = ("qwen_frozen", "qwen_semantic", "qwen_preference")
LABELS = dict(zip(MODES, ("Frozen", "Semantic", "Preference")))
COLORS = dict(zip(MODES, ("#546982", "#347CDB", "#1A967B")))
_DIVERGING = [[0, "#C96859"], [.5, "#F7F9FC"], [1, "#347CDB"]]


def _theme(figure: go.Figure, title: str, *, height: int = 430) -> go.Figure:
    figure.update_layout(
        title={"text": title, "font": {"size": 18}, "x": .01},
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
        font={"family": "Avenir Next, PingFang SC, Microsoft YaHei, sans-serif", "color": "#263750", "size": 12},
        margin={"l": 55, "r": 24, "t": 66, "b": 75}, height=height,
        legend={"orientation": "h", "y": 1.03, "x": 0},
        hoverlabel={"bgcolor": "white", "font_size": 12},
    )
    figure.update_xaxes(gridcolor="#EDF1F6", zeroline=False, linecolor="#DDE5EF")
    figure.update_yaxes(gridcolor="#EDF1F6", zeroline=False, linecolor="#DDE5EF")
    return figure


def _empty(title: str, message: str) -> go.Figure:
    figure = _theme(go.Figure(), title)
    figure.add_annotation(text=message, x=.5, y=.5, xref="paper", yref="paper", showarrow=False, font={"color": "#718198"})
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


def _modes(values) -> list[str]:
    return [mode for mode in MODES if mode in values] + [mode for mode in values if mode not in MODES]


def _label(mode, labels) -> str:
    return str((labels or {}).get(mode, LABELS.get(mode, mode)))


def _color(mode, colors) -> str:
    return str((colors or {}).get(mode, COLORS.get(mode, "#7B8CA4")))


def _name(row: dict) -> str:
    return str(row.get("title") or row.get("item_id") or "Untitled product")


def _short(value: str, limit: int = 26) -> str:
    return value if len(value) <= limit else value[:limit - 1] + "…"


def _rank_maps(results: dict) -> tuple[list[str], dict]:
    modes = _modes(results)
    maps = {}
    for mode in modes:
        maps[mode] = {}
        ranks = set()
        for row in results[mode]:
            item_id, rank = str(row["item_id"]), row["rank"]
            if isinstance(rank, bool) or not isinstance(rank, (int, float)) or not math.isfinite(rank) or int(rank) != rank or rank < 1:
                raise ValueError("Ranks must be recorded positive integers.")
            if item_id in maps[mode] or rank in ranks:
                raise ValueError("Recommendation items and ranks must not repeat within one list.")
            maps[mode][item_id] = int(rank)
            ranks.add(rank)
    return modes, maps


def rank_flow_figure(results: dict, catalog: dict, labels: dict | None = None, colors: dict | None = None, *, title: str = "Top-K rank flow") -> go.Figure:
    """Draw only observed ranks. An absent item creates a genuine line break."""
    modes, maps = _rank_maps(results)
    item_ids = list(dict.fromkeys(item for mode in modes for item in maps[mode]))
    if not item_ids:
        return _empty(title, "No recorded Top-K recommendations for the current preferences.")
    figure = go.Figure()
    for item_id in item_ids:
        row = catalog.get(item_id, {})
        ranks = [maps[mode].get(item_id) for mode in modes]
        customdata = [[html.escape(_name({"item_id": item_id, **row})), item_id, _label(mode, labels)] for mode in modes]
        figure.add_trace(go.Scatter(
            x=list(range(len(modes))), y=ranks, mode="lines+markers", connectgaps=False,
            line={"color": "#CBD5E3", "width": 1.6},
            marker={"color": [_color(mode, colors) for mode in modes], "size": 9, "line": {"width": 1, "color": "white"}},
            customdata=customdata, name=_short(_name({"item_id": item_id, **row})), showlegend=False,
            hovertemplate="%{customdata[0]}<br>ASIN: %{customdata[1]}<br>%{customdata[2]} · Recorded rank #%{y}<extra></extra>",
        ))
    max_rank = max(rank for ranks in maps.values() for rank in ranks.values())
    _theme(figure, title, height=max(400, min(650, 260 + 20 * max_rank)))
    figure.update_xaxes(tickmode="array", tickvals=list(range(len(modes))), ticktext=[_label(mode, labels) for mode in modes], range=[-.15, max(.15, len(modes) - .85)], title="Same preferences · Different training objectives", showgrid=False)
    figure.update_yaxes(title="Recorded rank (1 = best)", autorange="reversed", tickmode="linear", dtick=max(1, math.ceil(max_rank / 12)))
    figure.add_annotation(text="A gap means absent from that model's Top-K; ranks outside Top-K are unknown.", x=0, y=-.21, xref="paper", yref="paper", showarrow=False, xanchor="left", font={"color": "#718198", "size": 11})
    return figure


def overlap_figure(results: dict, labels: dict | None = None, colors: dict | None = None, *, title: str = "Top-K overlap · Jaccard") -> go.Figure:
    """Jaccard of actual Top-K sets; two empty sets remain undefined."""
    del colors  # Set similarity is a scalar, not a model-specific raw score.
    modes, maps = _rank_maps(results)
    if not modes:
        return _empty(title, "No recorded recommendation sets to compare.")
    z, texts, customdata = [], [], []
    for left in modes:
        zrow, trow, crow = [], [], []
        for right in modes:
            a, b = set(maps[left]), set(maps[right])
            union, intersection = len(a | b), len(a & b)
            value = intersection / union if union else None
            zrow.append(value)
            trow.append(f"{value:.0%}" if value is not None else "No items")
            crow.append([intersection, union, len(a), len(b)])
        z.append(zrow)
        texts.append(trow)
        customdata.append(crow)
    names = [_label(mode, labels) for mode in modes]
    figure = go.Figure(go.Heatmap(
        z=z, x=names, y=names, zmin=0, zmax=1,
        colorscale=[[0, "#EFF4FB"], [1, "#347CDB"]], text=texts, texttemplate="%{text}",
        customdata=customdata, xgap=5, ygap=5,
        colorbar={"title": {"text": "Intersection / union"}, "tickformat": ".0%"},
        hoverongaps=False,
        hovertemplate="%{y} × %{x}<br>Jaccard: %{z:.1%}<br>Shared items: %{customdata[0]}<br>Union size: %{customdata[1]}<br>Set sizes: %{customdata[2]} / %{customdata[3]}<extra></extra>",
    ))
    _theme(figure, title, height=360)
    figure.update_xaxes(side="bottom", showgrid=False)
    figure.update_yaxes(autorange="reversed", showgrid=False)
    figure.add_annotation(text="Same preferences, observed Top-K sets. Overlap is undefined when both sets are empty.", x=0, y=-.24, xref="paper", yref="paper", showarrow=False, xanchor="left", font={"color": "#718198", "size": 11})
    return figure


def _matrix(historyrows, candidaterows, similarities) -> list[list[float]]:
    matrix = [list(row) for row in similarities]
    if len(matrix) != len(historyrows) or any(len(row) != len(candidaterows) for row in matrix):
        raise ValueError("Similarity matrix must be history × candidate and match the supplied product rows.")
    numeric = [[float(value) for value in row] for row in matrix]
    if any(not math.isfinite(value) or abs(value) > 1.000001 for row in numeric for value in row):
        raise ValueError("Tower cosine values must be finite recorded values in [-1, 1].")
    return numeric


def evidence_heatmap(historyrows, candidaterows, similarities, *, title: str = "Preference–candidate similarity", value_label: str = "Tower cosine similarity") -> go.Figure:
    """History × candidate cosine values, with a zero-centred diverging scale."""
    matrix = _matrix(historyrows, candidaterows, similarities)
    if not historyrows or not candidaterows:
        return _empty(title, "Select preferences and generate candidates to inspect their vector similarities.")
    customdata = [[[html.escape(_name(h)), str(h.get("item_id", "")), html.escape(_name(c)), str(c.get("item_id", ""))] for c in candidaterows] for h in historyrows]
    figure = go.Figure(go.Heatmap(
        z=matrix, x=list(range(len(candidaterows))), y=list(range(len(historyrows))),
        zmin=-1, zmax=1, zmid=0, colorscale=_DIVERGING, customdata=customdata,
        colorbar={"title": {"text": value_label}, "tickformat": ".1f"}, xgap=3, ygap=3,
        hovertemplate="Preference: %{customdata[0]}<br>ASIN: %{customdata[1]}<br>Candidate: %{customdata[2]}<br>ASIN: %{customdata[3]}<br>Cosine: %{z:+.4f}<extra></extra>",
    ))
    _theme(figure, title, height=max(380, min(680, 245 + 30 * len(historyrows))))
    figure.update_xaxes(tickmode="array", tickvals=list(range(len(candidaterows))), ticktext=[_short(_name(row), 21) for row in candidaterows], tickangle=-28, title="Recommended candidates", showgrid=False)
    figure.update_yaxes(tickmode="array", tickvals=list(range(len(historyrows))), ticktext=[_short(_name(row), 24) for row in historyrows], autorange="reversed", title="Selected preferences", showgrid=False)
    figure.update_layout(margin={"l": 175, "b": 135})
    return figure


def network_figure(historyrows, candidaterows, similarities, *, title: str = "Preference–candidate network", min_abs: float = 0.0, max_edges: int = 80) -> go.Figure:
    """Deterministic bipartite view of measured cosine edges; no causal arrows."""
    matrix = _matrix(historyrows, candidaterows, similarities)
    if not math.isfinite(min_abs) or not 0 <= min_abs <= 1 or max_edges < 1:
        raise ValueError("Edge filtering requires 0 ≤ min_abs ≤ 1 and max_edges ≥ 1.")
    if not historyrows or not candidaterows:
        return _empty(title, "Select preferences and generate candidates to explore their network.")
    hy = [.5] if len(historyrows) == 1 else [1 - i / (len(historyrows) - 1) for i in range(len(historyrows))]
    cy = [.5] if len(candidaterows) == 1 else [1 - i / (len(candidaterows) - 1) for i in range(len(candidaterows))]
    edges = [(i, j, value) for i, row in enumerate(matrix) for j, value in enumerate(row) if value != 0 and abs(value) >= min_abs]
    edges.sort(key=lambda edge: (-abs(edge[2]), edge[0], edge[1]))
    chosen = edges[:max_edges]
    figure = go.Figure()
    seen_signs = set()
    for i, j, value in chosen:
        positive = value > 0
        role = "Positive cosine" if positive else "Negative cosine"
        figure.add_trace(go.Scatter(
            x=[0, 1], y=[hy[i], cy[j]], mode="lines", name=role,
            legendgroup=role, showlegend=positive not in seen_signs,
            line={"color": "#7EA9E5" if positive else "#D79587", "width": .7 + 3 * abs(value), "dash": "solid" if positive else "dot"},
            customdata=[[html.escape(_name(historyrows[i])), html.escape(_name(candidaterows[j])), value]] * 2,
            hovertemplate="Preference: %{customdata[0]}<br>Candidate: %{customdata[1]}<br>Tower cosine: %{customdata[2]:+.4f}<extra></extra>",
        ))
        seen_signs.add(positive)
    for rows, ys, x, role, color, symbol in ((historyrows, hy, 0, "Selected preferences", "#1A967B", "diamond"), (candidaterows, cy, 1, "Recommended candidates", "#347CDB", "circle")):
        figure.add_trace(go.Scatter(
            x=[x] * len(rows), y=ys, mode="markers+text", name=role,
            marker={"size": 13, "color": color, "symbol": symbol, "line": {"color": "white", "width": 2}},
            text=[_short(_name(row), 22) for row in rows], textposition="middle left" if x == 0 else "middle right",
            customdata=[[html.escape(_name(row)), str(row.get("item_id", ""))] for row in rows],
            hovertemplate=role + ": %{customdata[0]}<br>ASIN: %{customdata[1]}<extra></extra>",
        ))
    _theme(figure, title, height=max(400, min(720, 180 + 40 * max(len(historyrows), len(candidaterows)))))
    figure.update_xaxes(visible=False, range=[-.65, 1.65])
    figure.update_yaxes(visible=False, range=[-.12, 1.12])
    figure.add_annotation(text=f"|cosine| ≥ {min_abs:g}; showing {len(chosen)}/{len(edges)} nonzero links. Dashed = negative; links are not causal.", x=0, y=-.12, xref="paper", yref="paper", showarrow=False, xanchor="left", font={"color": "#718198", "size": 11})
    return figure


def _metric_records(rows, metric) -> tuple[list[str], dict]:
    grouped = defaultdict(dict)
    signatures = set()
    protocol_keys = ("candidate_set", "history", "averaging", "n_catalog_items", "positive_feedback", "max_history", "history_encoder")
    common_keys = ("domain", "dataset", "items_sha256", "events_sha256", "n_users", "n_targets", "featurehash")
    config_keys = ("model_id", "revision", "encoding_config", "epochs", "max_examples", "lr", "weight_decay")
    for row in rows:
        if row.get("split") != "test" or row.get("run_protocol", {}).get("evaluation_subset"):
            raise ValueError("Seed and paired plots require formal full-catalog test records.")
        model, seed = str(row["model"]), row["seed"]
        if seed is None or isinstance(seed, bool):
            raise ValueError("Every test record must declare its actual seed.")
        if seed in grouped[model]:
            raise ValueError("Duplicate seed within one model; records cannot be counted twice.")
        value = float(row[metric])
        if not math.isfinite(value):
            raise ValueError("Test metrics must be finite recorded values.")
        protocol = row.get("run_protocol", {})
        config = row.get("training_configuration", {})
        signature = {**{key: row.get(key) for key in common_keys}, "protocol": {key: protocol.get(key) for key in protocol_keys}, "budget": {key: config.get(key, row.get(key)) for key in config_keys}}
        signatures.add(json.dumps(signature, sort_keys=True, ensure_ascii=False))
        grouped[model][seed] = value
    if len(signatures) > 1:
        raise ValueError("Cannot combine records across domains, data fingerprints, training budgets or evaluation protocols.")
    return _modes(grouped), grouped


def seed_metric_figure(rows, metric: str, labels: dict | None = None, colors: dict | None = None, *, title: str | None = None) -> go.Figure:
    """Actual seed points and arithmetic means; no jitter, CI or imputed seeds."""
    title = title or f"{metric.upper()} · Seeds and mean"
    modes, grouped = _metric_records(rows, metric)
    if not modes:
        return _empty(title, "No formal test records with a shared domain and evaluation protocol.")
    figure = go.Figure()
    symbols = ("circle", "diamond", "square", "triangle-up", "cross")
    all_seeds = sorted({seed for values in grouped.values() for seed in values}, key=lambda value: (str(type(value)), value))
    seed_symbols = {seed: symbols[i % len(symbols)] for i, seed in enumerate(all_seeds)}
    for mode in modes:
        seeds = sorted(grouped[mode], key=lambda value: (str(type(value)), value))
        figure.add_trace(go.Scatter(
            x=[_label(mode, labels)] * len(seeds), y=[grouped[mode][seed] for seed in seeds], mode="markers", name=_label(mode, labels),
            marker={"size": 11, "color": _color(mode, colors), "opacity": .85, "symbol": [seed_symbols[seed] for seed in seeds], "line": {"width": 1, "color": "white"}},
            customdata=[[seed] for seed in seeds],
            hovertemplate="%{x}<br>Recorded seed: %{customdata[0]}<br>" + metric + ": %{y:.6f}<extra></extra>",
        ))
    figure.add_trace(go.Scatter(
        x=[_label(mode, labels) for mode in modes], y=[mean(grouped[mode].values()) for mode in modes], mode="markers", name="Mean across recorded seeds",
        marker={"symbol": "line-ew", "size": 25, "color": "#17243C", "line": {"width": 3, "color": "#17243C"}},
        customdata=[[len(grouped[mode])] for mode in modes],
        hovertemplate="%{x}<br>Mean: %{y:.6f}<br>Recorded seeds: n=%{customdata[0]}<extra></extra>",
    ))
    _theme(figure, title)
    figure.update_xaxes(title="Shared domain and evaluation protocol", categoryorder="array", categoryarray=[_label(mode, labels) for mode in modes], showgrid=False)
    figure.update_yaxes(title=metric.upper(), rangemode="tozero", tickformat=".4f")
    figure.add_annotation(text="Each point is one recorded seed; black bars mark means. No confidence intervals or imputed seeds.", x=0, y=-.22, xref="paper", yref="paper", showarrow=False, xanchor="left", font={"color": "#718198", "size": 11})
    return figure


def paired_delta_figure(rows, metric: str, labels: dict | None = None, colors: dict | None = None, *, reference: str = "qwen_frozen", title: str | None = None) -> go.Figure:
    """Within-seed test differences only; an absent pair never becomes zero."""
    title = title or f"{metric.upper()} · Paired differences vs {_label(reference, labels)}"
    modes, grouped = _metric_records(rows, metric)
    if reference not in grouped:
        return _empty(title, "Reference test records are missing; paired differences cannot be computed.")
    figure = go.Figure()
    pair_count = 0
    for mode in modes:
        if mode == reference:
            continue
        seeds = sorted(set(grouped[mode]) & set(grouped[reference]), key=lambda value: (str(type(value)), value))
        if not seeds:
            continue
        pair_count += len(seeds)
        figure.add_trace(go.Bar(
            x=[f"seed {seed}" for seed in seeds], y=[grouped[mode][seed] - grouped[reference][seed] for seed in seeds], name=_label(mode, labels),
            marker={"color": _color(mode, colors), "cornerradius": 4},
            customdata=[[seed, grouped[mode][seed], grouped[reference][seed]] for seed in seeds],
            hovertemplate="Recorded seed: %{customdata[0]}<br>This model: %{customdata[1]:.6f}<br>Reference: %{customdata[2]:.6f}<br>Paired difference: %{y:+.6f}<extra></extra>",
        ))
    if not pair_count:
        return _empty(title, "No shared recorded seeds between the reference and comparison models.")
    _theme(figure, title)
    figure.update_layout(barmode="group", bargap=.32)
    figure.add_hline(y=0, line={"color": "#718198", "width": 1, "dash": "dash"})
    figure.update_xaxes(title="Complete training and testing with the same seed", showgrid=False)
    figure.update_yaxes(title=f"Δ {metric.upper()} (model − reference)", tickformat="+.4f")
    figure.add_annotation(text=f"Showing {pair_count} observed pairs. Positive = above reference; negative = below reference.", x=0, y=-.22, xref="paper", yref="paper", showarrow=False, xanchor="left", font={"color": "#718198", "size": 11})
    return figure
