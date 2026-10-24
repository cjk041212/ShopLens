"""Publication-style views of recorded evaluation and adaptation histories.

Every position and error bar comes from supplied records. Standard deviations
describe the three pipeline seeds; they are neither confidence intervals nor
evidence of statistical significance. Domains are kept separate throughout.
"""

from __future__ import annotations

import html
import math
from statistics import mean, stdev

import plotly.graph_objects as go
from plotly.subplots import make_subplots


MODELS = ("qwen_frozen", "qwen_semantic", "qwen_preference")
DOMAINS = ("baby", "sports", "clothing")
LABELS = dict(zip(MODELS, ("Frozen", "Semantic", "Preference")))
COLORS = dict(zip(MODELS, ("#64748b", "#6366f1", "#14b8a6")))
METRICS = {"ndcg@10": "NDCG@10", "recall@10": "Recall@10",
           "ndcg@20": "NDCG@20", "recall@20": "Recall@20"}
DEFAULT_SOURCE = "Recorded 27-run evaluation · results_qwen_summary.json"
_DIVERGING = [[0, "#dc7666"], [.5, "#f8fafc"], [1, "#14b8a6"]]


def _number(value, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite recorded number.")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return float(value)


def _seeds(values) -> list[int]:
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        raise ValueError("At least two recorded seeds are required for sample SD.")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in values) or len(set(values)) != len(values):
        raise ValueError("Recorded seeds must be distinct integers.")
    return list(values)


def _metric(metric: str) -> str:
    if metric not in METRICS:
        raise ValueError("Select a recorded NDCG or Recall metric at 10 or 20.")
    return METRICS[metric]


def _theme(figure: go.Figure, title: str, *, height: int, note: str, source: str) -> go.Figure:
    figure.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
        title={"text": f"<b>{title}</b>", "x": .01, "xanchor": "left", "y": .985, "yanchor": "top", "font": {"size": 20}},
        font={"family": "Inter, Avenir Next, Arial, sans-serif", "size": 12, "color": "#334155"},
        height=height, autosize=True, margin={"l": 95, "r": 28, "t": 100, "b": 82},
        legend={"orientation": "h", "x": 0, "y": 1 + 32 / max(1, height - 182),
                "yanchor": "bottom", "font": {"size": 11},
                "bgcolor": "rgba(0,0,0,0)", "tracegroupgap": 0},
        hoverlabel={"bgcolor": "white", "font_size": 12}, hovermode="closest",
        meta={"source": source, "interpretation": note},
    )
    figure.update_xaxes(gridcolor="#e9eef5", zeroline=False, tickfont={"size": 11},
                        title_font={"size": 12}, automargin=True, ticks="outside", ticklen=3)
    figure.update_yaxes(gridcolor="#e9eef5", zeroline=False, tickfont={"size": 11},
                        title_font={"size": 12}, automargin=True)
    return figure


def _domains(records: dict) -> list[str]:
    return [domain for domain in DOMAINS if any(key[0] == domain for key in records)]


def _result_map(results: list[dict], metric: str) -> dict:
    _metric(metric)
    mapped = {}
    for row in results:
        key = row.get("domain"), row.get("model")
        if key[0] not in DOMAINS or key[1] not in MODELS:
            raise ValueError("Evaluation rows must identify a known domain and objective.")
        if key in mapped:
            raise ValueError("Duplicate domain/objective aggregate.")
        seeds = _seeds(row.get("seeds"))
        if row.get("n_seeds") != len(seeds):
            raise ValueError("n_seeds must match the recorded seed identities.")
        value = _number(row.get(metric), metric, nonnegative=True)
        spread = _number(row.get(f"{metric}_std"), "Sample SD", nonnegative=True)
        if value > 1:
            raise ValueError("Recorded ranking metrics must be in [0, 1].")
        mapped[key] = {"value": value, "sd": spread, "seeds": seeds}
    if not mapped or set(mapped) != {(domain, model) for domain in _domains(mapped) for model in MODELS}:
        raise ValueError("Each displayed domain requires all three objective aggregates; missing values are not imputed.")
    seed_sets = {tuple(sorted(row["seeds"])) for row in mapped.values()}
    if len(seed_sets) != 1:
        raise ValueError("Aggregate comparisons require the same recorded seed identities.")
    return mapped


def performance_figure(results: list[dict], metric: str = "ndcg@10", *, source: str = DEFAULT_SOURCE) -> go.Figure:
    """Nine recorded means with sample-SD whiskers, in separate domain panels."""
    records, label = _result_map(results, metric), _metric(metric)
    domains = _domains(records)
    figure = make_subplots(rows=len(domains), cols=1, shared_xaxes=True, vertical_spacing=.15,
                           subplot_titles=[domain.title() for domain in domains])
    source_text = html.escape(source)
    for index, domain in enumerate(domains, 1):
        for position, model in enumerate(MODELS):
            row = records[domain, model]
            figure.add_trace(go.Scatter(
                x=[row["value"]], y=[2 - position], mode="markers", name=LABELS[model],
                legendgroup=model, showlegend=index == 1,
                marker={"size": 14, "color": COLORS[model], "line": {"color": "white", "width": 2}},
                error_x={"type": "data", "array": [row["sd"]], "visible": True,
                         "color": COLORS[model], "thickness": 3, "width": 6},
                customdata=[[row["sd"], len(row["seeds"]), ", ".join(map(str, row["seeds"]))]],
                hovertemplate=(f"<b>{domain.title()} · {LABELS[model]}</b><br>{label} mean: %{{x:.7f}}"
                               "<br>Sample SD: %{customdata[0]:.7f}<br>Pipeline seeds: %{customdata[2]}"
                               "<br>n = %{customdata[1]} · whiskers = ±1 sample SD"
                               f"<br>{source_text}<extra></extra>"),
            ), row=index, col=1)
        figure.update_yaxes(tickmode="array", tickvals=[2, 1, 0], ticktext=list(LABELS.values()),
                            showgrid=False, range=[-.55, 2.55], row=index, col=1)
    upper = max(row["value"] + row["sd"] for row in records.values())
    figure.update_xaxes(range=[0, upper * 1.13 if upper else .01], tickformat=".3f")
    n_seeds = len(next(iter(records.values()))["seeds"])
    figure.update_xaxes(title=f"{label} · original units · higher is better<br><span style='font-size:10px'>Mean ± sample SD · n = {n_seeds} matched pipeline seeds</span>", row=len(domains), col=1)
    _theme(figure, f"Measured performance · {label}", height=270 + 125 * len(domains),
           note="Mean ± 1 sample SD across matched pipeline seeds; no cross-domain averaging or significance claim.", source=source)
    figure.update_annotations(font={"size": 13, "color": "#0f172a"}, x=0, xanchor="left")
    return figure


def _paired_map(comparisons: list[dict], metric: str) -> dict:
    _metric(metric)
    mapped = {}
    for row in comparisons:
        if row.get("metric") != metric or row.get("before") != MODELS[0]:
            continue
        key = row.get("domain"), row.get("after")
        if key[0] not in DOMAINS or key[1] not in MODELS[1:]:
            raise ValueError("Frozen-reference comparisons must identify a known domain and LoRA objective.")
        if key in mapped:
            raise ValueError("Duplicate Frozen-reference comparison.")
        seeds = _seeds(row.get("seeds"))
        differences = row.get("differences")
        if not isinstance(differences, (list, tuple)) or len(differences) != len(seeds):
            raise ValueError("Each recorded paired seed must have one difference.")
        differences = [_number(value, "Paired difference") for value in differences]
        average = _number(row.get("mean_difference"), "Paired mean")
        spread = _number(row.get("std_difference"), "Paired sample SD", nonnegative=True)
        if not math.isclose(mean(differences), average, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Recorded paired mean does not match its seed differences.")
        if not math.isclose(stdev(differences), spread, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Recorded paired sample SD does not match its seed differences.")
        mapped[key] = {"seeds": seeds, "differences": differences, "mean": average, "sd": spread}
    if not mapped or set(mapped) != {(domain, model) for domain in _domains(mapped) for model in MODELS[1:]}:
        raise ValueError("Each displayed domain requires both Frozen-reference comparisons for the selected metric.")
    if len({tuple(sorted(row["seeds"])) for row in mapped.values()}) != 1:
        raise ValueError("Paired panels require matching recorded seed identities.")
    return mapped


def paired_delta_figure(paired_comparisons: list[dict], metric: str = "ndcg@10", *, source: str = DEFAULT_SOURCE) -> go.Figure:
    """A signed seed-level heatmap for each LoRA objective against Frozen."""
    records, label = _paired_map(paired_comparisons, metric), _metric(metric)
    seeds = sorted(next(iter(records.values()))["seeds"])
    domains = _domains(records)
    figure = make_subplots(rows=2, cols=1, vertical_spacing=.22,
                           subplot_titles=[f"{LABELS[model]} − Frozen" for model in MODELS[1:]])
    limit = max(abs(value) for row in records.values() for value in row["differences"]) or 1e-6
    for index, model in enumerate(MODELS[1:], 1):
        matrix = []
        for domain in domains:
            row = records[domain, model]
            values = dict(zip(row["seeds"], row["differences"]))
            matrix.append([values[seed] for seed in seeds])
        figure.add_trace(go.Heatmap(
            x=[f"Seed {seed}" for seed in seeds], y=[domain.title() for domain in domains],
            z=matrix, text=[[f"{value:+.5f}" for value in row] for row in matrix],
            texttemplate="%{text}", textfont={"size": 12}, xgap=7, ygap=7,
            coloraxis="coloraxis", hoverongaps=False,
            customdata=[[[seed] for seed in seeds] for _ in domains],
            hovertemplate=(f"<b>%{{y}} · {LABELS[model]} − Frozen</b><br>Matched seed %{{customdata[0]}}"
                           f"<br>Δ {label}: %{{z:+.7f}}<br>Positive = above Frozen; negative = below Frozen"
                           f"<br>{html.escape(source)}<extra></extra>"),
        ), row=index, col=1)
        figure.update_yaxes(autorange="reversed", showgrid=False, row=index, col=1)
        figure.update_xaxes(showgrid=False, row=index, col=1)
    _theme(figure, f"Every seed tells a story · Δ {label}", height=410 + 75 * len(domains),
           note="Actual same-seed LoRA minus Frozen differences; zero-centred shared colour scale, not significance.", source=source)
    figure.update_layout(coloraxis={"colorscale": _DIVERGING, "cmin": -limit, "cmax": limit, "cmid": 0,
                                   "colorbar": {"orientation": "h", "x": .5, "y": -.18, "xanchor": "center",
                                                "len": .8, "thickness": 10, "outlinewidth": 0,
                                                "title": {"text": "Below Frozen ← 0 → Above Frozen", "side": "bottom"},
                                                "tickformat": "+.4f", "tickfont": {"size": 10}}},
                         margin={"l": 90, "r": 26, "t": 90, "b": 130})
    figure.update_annotations(font={"size": 13, "color": "#0f172a"}, x=0, xanchor="left")
    return figure


def paired_forest_figure(paired_comparisons: list[dict], metric: str = "ndcg@10", *, source: str = DEFAULT_SOURCE) -> go.Figure:
    """Observed seed dots and paired mean ± sample SD, separately by domain."""
    records, label = _paired_map(paired_comparisons, metric), _metric(metric)
    figure = go.Figure()
    domains = _domains(records)
    tickvals, ticktext = [], []
    for d_index, domain in enumerate(domains):
        for m_index, model in enumerate(MODELS[1:]):
            position = len(domains) * 2 - 1 - (d_index * 2 + m_index)
            row = records[domain, model]
            tickvals.append(position)
            ticktext.append(f"{domain.title()} · {LABELS[model]}")
            # Spread points only along the categorical axis; metric positions are exact.
            offsets = [((index / (len(row["seeds"]) - 1)) - .5) * .3 for index in range(len(row["seeds"]))]
            figure.add_trace(go.Scatter(
                x=row["differences"], y=[position + offset for offset in offsets], mode="markers",
                name=f"{LABELS[model]} · paired seeds", legendgroup=model, showlegend=False,
                marker={"size": 7, "color": COLORS[model], "opacity": .5, "symbol": "circle"},
                customdata=[[seed] for seed in row["seeds"]],
                hovertemplate=(f"<b>{domain.title()} · {LABELS[model]} − Frozen</b><br>Matched seed %{{customdata[0]}}"
                               f"<br>Δ {label}: %{{x:+.7f}}<br>{html.escape(source)}<extra></extra>"),
            ))
            figure.add_trace(go.Scatter(
                x=[row["mean"]], y=[position], mode="markers", name=LABELS[model],
                legendgroup=model, showlegend=d_index == 0,
                marker={"size": 13, "color": COLORS[model], "symbol": "diamond", "line": {"width": 1.5, "color": "white"}},
                error_x={"type": "data", "array": [row["sd"]], "visible": True,
                         "color": COLORS[model], "thickness": 2.5, "width": 5},
                customdata=[[row["sd"], len(row["seeds"])]],
                hovertemplate=(f"<b>{domain.title()} · {LABELS[model]} − Frozen</b><br>Paired mean Δ {label}: %{{x:+.7f}}"
                               "<br>Sample SD of paired differences: %{customdata[0]:.7f}"
                               "<br>n = %{customdata[1]} matched seeds · whiskers = ±1 sample SD"
                               f"<br>{html.escape(source)}<extra></extra>"),
            ))
    figure.add_vline(x=0, line={"color": "#94a3b8", "dash": "dash", "width": 1.5})
    figure.update_yaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, range=[-.55, len(domains) * 2 - .45], showgrid=False)
    figure.update_xaxes(title=f"Δ {label} · LoRA − Frozen · reference = 0<br><span style='font-size:10px'>Dots: matched seeds · diamonds: mean ± sample SD</span>", tickformat="+.4f")
    _theme(figure, "Direction, consistency, and seed variation", height=290 + 90 * len(domains),
           note="Small dots are exact seed differences; diamonds and whiskers show paired mean ± sample SD, not confidence intervals.", source=source)
    figure.update_layout(margin={"l": 155, "r": 28, "t": 95, "b": 85})
    return figure


def _training_map(traces: list[dict]) -> tuple[dict, int]:
    mapped, observed_seeds = {}, set()
    for trace in traces:
        key = trace.get("domain"), trace.get("model")
        if key[0] not in DOMAINS or key[1] not in MODELS:
            raise ValueError("Training rows must identify a known domain and objective.")
        if key in mapped:
            raise ValueError("Duplicate domain/objective training trace.")
        seed = trace.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Training seed must be a recorded integer.")
        epochs, values = trace.get("epochs"), trace.get("validation_ndcg")
        if not isinstance(epochs, (list, tuple)) or not epochs or not isinstance(values, (list, tuple)) or len(epochs) != len(values):
            raise ValueError("Each recorded epoch must have one validation NDCG@10 value.")
        if any(isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1 for epoch in epochs) or sorted(set(epochs)) != list(epochs):
            raise ValueError("Recorded epochs must be distinct, increasing positive integers.")
        values = [_number(value, "Validation NDCG@10", nonnegative=True) for value in values]
        if any(value > 1 for value in values):
            raise ValueError("Validation NDCG@10 must be in [0, 1].")
        selected = trace.get("selected_epoch")
        if isinstance(selected, bool) or not isinstance(selected, int) or selected not in epochs:
            raise ValueError("Selected checkpoint must identify an actually recorded epoch.")
        mapped[key] = {"epochs": list(epochs), "values": values, "selected": selected}
        observed_seeds.add(seed)
    if not mapped or set(mapped) != {(domain, model) for domain in _domains(mapped) for model in MODELS}:
        raise ValueError("Each displayed domain requires all three objective adaptation histories.")
    if len(observed_seeds) != 1:
        raise ValueError("A training view must use one matched seed; do not mix pipeline seeds.")
    return mapped, next(iter(observed_seeds))


def training_figure(traces: list[dict], *, source: str = "Recorded adaptation histories · results_qwen_summary.json") -> go.Figure:
    """Per-domain validation histories for one pipeline seed, with saved selections."""
    records, seed = _training_map(traces)
    domains = _domains(records)
    figure = make_subplots(rows=len(domains), cols=1, shared_xaxes=True, vertical_spacing=.15,
                           subplot_titles=[domain.title() for domain in domains])
    all_values = [value for row in records.values() for value in row["values"]]
    epochs = sorted({epoch for row in records.values() for epoch in row["epochs"]})
    for index, domain in enumerate(domains, 1):
        for model in MODELS:
            row = records[domain, model]
            figure.add_trace(go.Scatter(
                x=row["epochs"], y=row["values"], mode="lines+markers", name=LABELS[model],
                legendgroup=model, showlegend=index == 1,
                line={"color": COLORS[model], "width": 3},
                marker={"size": 8, "color": COLORS[model], "line": {"color": "white", "width": 1}},
                customdata=[["Selected checkpoint" if epoch == row["selected"] else "Recorded epoch", seed] for epoch in row["epochs"]],
                hovertemplate=(f"<b>{domain.title()} · {LABELS[model]}</b><br>Pipeline seed %{{customdata[1]}} · adaptation epoch %{{x}}"
                               "<br>Validation NDCG@10: %{y:.7f}<br>%{customdata[0]}"
                               f"<br>{html.escape(source)}<extra></extra>"),
            ), row=index, col=1)
            selected_index = row["epochs"].index(row["selected"])
            figure.add_trace(go.Scatter(
                x=[row["selected"]], y=[row["values"][selected_index]], mode="markers",
                marker={"size": 17, "symbol": "circle-open", "color": COLORS[model], "line": {"width": 2}},
                legendgroup=model, name="Selected checkpoint", showlegend=False,
                hoverinfo="skip",
            ), row=index, col=1)
        figure.update_yaxes(title="Validation<br>NDCG@10", range=[0, max(all_values) * 1.15 or .01], tickformat=".3f", row=index, col=1)
    figure.update_xaxes(tickmode="array", tickvals=epochs, range=[min(epochs) - .18, max(epochs) + .18])
    figure.update_xaxes(title="Recorded adaptation epoch · ring = selected checkpoint", row=len(domains), col=1)
    _theme(figure, f"Training dynamics · matched pipeline seed {seed}", height=325 + 135 * len(domains),
           note="Recorded adaptation validation NDCG@10 only. Rings indicate saved validation-selected checkpoints; no test points or interpolated epochs.", source=source)
    figure.update_annotations(font={"size": 13, "color": "#0f172a"}, x=0, xanchor="left")
    figure.layout.meta["seed"] = seed
    return figure
