#!/usr/bin/env python3
"""Export measured benchmark charts and exact data as a self-contained HTML page."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#d49642", "#64748b", "#6366f1", "#14b8a6"]
MODES = ["qwen_frozen", "qwen_semantic", "qwen_preference"]
LABELS = dict(zip(MODES, ["Frozen", "Semantic", "Preference"]))


def build_figures(root):
    def read(path):
        return json.loads((root / path).read_text())
    esci = read("results/esci/summary.json")
    fashion = read("results/fashion/summary.json")
    diagnostics = read("results/diagnostics/recommendation_diagnostics.json")
    performance = read("results/performance/studio.json")
    if not esci.get("complete") or fashion.get("status") != "complete" or not diagnostics.get("complete") or not performance.get("complete"):
        raise ValueError("All four real measurements must be complete before export.")
    figures = []
    fig = go.Figure(go.Bar(x=[r["label"] for r in esci["models"]], y=[r["metrics"]["ndcg@10"] for r in esci["models"]],
                           marker_color=COLORS))
    fig.update_yaxes(title="Mean query NDCG@10", rangemode="tozero")
    figures.append(("ESCI · 500 official English test queries", "Complete official candidate lists of at most 40 products; Clothing seed42 transfer, title-only. Not full-catalog search.", fig))
    fig = go.Figure()
    methods = [r["method"] for r in fashion["macro_metrics"]]
    for index, k in enumerate((1, 10, 50)):
        fig.add_bar(name=f"Recall@{k}", x=[m.replace("_", " ").title() for m in methods],
                    y=[r[f"recall@{k}"] for r in fashion["macro_metrics"]], marker_color=COLORS[index + 1])
    fig.update_yaxes(title="Macro target retrieval rate", tickformat=".0%", rangemode="tozero")
    figures.append(("FashionIQ · image + language retrieval", "287 available-image official-validation queries. Category pools: 281 / 292 / 293; each reference excluded. One paired target per query. Not full-gallery scores.", fig))
    curve = fashion["training_curve"]
    fig = go.Figure(go.Scatter(x=[r["epoch"] for r in curve], y=[r["dev_macro_recall@10"] for r in curve], mode="lines+markers", name="Internal dev", line={"color": COLORS[-1]}))
    selected = next(r for r in curve if r["epoch"] == fashion["selected_epoch"])
    fig.add_scatter(x=[selected["epoch"]], y=[selected["dev_macro_recall@10"]], mode="markers", name="Selected checkpoint", marker={"size": 15, "symbol": "circle-open", "color": COLORS[-2]})
    fig.update_xaxes(title="Training epoch · CLIP frozen")
    fig.update_yaxes(title="Internal-dev macro Recall@10", tickformat=".0%")
    figures.append(("Fusion checkpoint selection", "850 train and 82 internal-dev queries after cross-split JPEG deduplication. Epoch12 selected using internal dev only; fixed evaluation set and gallery preserved.", fig))
    for domain in ("baby", "sports", "clothing"):
        fig = go.Figure()
        for mode, color in zip(MODES, COLORS[1:]):
            rows = sorted([r for r in diagnostics["history_curves"] if r["domain"] == domain and r["model"] == mode], key=lambda r: r["history_cap"])
            fig.add_scatter(name=LABELS[mode], x=[r["history_cap"] for r in rows], y=[r["ndcg@10"] for r in rows], mode="lines+markers", line={"color": color})
        fig.update_xaxes(title="Visible history cap", type="category")
        fig.update_yaxes(title="User-macro NDCG@10", rangemode="tozero")
        figures.append((f"{domain.title()} · same-case history diagnostic", "Fixed seed42, 1,000 shared test users. All strictly earlier positives remain excluded from candidates at every cap.", fig))
        rows = [r for r in diagnostics["exposure"] if r["domain"] == domain]
        groups = rows[0]["popularity_exposure"]
        fig = go.Figure(go.Bar(name="Catalog share", x=[g["bucket"] for g in groups], y=[g["catalog_share"] for g in groups], marker_color="#cbd5e1"))
        for row, color in zip(rows, COLORS[1:]):
            fig.add_bar(name=LABELS[row["model"]], x=[g["bucket"] for g in row["popularity_exposure"]], y=[g["exposure_share"] for g in row["popularity_exposure"]], marker_color=color)
        fig.update_yaxes(title="Catalog / exposure share", tickformat=".0%")
        fig.update_xaxes(title="Distinct positive TRAIN users per item")
        figures.append((f"{domain.title()} · exposure concentration", "Cap20 Top-10 recommendation slots, event weighted. Catalog includes never-exposed products. Coverage is not a quality or fairness guarantee.", fig))
    stages = {r["stage"]: r for r in performance["stages"]}
    cases = ["text", "multimodal_feedback", "avoid_budget"]
    fig = go.Figure()
    for version, color in (("before", COLORS[1]), ("after", COLORS[-1])):
        fig.add_bar(name=version.title(), x=[c.replace("_", " ") for c in cases], y=[stages[f"three_rankers_{c}_{version}"]["p50_ms"] for c in cases], marker_color=color)
    fig.update_yaxes(title="Three rankers · p50 milliseconds", rangemode="tozero")
    figures.append(("Studio · metadata optimization", "Real 22,491-product catalog, three objectives. Alternating before/after order; identical complete returned rows required. CPU engine only, excluding browser.", fig))
    return figures, {"esci": esci, "fashion": fashion, "diagnostics": diagnostics, "performance": performance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/benchmark-gallery")
    args = parser.parse_args()
    figures, evidence = build_figures(ROOT)
    args.output.mkdir(parents=True, exist_ok=True)
    blocks = []
    for index, (title, note, figure) in enumerate(figures):
        figure.update_layout(template="plotly_white", height=460, font={"family": "Arial, sans-serif", "size": 14},
                             margin={"l": 80, "r": 35, "t": 35, "b": 70}, legend={"orientation": "h", "y": 1.15}, barmode="group")
        blocks.append(f"<section><h2>{html.escape(title)}</h2><p>{html.escape(note)}</p>" + pio.to_html(figure, full_html=False, include_plotlyjs=True if index == 0 else False,
                     config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}}, div_id=f"benchmark-{index}") + "</section>")
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ShopLens measured benchmarks</title><style>body{margin:40px auto;max-width:1120px;padding:0 24px;background:#f6f8fc;color:#17243a;font:16px/1.6 Arial}section{background:white;border:1px solid #e2e8f0;padding:24px;margin:24px 0;border-radius:16px}p{color:#536679}a{color:#5454bf}</style><h1>ShopLens · measured benchmark gallery</h1><p>Three independent data sources; distinct tasks and metrics. Original 27-run recommendation results remain separate. Charts contain real recorded outputs; no model inference, network or training is needed to view this page.</p><a href="evidence.json">Download exact source evidence JSON</a>' + ''.join(blocks) + '</html>'
    (args.output / "index.html").write_text(page)
    (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "figures": len(figures), "offline": True}))


if __name__ == "__main__":
    main()
