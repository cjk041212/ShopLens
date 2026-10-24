"""Presentation views of the recorded experiment, available without model weights."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from .experiment_data import load_experiment_summary
from . import experiment_charts as charts


DOMAIN_LABELS = {"all": "All three collections", "baby": "Baby", "sports": "Sports & Outdoors", "clothing": "Clothing & Accessories"}
METRICS = ("ndcg@10", "recall@10", "ndcg@20", "recall@20")
VIEWS = ("Performance landscape", "Paired effects", "Training dynamics")
GALLERY_CSS = """
<style>
.gallery-hero {padding:22px 28px;border:1px solid #dbe3ef;border-radius:22px;background:radial-gradient(ellipse at 95% 0%,#d4f3ea 0%,transparent 48%),linear-gradient(120deg,#eef0ff,#f8fafc);margin:0 0 18px;}
.gallery-kicker {font-size:11px;font-weight:700;letter-spacing:2px;color:#6366f1;margin-bottom:10px;}
.gallery-hero h1 {letter-spacing:-1.5px;margin:0 0 8px!important;padding:0!important;}
.gallery-hero p {color:#607089;max-width:760px;line-height:1.65;margin:0;font-size:14px;}
.gallery-finding {border-left:3px solid #6366f1;background:#f0f1ff;color:#37425b;border-radius:0 10px 10px 0;padding:13px 17px;margin:6px 0 22px;font-size:13px;line-height:1.7;}
.gallery-legend {display:flex;gap:20px;flex-wrap:wrap;font-size:12px;color:#52647c;margin:12px 0;}
.gallery-legend span::before {content:'';display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--c);margin-right:7px;}
.st-key-gallery_summary [data-testid="stMetric"] {padding:14px 18px;}
.st-key-gallery_summary [data-testid="stMetricValue"] {font-size:1.8rem!important;}
@media(max-width:700px) {.gallery-hero {padding:22px;} .gallery-hero h1 {font-size:2.15rem!important;}.st-key-gallery_summary [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}.st-key-gallery_summary [data-testid="stColumn"] {flex:1 1 40%!important;min-width:0!important;}}
</style>
"""


def _metric_label(metric):
    return metric.replace("ndcg", "NDCG").replace("recall", "Recall")


def _chart(figure, key, image_format):
    st.plotly_chart(figure, width="stretch", key=key, config={
        "displaylogo": False,
        "toImageButtonOptions": {"format": image_format, "filename": key, "width": 1600, "height": int(figure.layout.height or 700), "scale": 2},
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    })


def render_gallery(source_root, runtime_root=None):
    st.markdown(GALLERY_CSS, unsafe_allow_html=True)
    st.markdown('<div class="gallery-hero"><div class="gallery-kicker">SHOPLENS / EXPERIMENT GALLERY</div><h1>Experiments, in focus.</h1><p>Explore the measured landscape, follow matched seed differences, and inspect how validation changes during adaptation. Every mark comes from a recorded run.</p></div>', unsafe_allow_html=True)
    try:
        document = load_experiment_summary(Path(source_root), runtime_root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Recorded experiment summary is unavailable: {exc}")
        st.caption("The gallery requires a complete, validated summary. Missing measurements are never replaced with example scores.")
        return
    provenance = document["provenance"]
    results = document["results"]
    with st.container(key="gallery_summary"):
        for column, (label, value, note) in zip(st.columns(4), [
            ("Recorded runs", "27", "Complete training pipelines"),
            ("Collections", "3", "One Amazon 2014 source"),
            ("Text objectives", "3", "Frozen / Semantic / Preference"),
            ("Training seeds", "3", "42 / 43 / 44"),
        ]):
            column.metric(label, value)
            column.caption(note)
    view = st.radio("Explore the experiment", VIEWS, horizontal=True, key="gallery_view")
    controls = st.columns([2, 1, 1])
    domain = controls[0].selectbox("Collection scope", list(DOMAIN_LABELS), format_func=DOMAIN_LABELS.get, key="gallery_domain")
    metric = controls[1].selectbox("Test metric" if view != VIEWS[2] else "Validation metric", METRICS if view == VIEWS[0] else ("ndcg@10",), format_func=_metric_label, key="gallery_metric", disabled=view != VIEWS[0])
    image_format = controls[2].selectbox("Figure export", ["svg", "png"], format_func=lambda value: "Vector SVG" if value == "svg" else "High-resolution PNG", key="gallery_image_format")
    filtered = [row for row in results if domain == "all" or row["domain"] == domain]
    pairs = [row for row in document["paired_comparisons"] if domain == "all" or row["domain"] == domain]
    selected_domains = [name for name in ("baby", "sports", "clothing") if any(row["domain"] == name for row in filtered)]
    leaders = []
    for collection in selected_domains:
        rows = [row for row in filtered if row["domain"] == collection]
        best = max(row[metric] for row in rows)
        leaders.append(f'{DOMAIN_LABELS[collection]}: ' + ' / '.join(charts.LABELS[row["model"]] for row in rows if row[metric] == best))
    finding = "; ".join(leaders)
    if view == VIEWS[2]:
        insight = "Frozen keeps its text encoder fixed while its recommendation head trains. Lines show two recorded validation observations; circles mark validation-selected checkpoints."
        insight_title = "What the training view measures"
    elif view == VIEWS[1]:
        messages = []
        for model in ("qwen_semantic", "qwen_preference"):
            differences = [value for row in pairs if row["before"] == "qwen_frozen" and row["after"] == model for value in row["differences"]]
            messages.append(f"{charts.LABELS[model]}: {sum(value > 0 for value in differences)}/{len(differences)} matched runs above Frozen")
        insight = "; ".join(messages) + ". This describes observed pairs, not statistical significance."
        insight_title = "The direction of change · NDCG@10"
    else:
        insight = finding + ". These are observations under the declared training budget; differences alone do not establish significance."
        insight_title = "Highest recorded mean · " + _metric_label(metric)
    st.markdown(f'<div class="gallery-finding"><b>{html.escape(insight_title)}</b><br>{html.escape(insight)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="gallery-legend"><span style="--c:#64748b">Frozen</span><span style="--c:#6366f1">Semantic alignment</span><span style="--c:#14b8a6">User preference</span></div>', unsafe_allow_html=True)
    if view == VIEWS[0]:
        st.subheader("The full performance landscape")
        st.caption("One point per domain and objective. Whiskers show sample standard deviation across three complete training seeds.")
        figure = charts.performance_figure(filtered, metric)
        export_rows = [{"domain": row["domain"], "model": row["model"], "metric": metric,
                        "mean": row[metric], "sample_sd": row[f"{metric}_std"], "n_seeds": row["n_seeds"]} for row in filtered]
        filename = f"shoplens_landscape_{domain}_{metric.replace('@', '_')}"
    elif view == VIEWS[1]:
        st.subheader("What changed for the same seed?")
        st.caption("Coral is below Frozen; teal is above. The seed matrix shows each recorded NDCG@10 pair. The effect overview shows seed points and mean ± sample SD, which is not a confidence interval.")
        style = st.radio("Paired view", ["Seed matrix", "Effect overview"], horizontal=True, key="gallery_paired_style")
        figure = charts.paired_delta_figure(pairs) if style == "Seed matrix" else charts.paired_forest_figure(pairs)
        export_rows = [{"domain": row["domain"], "model": row["after"], "reference": row["before"], "metric": row["metric"], "seed": seed, "difference": delta}
                       for row in pairs if row["before"] == "qwen_frozen" for seed, delta in zip(row["seeds"], row["differences"])]
        filename = f"shoplens_paired_{domain}_{'matrix' if style == 'Seed matrix' else 'effects'}"
    else:
        st.subheader("Follow adaptation, epoch by epoch")
        seed = st.radio("Training seed", provenance.get("seeds", [42, 43, 44]), horizontal=True, key="gallery_training_seed")
        traces = [row for row in document["training_all"] if row["seed"] == seed and (domain == "all" or row["domain"] == domain)]
        figure = charts.training_figure(traces)
        st.caption(f"Validation NDCG@10 · seed {seed} · two recorded adaptation epochs. Marked checkpoints are selected using validation; test metrics do not select the epoch. These lines do not show unrecorded intermediate steps.")
        export_rows = [{"domain": row["domain"], "model": row["model"], "seed": row["seed"], "epoch": epoch,
                        "validation_ndcg@10": value, "selected_checkpoint": epoch == row["selected_epoch"]}
                       for row in traces for epoch, value in zip(row["epochs"], row["validation_ndcg"])]
        filename = f"shoplens_training_{domain}_seed{seed}"
    _chart(figure, filename, image_format)
    st.caption("Hover to inspect exact values. Use the camera in the chart toolbar to save the selected SVG or PNG format; use fullscreen for presentation.")
    exports = st.columns(3)
    exports[0].download_button("Save interactive figure", figure.to_html(full_html=True, include_plotlyjs=True, config={"displaylogo": False}).encode(),
                               file_name=filename+".html", mime="text/html", key="gallery_html", width="stretch")
    exports[1].download_button("Download plotted data", pd.DataFrame(export_rows).to_csv(index=False).encode(),
                               file_name=filename+".csv", mime="text/csv", key="gallery_csv", width="stretch")
    exports[2].download_button("Download source record", json.dumps({"provenance": provenance, "view": view, "rows": export_rows}, indent=2).encode(),
                               file_name=filename+".json", mime="application/json", key="gallery_json", width="stretch")
    with st.expander("Read the values & source scope"):
        st.dataframe(pd.DataFrame(export_rows), hide_index=True, width="stretch")
        st.caption(f"Source: {provenance.get('source_relative', 'submission/results_qwen_summary.json')} · SHA256 {provenance.get('source_sha256', 'unavailable')}")
        st.write("Formal history-to-item evaluation uses the full eligible catalog and a fixed chronological protocol. Studio free-text retrieval is a separate exploratory interaction. Three seeds describe observed variability; standard deviations are not confidence intervals. The gallery reads historical measured outputs and does not run training.")
