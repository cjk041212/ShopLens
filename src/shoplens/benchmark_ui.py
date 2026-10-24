"""Inspect measured, portable benchmarks without loading training or runtime data."""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .experiment_charts import COLORS, LABELS

SEARCH_LABELS = {"tfidf": "TF-IDF", **LABELS}
SEARCH_COLORS = {"tfidf": "#d49642", **COLORS}
FASHION_LABELS = {"image_only": "Image only", "text_only": "Text only",
                  "fixed_fusion": "Fixed fusion", "trained_fusion": "Learned fusion"}
FASHION_COLORS = {"image_only": "#94a3b8", "text_only": "#d49642",
                  "fixed_fusion": "#6366f1", "trained_fusion": "#14b8a6"}


@st.cache_data(show_spinner=False)
def _json(path: str, modified_ns: int):
    del modified_ns
    return json.loads(Path(path).read_text())


def _load(path):
    return _json(str(path), path.stat().st_mtime_ns)


def _chart(fig, *, height=370):
    fig.update_layout(template="plotly_white", height=height,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"family": "Inter, Avenir Next, Arial, sans-serif", "color": "#334155", "size": 12},
                      margin={"l": 58, "r": 22, "t": 45, "b": 65},
                      legend={"orientation": "h", "y": 1.12, "x": 0},
                      hoverlabel={"bgcolor": "white"})
    fig.update_xaxes(gridcolor="#e9eef5", zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor="#e9eef5", zeroline=False, automargin=True)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def _kpis(values):
    for col, (label, value, detail) in zip(st.columns(len(values)), values):
        with col:
            st.metric(label, value)
            st.caption(detail)


def _download(value, name):
    st.download_button("Download measured evidence · JSON", json.dumps(value, indent=2, allow_nan=False),
                       file_name=name, mime="application/json", key=f"download_{name}")


def _limitations(summary):
    with st.expander("Protocol, provenance and limits"):
        for value in summary.get("limitations", []):
            st.markdown(f"- {value}")
        st.json(summary.get("protocol", {}), expanded=False)


def _search(root):
    path = root / "results/esci/summary.json"
    if not path.is_file():
        st.info("The measured ESCI benchmark is not present in this checkout.")
        return
    from .search_benchmark import load_search_results
    recorded = load_search_results(root)
    summary, queries = recorded["summary"], recorded["queries"]
    st.subheader("Does a shopping request find the right product?")
    st.caption("Amazon ESCI · official English test queries · graded relevance · fixed candidate reranking")
    _kpis([("Test queries", f"{len(queries):,}", "Selected before inspecting model outcomes"),
           ("Candidates", f"{sum(len(q['candidates']) for q in queries):,}", "Official query-specific lists, at most 40"),
           ("Encoders", "3 + TF-IDF", "Frozen / Semantic / Preference + lexical baseline"),
           ("New manual labels", "0", "Exact / Substitute / Complement / Irrelevant")])
    metric = st.segmented_control("Search metric", ["ndcg@10", "ndcg@all", "mrr_exact@10", "recall_exact@10"],
                                  default="ndcg@10", key="esci_metric") or "ndcg@10"
    models = summary["models"]
    fig = go.Figure()
    for row in models:
        mode = row["model"]
        fig.add_bar(name=SEARCH_LABELS.get(mode, row.get("label", mode)), x=[SEARCH_LABELS.get(mode, row.get("label", mode))],
                    y=[row["metrics"][metric]], marker_color=SEARCH_COLORS.get(mode, "#d49642"),
                    text=[f"{row['metrics'][metric]:.4f}"], textposition="outside",
                    hovertemplate="%{x}<br>Measured score: %{y:.5f}<extra></extra>")
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title=metric.upper(), rangemode="tozero")
    _chart(fig, height=320)
    st.caption("Title-only transfer test. Clothing seed42 LoRA adapters were trained on Amazon 2014; they were not fitted on ESCI. These scores do not measure whole-catalog search.")
    with st.expander("Which queries improve or decline against Frozen?"):
        fig = go.Figure()
        paired = []
        for mode in ("qwen_semantic", "qwen_preference"):
            differences = [q["metrics"][mode][metric] - q["metrics"]["qwen_frozen"][metric] for q in queries]
            fig.add_histogram(x=differences, name=LABELS[mode], marker_color=COLORS[mode], opacity=.65,
                              xbins={"start": -1, "end": 1, "size": .05})
            paired.append({"Model": LABELS[mode], "Wins": sum(d > 1e-12 for d in differences),
                           "Ties": sum(abs(d) <= 1e-12 for d in differences), "Losses": sum(d < -1e-12 for d in differences)})
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title=f"Same-query {metric.upper()} change versus Frozen")
        fig.update_yaxes(title="Official test queries")
        _chart(fig)
        st.dataframe(pd.DataFrame(paired), hide_index=True, width="stretch")
        st.caption("Paired query outcomes are descriptive transfer results from the fixed checkpoints. No test-based adapter selection or significance claim.")
    st.markdown("#### Inspect a real labeled query")
    choices = list(range(len(queries)))
    selected = st.selectbox("Official test request", choices,
                            format_func=lambda i: f"{queries[i]['query']} · #{queries[i]['query_id']}", key="esci_query")
    query = queries[selected]
    st.info(f"“{query['query']}” · {len(query['candidates'])} official candidates")
    mode_ids = [row["model"] for row in models]
    columns = st.columns(len(mode_ids))
    for col, mode in zip(columns, mode_ids):
        with col:
            st.markdown(f"**{SEARCH_LABELS.get(mode, mode)}**")
            ordered = sorted(query["candidates"], key=lambda c: c["ranks"][mode])
            for product in ordered[:3]:
                label = product["esci_label"]
                color = {"E": "#117d68", "S": "#5b55bc", "C": "#9d6d21", "I": "#9b5362"}.get(label, "#64748b")
                title = html.escape(str(product.get("product_title") or product["product_id"]))
                st.markdown(f'<div class="bench-product"><span style="color:{color};font-weight:700">#{product["ranks"][mode]} · {label}</span><p>{title}</p></div>', unsafe_allow_html=True)
    rows = []
    for product in query["candidates"]:
        rows.append({"Product": product.get("product_title", ""), "Official label": product["esci_label"],
                     **{SEARCH_LABELS.get(mode, mode) + " rank": product["ranks"][mode] for mode in mode_ids}})
    with st.expander("Every official candidate and rank"):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("E = exact · S = substitute · C = complement · I = irrelevant. Labels are evaluation evidence, never an encoder input.")
    _limitations(summary)
    _download(summary, "shoplens_esci_summary.json")


def _fashion_results(root, methods, target=None):
    image_dir = root / "assets/benchmarks/fashion/images"
    for col, (method, label) in zip(st.columns(4), FASHION_LABELS.items()):
        record = methods[method]
        with col:
            st.markdown(f"**{label}**")
            if target is not None:
                st.metric("Paired target rank", record["target_rank"])
            for rank, image_id in enumerate(record["top_ids"][:3], 1):
                st.image(str(image_dir / f"{image_id}.jpg"), width="stretch")
                st.caption(f"#{rank} · {'✓ paired target' if image_id == target else image_id}")


def _fashion(root, runtime_root):
    path = root / "results/fashion/summary.json"
    if not path.is_file():
        st.info("The measured FashionIQ experiment is not present in this checkout.")
        return
    from .fashion_retrieval import load_fashion_results
    recorded = load_fashion_results(root)
    summary, queries = recorded["summary"], recorded["queries"]
    if summary.get("status") != "complete":
        st.warning("FashionIQ measurement is incomplete; no metrics are displayed.")
        return
    st.subheader("“Like this, but…” becomes an image retrieval task")
    st.caption("FashionIQ · actual reference images and modification captions · a lightweight learned fusion model")
    phase = summary["phase_queries"]
    _kpis([("Training queries", str(phase["train"]), "Frozen CLIP; train fusion only"),
           ("Evaluation queries", str(phase["evaluation"]), "Official validation subset, held out from training"),
           ("Downloaded images", f"{summary['image_count']:,}", "Real photographs, with per-file source hashes"),
           ("Trainable parameters", f"{summary['trainable_parameters']:,}", "Small projection, not a new foundation model")])
    metric = st.segmented_control("Image retrieval metric", ["recall@1", "recall@5", "recall@10", "recall@50"], default="recall@10", key="fashion_metric") or "recall@10"
    fig = go.Figure()
    for row in summary["macro_metrics"]:
        method = row["method"]
        fig.add_bar(name=FASHION_LABELS[method], x=[FASHION_LABELS[method]], y=[row[metric]],
                    marker_color=FASHION_COLORS[method], text=[f"{row[metric]:.1%}"], textposition="outside")
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title="Target retrieval rate", tickformat=".0%", rangemode="tozero")
    _chart(fig, height=320)
    st.caption("Retrieval uses the fixed selected image pool within each category; the reference image is excluded. These subset scores are not the official full-gallery leaderboard.")
    st.caption(f"Available-image evaluation pools: " + " · ".join(f"{category}: {count}" for category, count in summary["candidate_counts"]["evaluation"].items()) + f". {summary['download_failures']} unavailable images were omitted without replacement.")
    st.markdown("#### Follow a reference image through four retrieval methods")
    category = st.segmented_control("Garment category", ["dress", "shirt", "toptee"], default="dress", key="fashion_category") or "dress"
    examples = [q for q in queries if q["category"] == category]
    position = st.selectbox("Recorded image + language request", range(len(examples)),
                            format_func=lambda i: examples[i]["text"], key=f"fashion_query_{category}")
    query = examples[position]
    view = st.radio("Image + language interaction", ["Recorded labeled example", "Try my own modification"], horizontal=True, key="fashion_interaction")
    if view == "Try my own modification":
        left, right = st.columns([1, 3])
        with left:
            st.image(str(root / "assets/benchmarks/fashion/images" / f"{query['reference']}.jpg"), caption="Reference image", width="stretch")
        with right:
            with st.form(f"fashion_edit_{query['query_id']}"):
                text = st.text_input("Describe your modification", value=query["text"], max_chars=1000)
                submitted = st.form_submit_button("Run my modification", type="primary")
            st.caption("Runs the prepared local CLIP text encoder and the bundled fusion head. No downloads or training. Your edited request has no ground-truth target or measured accuracy.")
        token = (query["reference"], category, text.strip(), summary["model_sha256"], summary["features_sha256"])
        if submitted:
            from .fashion_retrieval import compose_fashion_query
            try:
                with st.spinner("Encoding your edit locally and ranking the selected image pool…"):
                    value = compose_fashion_query(root, runtime_root, query["reference"], text, category)
                st.session_state["fashion_custom_result"] = (token, value)
            except (OSError, ValueError, RuntimeError, ImportError) as exc:
                st.session_state.pop("fashion_custom_result", None)
                st.error(str(exc))
        saved = st.session_state.get("fashion_custom_result")
        if saved and saved[0] == token:
            st.success(f"New local ranking · {saved[1]['candidate_count']} candidates · reference excluded")
            st.caption(f"Showing the last submitted modification: {saved[1].get('text', text)}")
            _fashion_results(root, saved[1]["methods"])
        else:
            st.info("Run the edit to compare fresh image-only, text-only, fixed-fusion and learned-fusion rankings.")
        _limitations(summary)
        return
    left, middle, right = st.columns([1, 2.3, 1])
    image_dir = root / "assets/benchmarks/fashion/images"
    with left:
        st.image(str(image_dir / f"{query['reference']}.jpg"), caption="Reference · excluded from results", width="stretch")
    with middle:
        st.markdown("**Change requested by the dataset annotator**")
        st.markdown(f"### {query['text']}")
        st.caption("The paired target is used only to measure retrieval. The model sees the reference photograph and the modification text.")
    with right:
        st.image(str(image_dir / f"{query['target']}.jpg"), caption="Held-out paired target", width="stretch")
    _fashion_results(root, query["methods"], query["target"])
    with st.expander("Per-category measurements and training curve"):
        st.dataframe(pd.DataFrame(summary["category_metrics"]), hide_index=True, width="stretch")
        curve = summary["training_curve"]
        fig = go.Figure(go.Scatter(x=[r["epoch"] for r in curve], y=[r["dev_macro_recall@10"] for r in curve],
                                  mode="lines+markers", name="Internal development R@10", line={"color": "#14b8a6"}))
        selected = next(r for r in curve if r["epoch"] == summary["selected_epoch"])
        fig.add_scatter(x=[selected["epoch"]], y=[selected["dev_macro_recall@10"]], mode="markers", name="Selected checkpoint",
                        marker={"size": 16, "color": "#6366f1", "symbol": "circle-open", "line": {"width": 3}})
        fig.update_xaxes(title="Fusion training epoch · CLIP remains frozen")
        fig.update_yaxes(title="Internal-dev macro Recall@10", tickformat=".0%")
        _chart(fig)
    _limitations(summary)
    _download(summary, "shoplens_fashion_summary.json")


def _diagnostics(root):
    from .recommendation_diagnostics import load_recommendation_diagnostics
    path = root / "results/diagnostics/recommendation_diagnostics.json"
    if not path.is_file():
        st.info("The measured recommendation diagnostics are not present in this checkout.")
        return
    summary = load_recommendation_diagnostics(path)
    st.subheader("What changes when the history gets longer?")
    st.caption("A separate fixed seed42 diagnostic · identical test users and targets across every history cap")
    _kpis([("Users", f"{sum(d['selected_users'] for d in summary['domains']):,}", "1,000 deterministic users per domain"),
           ("Targets", f"{sum(d['selected_targets'] for d in summary['domains']):,}", "All eligible targets for those selected users"),
           ("History caps", "1 / 3 / 5 / 10 / 20", "Complete earlier-positive exclusions preserved"),
           ("Checkpoint seed", "42", "Original 27-run evidence remains separate")])
    domain = st.segmented_control("Diagnostic collection", ["baby", "sports", "clothing"], default="clothing", key="diagnostic_domain") or "clothing"
    metric = st.selectbox("Recommendation metric", ["ndcg@10", "recall@10", "ndcg@20", "recall@20"], key="diagnostic_metric")
    left, right = st.columns([1.3, 1])
    with left:
        fig = go.Figure()
        for mode in LABELS:
            rows = sorted([r for r in summary["history_curves"] if r["domain"] == domain and r["model"] == mode], key=lambda r: r["history_cap"])
            fig.add_scatter(x=[r["history_cap"] for r in rows], y=[r[metric] for r in rows], mode="lines+markers",
                            name=LABELS[mode], line={"color": COLORS[mode], "width": 3}, marker={"size": 9})
        fig.update_xaxes(title="Maximum visible positive history", type="category")
        fig.update_yaxes(title=metric.upper(), rangemode="tozero")
        _chart(fig)
    with right:
        st.markdown("#### Same cases, different input length")
        st.write("Only the visible model history is truncated. Every earlier positive item stays excluded from the candidate set, so changing the cap cannot make the task easier by re-admitting seen products.")
        st.write("Longer histories need not improve ranking. Some targets already have short histories; others can include less useful older preferences.")
        st.caption("User macro-average. Lines connect recorded measurements. This one-checkpoint subset cannot establish a training-seed effect.")
    st.markdown("#### Accuracy, catalog coverage and popularity exposure")
    exposure = [r for r in summary["exposure"] if r["domain"] == domain]
    cards = []
    for row in exposure:
        cards.append({"Model": LABELS[row["model"]], "Top-10 catalog coverage": row["coverage@10"],
                      "Exposure Gini": row["gini"], "Most-exposed 1% share": row["top_1pct_exposure_share"]})
    st.dataframe(pd.DataFrame(cards), hide_index=True, width="stretch",
                 column_config={"Top-10 catalog coverage": st.column_config.NumberColumn(format="percent"),
                                "Most-exposed 1% share": st.column_config.NumberColumn(format="percent")})
    fig = go.Figure()
    buckets = ["0", "1-5", "6-20", "21+"]
    fig.add_bar(name="Catalog share", x=buckets, y=[r["catalog_share"] for r in exposure[0]["popularity_exposure"]], marker_color="#cbd5e1")
    for row in exposure:
        fig.add_bar(name=LABELS[row["model"]], x=buckets, y=[r["exposure_share"] for r in row["popularity_exposure"]], marker_color=COLORS[row["model"]])
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Distinct positive TRAIN users per product")
    fig.update_yaxes(title="Share of catalog / recommendation slots", tickformat=".0%")
    _chart(fig)
    st.caption("Exposure counts Top-10 slots across test events at cap20. Catalog share includes zero-exposure products. Broader exposure alone does not establish better relevance, fairness or user satisfaction.")
    with st.expander("Popularity-cohort accuracy and paired history effects"):
        rows = [r for r in summary["popularity_cohorts"] if r["domain"] == domain]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        changes = [{"Model": LABELS[r["model"]], "Cap": r["history_cap"], **r["paired_ndcg_delta_vs_cap20"]}
                   for r in summary["history_curves"] if r["domain"] == domain]
        st.dataframe(pd.DataFrame(changes), hide_index=True, width="stretch")
        st.caption("Paired user-bootstrap intervals condition on the fixed checkpoint and cohort; they are not uncertainty over training seeds.")
    _limitations(summary)
    _download(summary, "shoplens_recommendation_diagnostics.json")


def _performance(root):
    path = root / "results/performance/studio.json"
    if not path.is_file():
        st.info("The measured Studio performance profile is not present in this checkout.")
        return
    summary = _load(path)
    if not summary.get("complete"):
        st.warning("Performance measurement is incomplete.")
        return
    st.subheader("Where does the local request spend its time?")
    st.caption("Real Qwen + CLIP + recommendation heads · local CPU · engine timing, separate from browser rendering")
    env, protocol = summary["environment"], summary["protocol"]
    _kpis([("Device", env["device"].upper(), f"{env['torch_threads']} Torch threads · {env['machine']}"),
           ("Catalog", f"{protocol['catalog_size']:,}", "Clothing · seed42 · three model groups"),
           ("Repeated measurements", str(protocol["repeats"]), "Descriptive p50 and p95; small local sample"),
           ("Ranking parity", f"{len(summary['ranking_parity'])} / {len(summary['ranking_parity'])}", "Complete rows identical to the public baseline")])
    rows = summary["stages"]
    stages = {r["stage"]: r for r in rows}
    labels = {"cold_catalog_validation": "First catalog validation", "cold_selected_exports": "First selected exports",
              "cold_three_query_encoders": "First Qwen + 2 adapters", "warm_unique_three_queries": "New text · 3 encoders",
              "exact_repeat_three_query_cache": "Same text · embedding cache", "cold_image_encoder": "First CLIP image",
              "warm_unique_image": "New image · CLIP", "exact_repeat_image_cache": "Same image · embedding cache",
              "warm_three_user_heads": "Warm · 3 user heads", "warm_unique_request_engine": "New text · full engine",
              "warm_exact_repeat_engine": "Exact repeat · full engine"}
    fig = go.Figure()
    for key, label in labels.items():
        if key not in stages:
            continue
        row = stages[key]
        fig.add_bar(x=[row["p50_ms"]], y=[label], orientation="h", marker_color="#6366f1" if "unique" in key else "#14b8a6" if "cache" in key else "#94a3b8",
                    customdata=[[row["p95_ms"], row["n"]]], showlegend=False,
                    hovertemplate="%{y}<br>p50 %{x:.3f} ms<br>p95 %{customdata[0]:.3f} ms<br>n = %{customdata[1]}<extra></extra>")
    fig.update_xaxes(title="Wall time in milliseconds · logarithmic axis", type="log")
    fig.update_yaxes(autorange="reversed")
    _chart(fig, height=520)
    st.caption("Cold observations include loading and validation in a fresh process, with an uncontrolled OS disk cache. Exact repeats reuse embeddings; new requests still execute the real model.")
    st.markdown("#### Process metadata only for products admitted to the retrieval pool")
    fig = go.Figure()
    names = {"text": "Text request", "multimodal_feedback": "Image + positive feedback", "avoid_budget": "Avoid terms + budget"}
    for version, color in (("before", "#94a3b8"), ("after", "#14b8a6")):
        selected = [stages[f"three_rankers_{case}_{version}"] for case in names]
        fig.add_bar(name="Original" if version == "before" else "Pool metadata optimization", x=list(names.values()),
                    y=[r["p50_ms"] for r in selected], marker_color=color,
                    customdata=[[r["p95_ms"], r["n"]] for r in selected],
                    hovertemplate="%{x}<br>p50 %{y:.2f} ms<br>p95 %{customdata[0]:.2f} ms<br>n=%{customdata[1]}<extra></extra>")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Three full-catalog rankers · milliseconds", rangemode="tozero")
    _chart(fig)
    st.write("Global cosine scores and rank percentiles are preserved. Only text matching, price parsing and category parsing move inside the already fixed retrieval pool. Before/after timings alternate order, and complete returned rows must match exactly.")
    with st.expander("All timing samples and environment"):
        st.dataframe(pd.DataFrame(rows).drop(columns="samples_ms"), hide_index=True, width="stretch")
        st.json(env)
    _limitations(summary)
    _download(summary, "shoplens_studio_performance.json")


def render_benchmarks(root, runtime_root=None):
    root = Path(root)
    runtime_root = Path(runtime_root) if runtime_root else root
    st.markdown("""<style>
    .bench-product{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:14px 16px;margin:10px 0;min-height:128px;box-shadow:0 2px 10px #0f172a04}
    .bench-product p{font-size:13px;line-height:1.45;margin:8px 0 0;color:#334155}
    </style>""", unsafe_allow_html=True)
    st.markdown("# Benchmark lab")
    st.markdown("Explore **query relevance**, **image + language retrieval**, **history and exposure**, and **local inference cost** using recorded measurements and real examples.")
    selection = st.radio("Benchmark", ["Search relevance", "Image + language", "Recommendation diagnostics", "Local performance"],
                          horizontal=True, key="benchmark_view")
    render = {"Search relevance": _search, "Image + language": lambda base: _fashion(base, runtime_root),
              "Recommendation diagnostics": _diagnostics, "Local performance": _performance}[selection]
    try:
        render(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"The recorded benchmark could not be read: {exc}")
        st.caption("Incomplete or incompatible records are not replaced with invented scores.")
