"""Local, evidence-first interface for the ShopLens coursework project.

The interface never substitutes generated recommendations or scores for missing
artifacts. Every recommendation calls the same inference API used by evaluation.
"""

from __future__ import annotations

import html
import json
import math
import os
from importlib import import_module
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


DOMAINS = {
    "baby": ("Baby", "Baby", "◉"),
    "sports": ("Sports & Outdoors", "Sports & Outdoors", "↗"),
    "clothing": ("Clothing & Accessories", "Clothing & Accessories", "◇"),
}
MODELS = {
    "qwen_frozen": "Frozen text · Qwen frozen",
    "qwen_semantic": "Semantic alignment · Qwen semantic LoRA",
    "qwen_preference": "Preference ranking · Qwen preference LoRA",
    "popularity": "Popular items · Popularity",
    "mf": "Behavioral · BPR + history fold-in",
    "tfidf": "Keywords · TF-IDF",
    "text": "Text semantics · MiniLM",
    "multimodal": "Multimodal · CLIP + MiniLM",
    "adapted": "Review adaptation · CLIP + adapted MiniLM",
}
MODEL_SHORT = {
    "qwen_frozen": "Qwen · Frozen", "qwen_semantic": "Qwen · Semantic", "qwen_preference": "Qwen · Preference",
    "popularity": "Popularity", "mf": "BPR + fold-in", "tfidf": "TF-IDF",
    "text": "MiniLM", "multimodal": "CLIP + MiniLM", "adapted": "CLIP + adapted MiniLM",
}
QWEN_MODES = ("qwen_frozen", "qwen_semantic", "qwen_preference")
OBJECTIVES = {
    "qwen_frozen": ("01", "Frozen reference", "Keep the text encoder frozen", "BPR trains only the ranking head. This is the reference for the three objectives.", "#546982"),
    "qwen_semantic": ("02", "Semantic alignment", "Align reviews with product text", "Multi-positive InfoNCE updates LoRA; a separate BPR gradient updates the ranking head.", "#347cdb"),
    "qwen_preference": ("03", "Preference ranking", "Learn preference order directly", "BPR updates candidate-text LoRA and the ranking head together to rank positives above negatives.", "#1a967b"),
}
DEMO_THEMES = {
    "baby": ("Baby on the go", "stroller"),
    "sports": ("Yoga & fitness", "yoga"),
    "clothing": ("Everyday running", "running"),
}
COLORS = ["#4c7bff", "#16a693", "#f49b45", "#9c6ade", "#df6787", "#60738e"]

CSS = """
<style>
:root { --sl-ink:#17243c; --sl-muted:#66758b; --sl-blue:#3868ec; --sl-line:#e6ebf2; }
html,body,[class*="css"],.stApp {font-family:'Avenir Next','PingFang SC','Microsoft YaHei',sans-serif;}
.stApp {background:#f6f8fc; color:var(--sl-ink);}
[data-testid="stHeader"] {background:transparent;}
.stDeployButton {display:none;}
.block-container {padding-top:2.0rem; padding-bottom:3rem; max-width:1480px;}
[data-testid="stSidebar"] {background:#101d34; color:#e6edfc;}
[data-testid="stSidebar"] * {color:inherit;}
[data-testid="stSidebar"] [data-baseweb="select"] * {color:#17243c;}
[data-testid="stSidebar"] input {color:#17243c;}
[data-testid="stSidebar"] .stCaption {color:#a7b6cf;}
[data-testid="stSidebar"] hr {border-color:#30405a;}
h1,h2,h3,h4 {letter-spacing:-.035em; color:#17243c;}
h1 {font-size:2.7rem !important; font-weight:700 !important;}
h2 {font-size:1.65rem !important;}
h3 {font-size:1.15rem !important;}
[data-testid="stMetric"] {background:white; padding:18px 20px; border:1px solid var(--sl-line); border-radius:16px;}
[data-testid="stMetricLabel"] {color:var(--sl-muted);}
[data-testid="stMetricValue"] {font-size:1.65rem; color:var(--sl-ink);}
.stTabs [data-baseweb="tab-list"] {gap:18px; border-bottom:1px solid var(--sl-line); margin-top:22px; margin-bottom:18px;}
.stTabs [data-baseweb="tab-list"] {flex-wrap:wrap; height:auto; row-gap:0;}
.stTabs [data-baseweb="tab-highlight"] {display:none;}
.stTabs [role="tablist"] {flex-wrap:wrap !important; height:auto !important; overflow:visible !important; gap:0 18px; margin-top:18px; margin-bottom:14px; border-bottom:1px solid var(--sl-line);}
.stTabs [role="tab"] {height:42px; flex-shrink:0; color:#66758b; font-size:14px; padding:0 2px;}
.stTabs [data-baseweb="tab"] {height:48px; font-size:14px; background:transparent; padding:0 2px;}
.stTabs [data-baseweb="tab"] {color:#66758b;}
.stTabs [aria-selected="true"] {font-weight:700; color:var(--sl-blue); border-bottom:2px solid var(--sl-blue);}
.stAlert [data-testid="stMarkdownContainer"] {color:#5a4c24;}
.stButton button {border-radius:10px; min-height:39px; font-weight:600;}
.stButton button[kind="primary"] {background:#3868ec; border-color:#3868ec;}
[data-testid="stVerticalBlockBorderWrapper"] {background:#fff; border-radius:16px;}
.sl-brand {font-size:27px; font-weight:700; letter-spacing:-1px; color:white; margin-bottom:4px;}
.sl-eyebrow {font-size:11px; font-weight:700; letter-spacing:1.8px; color:#7898e1; margin-bottom:12px;}
.sl-hero {background:linear-gradient(115deg,#e9efff 0%,#f6f9ff 66%,#e9f5f1 100%); border:1px solid #dfe8fa; border-radius:20px; padding:28px 32px; margin-bottom:22px;}
.sl-hero h1 {margin:0 0 8px; padding:0;}
.sl-title-phrase {display:inline-block;}
.sl-hero p {color:#576982; font-size:15px; max-width:800px; line-height:1.8; margin:0;}
.sl-tag {display:inline-block; font-size:11px; padding:5px 10px; border-radius:999px; background:white; color:#42638f; border:1px solid #d9e2f2; margin:0 6px 11px 0;}
.sl-section-note {color:#66758b; font-size:13px; margin-top:-5px; margin-bottom:16px;}
.sl-placeholder {height:145px; background:linear-gradient(145deg,#eff3f9,#e3eaf5); border-radius:11px; display:flex; flex-direction:column; justify-content:center; align-items:center; color:#8393ae; margin-bottom:12px;}
.sl-placeholder span {font-size:42px; line-height:1.3; color:#a3b2cd;}
.sl-placeholder small {font-size:11px; margin-top:6px;}
.sl-product-title {font-size:13px; font-weight:600; line-height:1.55; height:42px; overflow:hidden; margin:3px 0 7px; color:#24334d;}
.sl-id {color:#8995a6; font-size:10px; letter-spacing:.5px; margin-bottom:8px;}
[data-testid="stImage"] img {height:145px; object-fit:contain; border-radius:11px; background:#f7f9fc;}
.sl-rank {font-size:11px; color:#3a69d6; font-weight:700; margin-bottom:7px;}
.sl-pill {background:#edf3ff; border-radius:7px; padding:4px 8px; display:inline-block; color:#4063a4; font-size:11px; margin-right:4px; margin-bottom:4px;}
.sl-footer {font-size:11px; color:#8794a7; border-top:1px solid #e2e8f2; padding-top:16px; margin-top:30px;}
.sl-empty {background:white; padding:35px; border:1px dashed #cfd9e8; border-radius:16px; color:#62718a; line-height:1.9; margin:12px 0;}
.sl-legend {font-size:12px; color:#607089; margin:7px 0 12px;}
.sl-objective {background:#fff; border:1px solid #e0e8f1; border-top:3px solid var(--accent); border-radius:14px; padding:17px 19px; min-height:176px;}
.sl-objective-number {font-size:11px; font-weight:700; color:var(--accent); letter-spacing:1.2px;}
.sl-objective h3 {font-size:17px !important; margin:7px 0 3px;}
.sl-objective p {font-size:12px; color:#66758b; line-height:1.65; margin:7px 0;}
.sl-status {font-size:10px; padding:3px 7px; border-radius:5px; background:#eef3f8; color:#546982; display:inline-block;}
.sl-status-ready {background:#e5f4ef; color:#187c64;}
.sl-history-label {font-size:11px; color:#66758b; margin:0 0 8px;}
.sl-delta {font-size:11px; color:#168269; font-weight:600; margin-bottom:6px;}
.sl-pipeline {background:#edf3ff; border-radius:12px; padding:14px 18px; color:#426087; font-size:12px; line-height:1.8; margin:16px 0;}
[class*="st-key-shoplens_history_"] [data-testid="stImage"] img {height:74px;}
[class*="st-key-shoplens_history_"] .sl-placeholder {height:74px !important; margin-bottom:4px;}
[class*="st-key-shoplens_history_"] .sl-placeholder span {font-size:22px;}
[class*="st-key-shoplens_history_"] .sl-placeholder small {display:none;}
[class*="st-key-shoplens_history_"] .stButton button {font-size:11px; min-height:28px; padding:2px 5px;}
[class*="st-key-shoplens_compare_"] [data-testid="stImage"] img {height:155px;}
@media(max-width:1100px) {
 [data-testid="stSidebar"][aria-expanded="true"] {max-width:220px;}
 .block-container {padding:1rem 1.2rem 2rem;}
 .sl-hero {padding:22px;}
 .sl-hero h1 {font-size:clamp(1.65rem,4.1vw,2.2rem) !important;}
 .st-key-shoplens_metrics [data-testid="stHorizontalBlock"],
 [class*="st-key-shoplens_product_grid_"] [data-testid="stHorizontalBlock"] {flex-wrap:wrap !important; gap:1rem;}
 .st-key-shoplens_metrics [data-testid="stColumn"],
 [class*="st-key-shoplens_product_grid_"] [data-testid="stColumn"] {flex:1 1 calc(50% - .5rem) !important; width:calc(50% - .5rem) !important; min-width:0 !important;}
 [data-testid="stMetric"] {padding:14px;}
 [data-testid="stMetricValue"] {font-size:1.45rem;}
 [class*="st-key-shoplens_objectives_"] [data-testid="stHorizontalBlock"],
 [class*="st-key-shoplens_compare_"] [data-testid="stHorizontalBlock"],
 [class*="st-key-shoplens_atlas_"] [data-testid="stHorizontalBlock"] {flex-wrap:wrap !important;}
 [class*="st-key-shoplens_objectives_"] [data-testid="stColumn"],
 [class*="st-key-shoplens_compare_"] [data-testid="stColumn"],
 [class*="st-key-shoplens_atlas_"] [data-testid="stColumn"] {flex:1 1 100% !important; width:100% !important; min-width:0 !important;}
 .sl-objective {min-height:0;}
 [class*="st-key-shoplens_history_"] [data-testid="stHorizontalBlock"] {flex-wrap:wrap !important;}
 [class*="st-key-shoplens_history_"] [data-testid="stColumn"] {flex:1 1 calc(25% - 1rem) !important; min-width:80px !important;}
}
@media(max-width:430px) {.block-container {padding-left:.75rem; padding-right:.75rem;} .sl-hero {padding:18px;}}
</style>
"""


def _root() -> Path:
    return Path(os.environ.get("SHOPLENS_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()


def _model_source(artifacts: Path, model: str) -> tuple[str, Path]:
    """Keep new two-tower artifacts out of the legacy scorer and cache namespace."""
    if model in QWEN_MODES:
        return "shoplens.qwen_inference", artifacts.parent / "qwen" / artifacts.name / model / "seed42"
    if model not in MODELS:
        raise ValueError(f"Unknown UI model: {model}")
    return "shoplens.inference", artifacts


def _available_models(artifacts: Path) -> list[str]:
    legacy = import_module("shoplens.inference").available_models(artifacts)
    ready = [model for model in legacy if model in MODELS and model not in QWEN_MODES]
    try:
        backend = import_module("shoplens.qwen_inference")
    except ModuleNotFoundError as exc:
        if exc.name != "shoplens.qwen_inference":
            raise
    else:
        for model in QWEN_MODES:
            _, directory = _model_source(artifacts, model)
            if model in backend.available_models(directory):
                ready.append(model)
    return [model for model in MODELS if model in ready]


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {"records": value}
    except (OSError, ValueError):
        return {}


def _signature(*paths: Path) -> tuple[tuple[str, int, int], ...]:
    """Invalidate cached inference after model, feature or processed-data updates."""
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl", ".npz", ".pt", ".pkl", ".joblib"})
    signature = []
    for path in sorted(set(files)):
        try:
            stat = path.stat()
            signature.append((str(path), stat.st_mtime_ns, stat.st_size))
        except OSError:
            continue
    return tuple(signature)


@st.cache_data(show_spinner=False, max_entries=12)
def _items(processed: str, signature: tuple) -> list[dict]:
    del signature
    from shoplens.inference import load_items
    return load_items(Path(processed))


@st.cache_data(show_spinner=False, max_entries=256)
def _recommend(processed: str, artifacts: str, model: str, liked: tuple[str, ...], k: int, signature: tuple) -> list[dict]:
    del signature
    backend, directory = _model_source(Path(artifacts), model)
    return import_module(backend).recommend(Path(processed), directory, model, list(liked), k=k)


@st.cache_data(show_spinner=False, max_entries=256)
def _explain(processed: str, artifacts: str, model: str, liked: tuple[str, ...], item_id: str, signature: tuple) -> dict:
    del signature
    backend, directory = _model_source(Path(artifacts), model)
    return import_module(backend).explain(Path(processed), directory, model, list(liked), item_id)


def _synthetic(*documents: dict) -> bool:
    for doc in documents:
        if doc.get("synthetic") is True or doc.get("is_synthetic") is True or doc.get("fixture") is True:
            return True
        values = [str(doc.get(k, "")).lower() for k in ("status", "source", "data_source", "data_kind", "feature_type")]
        if any("synthetic" in v or "fixture" in v for v in values):
            return True
    return False


def _is_subset(audit: dict) -> bool:
    subset = audit.get("subset")
    if isinstance(subset, dict):
        subset = any(subset.get(key) is not None for key in ("max_users", "max_reviews", "max_items"))
    return bool(subset or audit.get("is_subset") or audit.get("full") is False or "subset" in str(audit.get("scope", "")).lower())


def _local_image(root: Path, item: dict) -> Path | None:
    """Only render image paths inside the project; metadata is untrusted input."""
    raw = item.get("image_path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = Path(raw).expanduser()
    path = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        return None
    return path if path.is_file() else None


def _picture(root: Path, item: dict, symbol: str, height: int = 145) -> None:
    path = _local_image(root, item)
    if path is not None:
        st.image(str(path), width="stretch")
    else:
        st.markdown(f'<div class="sl-placeholder" style="height:{height}px"><span>{html.escape(symbol)}</span><small>Image not cached locally</small></div>', unsafe_allow_html=True)


def _title(item: dict) -> str:
    return str(item.get("title") or item.get("item_id") or "Untitled product")


def _card(root: Path, item: dict, symbol: str, liked: list[str], key: str, rank: int | None = None, score: float | None = None) -> None:
    item_id = str(item["item_id"])
    with st.container(border=True):
        if rank is not None:
            st.markdown(f'<div class="sl-rank">TOP {rank:02d}</div>', unsafe_allow_html=True)
        _picture(root, item, symbol)
        st.markdown(f'<div class="sl-product-title" title="{html.escape(_title(item), quote=True)}">{html.escape(_title(item))}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sl-id">{html.escape(item_id)}</div>', unsafe_allow_html=True)
        if score is not None and math.isfinite(float(score)):
            st.caption(f"Model score {float(score):.4f}")
        selected = item_id in liked
        if st.button("♥ Selected" if selected else "♡ Add to preferences", key=key, type="primary" if selected else "secondary", width="stretch"):
            if selected:
                liked.remove(item_id)
            else:
                liked.append(item_id)
            st.rerun()


def _cards(root: Path, records: list[dict], catalog: dict[str, dict], symbol: str, liked: list[str], prefix: str, columns: int = 4) -> None:
    with st.container(key=f"shoplens_product_grid_{prefix}"):
        for offset in range(0, len(records), columns):
            row = st.columns(columns)
            for i, record in enumerate(records[offset:offset + columns]):
                item_id = str(record["item_id"])
                with row[i]:
                    _card(root, catalog.get(item_id, {"item_id": item_id}), symbol, liked, f"{prefix}_{offset+i}_{item_id}", record.get("rank"), record.get("score"))


def _show_error(exc: Exception) -> None:
    st.error("Inference could not use the current artifacts. Check that the model, features and product catalog come from the same processing run.")
    with st.expander("View diagnostics"):
        st.code(f"{type(exc).__name__}: {exc}", language="text")


def _objectives(domain: str, models: list[str], prefix: str) -> None:
    with st.container(key=f"shoplens_objectives_{prefix}_{domain}"):
        columns = st.columns(3)
        for column, model in zip(columns, QWEN_MODES):
            number, english, title, description, color = OBJECTIVES[model]
            ready = model in models
            status = "seed 42 · Ready to explore" if ready else "Artifacts pending"
            with column:
                st.markdown(f'<div class="sl-objective" style="--accent:{color}"><div class="sl-objective-number">{number} / {english.upper()}</div><h3>{title}</h3><p>{description}</p><span class="sl-status{ " sl-status-ready" if ready else ""}">{status}</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sl-pipeline"><b>Shared inputs</b> Same history → base Qwen text + CLIP images → history projection and user network<br><b>Candidate-text objectives</b> Frozen / semantic alignment / preference ranking → candidate vectors → full-catalog ranking<br>Base text and image input features stay frozen; history and candidate projections and the user network train as part of the ranking head. All three objectives share its initialization and adaptation examples. The interactive demo uses seed 42; formal evaluation repeats the full pipeline with three seeds.</div>', unsafe_allow_html=True)


def _history(root: Path, domain: str, catalog: dict[str, dict], liked: list[str], prefix: str) -> None:
    if not liked:
        return
    st.markdown(f'<div class="sl-history-label">CURRENT PREFERENCE · {len(liked)} selected · The latest 20 enter the encoder; all selected products are excluded from candidates</div>', unsafe_allow_html=True)
    # Compact visible strip; the complete history remains available below.
    with st.container(key=f"shoplens_history_{prefix}_{domain}"):
        visible = liked[-6:]
        columns = st.columns(len(visible))
        for column, item_id in zip(columns, visible):
            with column:
                item = catalog.get(item_id, {"item_id": item_id})
                _picture(root, item, DOMAINS[domain][2], 74)
                if st.button("× " + _title(item)[:17], help=_title(item), key=f"history_remove_{prefix}_{domain}_{item_id}", width="stretch"):
                    liked.remove(item_id)
                    st.rerun()
    if len(liked) > 6:
        with st.expander(f"View all {len(liked)} preferences"):
            st.write(" · ".join(_title(catalog[item_id]) for item_id in liked if item_id in catalog))


def _rank_delta(before: int | None, after: int | None) -> str:
    if before is None and after is not None:
        return "Entered Top-K"
    if before is not None and after is None:
        return "Left Top-K"
    if before is None or after is None:
        return "—"
    delta = before - after
    return f"↑ {delta}" if delta > 0 else f"↓ {abs(delta)}" if delta < 0 else "= Unchanged"


def _demo_history(root: Path, domain: str, items: list[dict]) -> list[str]:
    """A clearly labeled display preference, selected from real catalog titles."""
    keyword = DEMO_THEMES[domain][1]
    matching = [item for item in items if keyword in _title(item).lower() and _local_image(root, item) is not None]
    # Fixed catalog order keeps the classroom demonstration reproducible.
    return [str(item["item_id"]) for item in matching[:3]] if len(matching) >= 3 else []


def _sidebar(root: Path) -> str:
    with st.sidebar:
        st.markdown('<div class="sl-brand">ShopLens<span style="color:#7d9fff">.</span></div>', unsafe_allow_html=True)
        st.caption("LANGUAGE × VISION × PREFERENCE")
        st.radio("Workspace", ["Shopping Studio", "Research lab"], key="shoplens_workspace")
        st.divider()
        st.markdown("**Product collection**")
        domain = st.selectbox("Select dataset", list(DOMAINS), format_func=lambda d: f"{DOMAINS[d][2]}  {DOMAINS[d][1]}", label_visibility="collapsed")
        st.caption(DOMAINS[domain][0])
        st.divider()
        st.markdown("**How to explore**")
        if st.session_state.get("shoplens_workspace") == "Shopping Studio":
            st.markdown("① Describe what you need\n\n② Add optional visual inspiration\n\n③ Refine with feedback\n\n④ Compare and save your shortlist")
        else:
            st.markdown("① Choose 3–5 products you like\n\n② Explore recommendations on the map\n\n③ Compare ranks and preference links\n\n④ Check the measured results")
        st.caption("Qwen3-Embedding · LoRA\n\nFrozen CLIP · 6 classic baselines")
        st.divider()
        st.caption("Runs locally · Preferences stay in this session")
        with st.expander("Project & artifact paths"):
            st.code(str(root), language="text")
            st.caption("Set SHOPLENS_ROOT to choose the project directory. Each view reads the prepared artifacts for the selected collection.")
    return domain


def _empty_state(root: Path, domain: str, missing: str) -> None:
    st.markdown(f'<div class="sl-empty"><b>{html.escape(missing)}</b><br>Prepare real data, features and trained artifacts to explore the product catalog and recommendations here.<br>No recommendation results or performance metrics are available yet.</div>', unsafe_allow_html=True)
    st.markdown("**Setup sequence**")
    st.markdown("1. Prepare real product text, image references and chronological interactions.\n2. Extract frozen CLIP and text features.\n3. Train and formally evaluate the three Qwen objectives.\n4. Use the completed seed 42 exports for local interaction.")
    st.caption(f"Collection: {DOMAINS[domain][0]} · Data directory: {root / 'data' / 'processed' / domain}")
    st.info("See the project README for commands and parameters. Missing artifacts do not produce placeholder scores.")


def _catalog_tab(root: Path, domain: str, items: list[dict], catalog: dict[str, dict], models: list[str], processed: Path, artifacts: Path, signature: tuple, liked: list[str]) -> None:
    title_cols = st.columns([3, 1])
    title_cols[0].subheader("Start with what you like")
    if title_cols[1].button("Clear preferences", width="stretch", disabled=not liked, key=f"clear_{domain}"):
        liked.clear()
        st.rerun()
    st.markdown('<div class="sl-section-note">Choose products you like to rerank recommendations from the same history. Start with 3–5 items; the encoder uses the latest 20.</div>', unsafe_allow_html=True)
    if not liked:
        demo = _demo_history(root, domain, items)
        if demo:
            if st.button(f"Start with 3 real products · {DEMO_THEMES[domain][0]}", key=f"demo_history_{domain}"):
                liked[:] = demo
                st.rerun()
            st.caption("Optional demo preferences: three real products with a shared theme, selected in fixed catalog order. These are display inputs, not observed user behavior or an evaluation case; the current model computes their recommendations live.")
    _history(root, domain, catalog, liked, "catalog")
    filter_cols = st.columns([3, 1])
    query = filter_cols[0].text_input("Search products", placeholder="Search by title, description or ASIN…", key=f"catalog_query_{domain}")
    page_size = filter_cols[1].selectbox("Items per page", [8, 12, 20], index=1, key=f"page_size_{domain}")
    query_words = query.strip().lower().split()
    filtered = [item for item in items if all(word in (str(item.get("title", "")) + " " + str(item.get("description", "")) + " " + str(item["item_id"])).lower() for word in query_words)]
    n_pages = max(1, math.ceil(len(filtered) / page_size))
    page = int(st.number_input("Catalog page", min_value=1, max_value=n_pages, value=1, step=1, key=f"catalog_page_{domain}_{query}_{page_size}"))
    st.caption(f"{len(filtered):,} matching products · Page {page} of {n_pages} · Fixed order by product ID")
    if filtered:
        _cards(root, filtered[(page-1)*page_size:page*page_size], catalog, DOMAINS[domain][2], liked, f"catalog_{domain}")
    else:
        st.info("No matching products. Try a shorter keyword or product ID.")
    st.divider()
    st.subheader("Recommended for your preferences")
    if not models:
        st.info("The catalog is ready. Train a recommendation model to enable recommendations.")
        return
    selector_cols = st.columns([3, 1])
    default_model = next((m for m in (*QWEN_MODES, "multimodal", "adapted", "text", "tfidf", "popularity", "mf") if m in models), models[0])
    model = selector_cols[0].selectbox("Recommendation method", models, index=models.index(default_model), format_func=lambda m: MODELS.get(m, m), key=f"recommend_model_{domain}")
    k = selector_cols[1].selectbox("Number of recommendations", [4, 8, 12, 20], index=1, key=f"recommend_k_{domain}")
    if model == "popularity":
        st.caption("Popularity does not adapt to preferences. Selections only exclude already selected products.")
    elif model == "mf":
        st.caption("The behavioral baseline estimates a user vector from the current history with ridge fold-in, then ranks the trained item factors.")
    elif model in QWEN_MODES:
        st.caption(f"{OBJECTIVES[model][1]} · Candidate text: {OBJECTIVES[model][2]}; history inputs: base Qwen + frozen CLIP. The history projection and user network train as part of the ranking head. Local inference loads lightweight exported vectors.")
    if not liked:
        st.info("Add a few products above to establish preferences for this session.")
        return
    try:
        with st.spinner("Computing recommendations with the trained model…"):
            recommendations = _recommend(str(processed), str(artifacts), model, tuple(liked), int(k), signature)
        if recommendations:
            st.caption("Scores rank products within this model; raw scores across models are not directly comparable. Selected products are excluded.")
            _cards(root, recommendations, catalog, DOMAINS[domain][2], liked, f"rec_{domain}_{model}")
        else:
            st.info("No products remain in the eligible candidate set.")
    except Exception as exc:
        _show_error(exc)


def _compare_tab(root: Path, domain: str, catalog: dict[str, dict], models: list[str], processed: Path, artifacts: Path, signature: tuple, liked: list[str]) -> None:
    st.subheader("One history. Three training objectives.")
    st.caption("How should text adapt? Compare actual recommendations for the same history and catalog, then use formal test results to assess overall performance.")
    _objectives(domain, models, "compare")
    if len(models) < 2:
        st.info("Prepare at least two models to compare recommendations side by side.")
        return
    if not liked:
        st.info("Choose products in Preferences first.")
        return
    _history(root, domain, catalog, liked, "compare")
    qwen_ready = all(m in models for m in QWEN_MODES)
    defaults = list(QWEN_MODES) if qwen_ready else [m for m in ("text", "multimodal", "adapted") if m in models][:3] or models[:2]
    selected = st.multiselect("Models to compare", models, default=defaults, format_func=lambda m: MODELS.get(m, m), max_selections=3, key=f"compare_models_{domain}_{'qwen' if qwen_ready else 'baseline'}")
    if len(selected) < 2:
        st.info("Choose 2–3 models to compare.")
        return
    k = st.select_slider("Compare Top-K products", options=[5, 10, 20], value=10, key=f"compare_k_{domain}")
    results: dict[str, list[dict]] = {}
    try:
        with st.spinner("Comparing models for the same preferences…"):
            for model in selected:
                results[model] = _recommend(str(processed), str(artifacts), model, tuple(liked), k, signature)
    except Exception as exc:
        _show_error(exc)
        return
    reference = "qwen_frozen" if "qwen_frozen" in selected else selected[0]
    ranks = {model: {str(record["item_id"]): int(record.get("rank", i+1)) for i, record in enumerate(records)} for model, records in results.items()}
    from shoplens.visual_charts import rank_flow_figure, overlap_figure
    flow, overlap = st.tabs(["Rank flow", "Recommendation overlap"])
    colors = {model: OBJECTIVES[model][-1] for model in QWEN_MODES}
    with flow:
        st.plotly_chart(rank_flow_figure(results, catalog, MODEL_SHORT, colors), width="stretch", key=f"rank_flow_{domain}")
    with overlap:
        st.plotly_chart(overlap_figure(results, MODEL_SHORT, colors), width="stretch", key=f"rank_overlap_{domain}")
    st.markdown("### Browse recommended products")
    with st.container(key=f"shoplens_compare_{domain}"):
        columns = st.columns(len(selected))
        for col, model in zip(columns, selected):
            with col:
                st.markdown(f"### {MODEL_SHORT.get(model, model)}")
                if model in OBJECTIVES:
                    st.caption(OBJECTIVES[model][2])
                overlap = len(set(ranks[model]) & set(ranks[reference]))
                st.caption("Ranking reference" if model == reference else f"Shares {overlap}/{len(ranks[model])} Top-K products with the reference")
                for index, record in enumerate(results[model][:3]):
                    item_id = str(record["item_id"])
                    if model != reference:
                        delta = _rank_delta(ranks[reference].get(item_id), ranks[model].get(item_id))
                        st.markdown(f'<div class="sl-delta">vs. reference: {html.escape(delta)}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="sl-delta">Reference rank</div>', unsafe_allow_html=True)
                    _card(root, catalog.get(item_id, {"item_id": item_id}), DOMAINS[domain][2], liked, f"compare_card_{domain}_{model}_{index}_{item_id}", ranks[model][item_id])
                with st.expander(f"View all Top-{k}"):
                    frame = pd.DataFrame([{"Rank": r.get("rank", i+1), "Product": _title(catalog.get(str(r["item_id"]), r))[:65], "ASIN": str(r["item_id"])} for i, r in enumerate(results[model])])
                    st.dataframe(frame, hide_index=True, width="stretch")
    st.markdown(f"**Full rank changes · Reference: {MODEL_SHORT.get(reference, reference)}**")
    item_order = list(dict.fromkeys(str(r["item_id"]) for model in selected for r in results[model]))
    rows = []
    for item_id in item_order:
        row = {"Product": _title(catalog.get(item_id, {"item_id": item_id}))[:65], "ASIN": item_id}
        for model in selected:
            row[MODEL_SHORT.get(model, model)] = ranks[model].get(item_id)
            if model != reference:
                row[f"Δ {MODEL_SHORT.get(model, model)}"] = _rank_delta(ranks[reference].get(item_id), ranks[model].get(item_id))
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("↑ indicates a rise from the reference. Missing Top-K positions stay blank; no full rank is inferred. Raw scores across models are not compared.")


def _explain_tab(root: Path, domain: str, catalog: dict[str, dict], models: list[str], processed: Path, artifacts: Path, signature: tuple, liked: list[str]) -> None:
    st.subheader("Inspect the recommendation evidence")
    st.caption("Inspect history products, original text and similarity evidence returned by the model. Similarity supports inspection, not causal attribution.")
    if not models or not liked:
        st.info("Choose products you like and prepare at least one recommendation model.")
        return
    model = st.selectbox("Model to inspect", models, format_func=lambda m: MODELS.get(m, m), key=f"explain_model_{domain}")
    if model in QWEN_MODES:
        st.caption("Evidence uses the independent Qwen artifacts. History inputs retain base Qwen and frozen CLIP features; the history projection and user network train as part of the ranking head. Candidates use the selected checkpoint for this objective. Adapted candidate vectors are not reused as history inputs.")
    try:
        recommendations = _recommend(str(processed), str(artifacts), model, tuple(liked), 20, signature)
        if not recommendations:
            st.info("No recommendations are available to inspect.")
            return
        item_id = st.selectbox("Select a recommended product", [str(r["item_id"]) for r in recommendations], format_func=lambda i: _title(catalog.get(i, {"item_id": i})), key=f"explain_item_{domain}_{model}")
        item = catalog.get(item_id, {"item_id": item_id})
        left, right = st.columns([1, 3])
        with left:
            _picture(root, item, DOMAINS[domain][2], 190)
        with right:
            st.markdown(f"**{_title(item)}**")
            st.caption(item_id)
            text = str(item.get("description") or item.get("text") or "No original description is available.")
            st.write(text[:1800])
            if len(text) > 1800:
                with st.expander("Read the full product text"):
                    st.write(text)
        evidence = _explain(str(processed), str(artifacts), model, tuple(liked), item_id, signature)
        notes = evidence.get("note") or evidence.get("notes") or evidence.get("method")
        if notes:
            st.info(str(notes))
        history = evidence.get("history_evidence") or evidence.get("evidence") or evidence.get("history_items")
        if isinstance(history, list) and history:
            records = []
            for record in history:
                if isinstance(record, dict):
                    entry = dict(record)
                    evidence_id = str(entry.get("item_id", ""))
                    if evidence_id in catalog:
                        entry.setdefault("title", _title(catalog[evidence_id]))
                    records.append(entry)
            if records:
                st.markdown("**History evidence returned by the model**")
                st.dataframe(pd.DataFrame(records), hide_index=True, width="stretch")
                channels = [channel for channel in ("text_similarity", "image_similarity") if any(channel in row for row in records)]
                if channels:
                    plot_rows = []
                    for record in records[:10]:
                        for channel in channels:
                            if channel in record:
                                plot_rows.append({"History product": str(record.get("title") or record.get("item_id"))[:35], "Modality": "Text" if channel == "text_similarity" else "Image", "Cosine similarity": record[channel]})
                    figure = px.bar(pd.DataFrame(plot_rows), x="Cosine similarity", y="History product", color="Modality", barmode="group", orientation="h", color_discrete_sequence=COLORS, template="plotly_white")
                    figure.update_layout(height=max(260, 42*len(records[:10])), margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title_text="")
                    st.plotly_chart(figure, width="stretch", key=f"evidence_chart_{domain}_{model}")
                    st.caption("These are cosine similarities of the original modality features, not modality weights learned by the fusion network.")
                elif model in QWEN_MODES and any("similarity" in row for row in records):
                    plot_rows = [{"History product": str(row.get("title") or row.get("item_id"))[:35], "Cross-tower cosine": row["similarity"]} for row in records[:10] if isinstance(row.get("similarity"), (int, float))]
                    if plot_rows:
                        figure = px.bar(pd.DataFrame(plot_rows), x="Cross-tower cosine", y="History product", orientation="h", template="plotly_white", color_discrete_sequence=["#347cdb"])
                        figure.update_layout(height=max(250, 40*len(plot_rows)), margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(figure, width="stretch", key=f"evidence_chart_{domain}_{model}")
                        st.caption("This is the cosine between the history-tower output from frozen inputs and the selected candidate-tower vector. It is not a text/image weight decomposition or a causal explanation of the final user-network score.")
        st.markdown("**Original products in your preferences**")
        for history_id in liked[-20:]:
            history_item = catalog.get(history_id, {"item_id": history_id})
            with st.expander(_title(history_item)):
                st.caption(history_id)
                st.write(str(history_item.get("description") or history_item.get("text") or "No description is available.")[:2500])
        with st.expander("Full inference evidence / JSON"):
            st.json(evidence)
    except Exception as exc:
        _show_error(exc)


def _metric_rows(document: dict) -> list[dict]:
    rows = document.get("results", document.get("records", []))
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and isinstance(row.get("model"), str)]


def _metric_compatibility(rows: list[dict]) -> bool:
    comparison_keys = ("items_sha256", "events_sha256", "n_users", "n_targets")
    protocol_keys = ("candidate_set", "history", "averaging", "n_catalog_items", "positive_feedback", "max_history")
    fingerprints = {json.dumps({**{key: row.get(key) for key in comparison_keys}, "protocol": {key: row.get("run_protocol", {}).get(key) for key in protocol_keys}}, sort_keys=True) for row in rows}
    if len(fingerprints) > 1:
        return False
    qwen_rows = [row for row in rows if row["model"] in QWEN_MODES]
    shared_keys = ("model_id", "revision", "encoding_config", "epochs", "max_examples", "lr", "weight_decay")
    qwen_inputs = {json.dumps({"base_features_sha256": row.get("featurehash"),
                              **{key: row.get("training_configuration", {}).get(key, row.get(key)) for key in shared_keys},
                              "history_encoder": row.get("run_protocol", {}).get("history_encoder")}, sort_keys=True) for row in qwen_rows}
    if len(qwen_inputs) > 1:
        return False
    for model in {row["model"] for row in rows}:
        model_rows = [row for row in rows if row["model"] == model]
        seeds = [row.get("seed") for row in model_rows]
        if len(seeds) != len(set(seeds)):
            return False
        if model not in QWEN_MODES:
            if len({row.get("selected_features_sha256") for row in model_rows}) > 1:
                return False
        else:
            # Each seed retrains LoRA; candidate/cache hashes are expected to vary.
            # Compare the declared shared protocol, not the selected learned weights.
            keys = ("model_id", "revision", "encoding_config", "lora_config", "epochs", "max_examples", "lr", "weight_decay", "objective")
            configs = {json.dumps({key: row.get("training_configuration", {}).get(key, row.get(key)) for key in keys}, sort_keys=True) for row in model_rows}
            if len(configs) > 1:
                return False
    return True


def _training_traces(document: dict, prefix: str = "") -> list[dict]:
    """Extract recorded epoch scalars only; never interpolate missing losses."""
    traces = []
    for key, value in document.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            traces.extend(_training_traces(value, name))
        elif isinstance(value, list):
            for record in value:
                if not isinstance(record, dict) or not isinstance(record.get("epoch"), (int, float)):
                    continue
                for metric, number in record.items():
                    if metric == "epoch" or not any(token in metric.lower() for token in ("loss", "ndcg", "recall")):
                        continue
                    if isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(number):
                        traces.append({"epoch": record["epoch"], "trace": f"{name}/{metric}", "value": number})
    return traces


def _training_panel(artifacts: Path) -> None:
    st.markdown("**Recorded training evidence**")
    st.caption("Only scalars recorded for seed 42 are shown. Lower semantic-alignment loss does not establish better recommendation metrics. Each objective selects adaptation epoch 1 or 2 using downstream validation NDCG@10.")
    found = False
    for model in QWEN_MODES:
        _, directory = _model_source(artifacts, model)
        training = _json(directory / "training.json")
        metadata = _json(directory / "models" / f"{model}.json")
        if not training and not metadata:
            continue
        found = True
        with st.expander(f"{MODEL_SHORT[model]} · seed 42"):
            selected = metadata.get("selected_epoch", training.get("selected_epoch"))
            samples = metadata.get("n_training_examples", training.get("n_training_examples"))
            st.caption(f"Selected adaptation epoch: {selected if selected is not None else 'not recorded'} · Actual adaptation examples: {samples if samples is not None else 'not recorded'}")
            traces = _training_traces(training)
            if model != "qwen_semantic":
                traces = [row for row in traces if not row["trace"].endswith("semantic_loss")]
            if traces:
                trace_frame = pd.DataFrame(traces)
                names = list(trace_frame["trace"].unique())
                preferred = "adaptation_curve/validation_ndcg@10"
                name = st.selectbox("Recorded training metric", names, index=names.index(preferred) if preferred in names else 0, key=f"training_trace_{artifacts.name}_{model}")
                figure = px.line(trace_frame[trace_frame["trace"] == name], x="epoch", y="value", markers=True, template="plotly_white")
                figure.update_layout(height=245, margin=dict(l=5, r=10, t=10, b=10), xaxis_title="Recorded epoch", yaxis_title=name.rsplit("/", 1)[-1], paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(figure, width="stretch", key=f"training_chart_{artifacts.name}_{model}")
            elif training:
                st.caption("This record has no plottable epoch scalars. The original training evidence is shown below.")
            st.json(training or {"status": "Training records are not available yet"})
    if not found:
        st.info("Qwen training records are not ready. Recorded loss, validation traces and selected checkpoints will appear when available. No training curves are prefilled.")


def _metrics_tab(root: Path, domain: str, audit: dict, features: dict, synthetic: bool) -> None:
    st.subheader("From individual recommendations to full experiments")
    st.caption("Charts read local evaluation outputs. Interpret results separately when splits, data sizes or candidate sets differ.")
    artifact_dir = root / "artifacts" / domain
    legacy_document = _json(artifact_dir / "results.json")
    qwen_document = _json(root / "artifacts" / "qwen" / domain / "results.json")
    result_document = {"legacy": legacy_document, "qwen": qwen_document}
    legacy_rows = [row for row in _metric_rows(legacy_document) if row["model"] in MODELS and row["model"] not in QWEN_MODES]
    qwen_rows = [row for row in _metric_rows(qwen_document) if row["model"] in QWEN_MODES]
    family = st.radio("Experiment family", ["Three Qwen objectives", "Six original baselines", "All methods"], index=0 if qwen_rows else 1, horizontal=True, key=f"metric_family_{domain}")
    all_rows = qwen_rows if family == "Three Qwen objectives" else legacy_rows if family == "Six original baselines" else legacy_rows + qwen_rows
    if family == "Three Qwen objectives":
        st.caption("Qwen3-Embedding-0.6B + frozen CLIP · Three full-pipeline seeds per objective · Shared warmup initialization, 2 adaptation epochs per run, and up to 30,000 matched training examples. This cap is not the full dataset size.")
    elif family == "Six original baselines":
        st.caption("The original main experiment uses a fixed 5-epoch budget. BPR + history fold-in also has a separate 20-epoch budget check. The original results are preserved; the main budget does not establish full tuning.")
    else:
        st.caption("The two experiment families use different text encoders, objectives and budgets. This chart compares values under a shared evaluation protocol; cross-family differences cannot be attributed to the LoRA objective.")
    diagnostics = [row for row in all_rows if row.get("run_protocol", {}).get("evaluation_subset")]
    rows = [row for row in all_rows if not row.get("run_protocol", {}).get("evaluation_subset")]
    if diagnostics:
        with st.expander(f"Diagnostic target-subset records ({len(diagnostics)}; excluded from formal means)"):
            st.json(diagnostics)
    if not rows:
        st.info("No formal evaluation results are available for this collection yet. Run evaluation after training; results.json will appear here.")
    else:
        frame = pd.DataFrame(rows)
        splits = sorted(str(v) for v in frame.get("split", pd.Series(["unspecified"])).dropna().unique())
        if "split" not in frame:
            frame["split"] = "unspecified"
        split = st.selectbox("Evaluation split", splits, index=splits.index("test") if "test" in splits else 0, key=f"metric_split_{domain}")
        frame = frame[frame["split"].astype(str) == split].copy()
        compatible = _metric_compatibility([row for row in rows if str(row.get("split", "unspecified")) == split])
        if not compatible:
            st.warning("Records have incompatible data, features or evaluation protocols, or duplicate seeds. They are not aggregated. Inspect the raw table and audit records below.")
        metrics = [m for m in ("recall@10", "ndcg@10", "recall@20", "ndcg@20") if m in frame]
        if synthetic or _synthetic(legacy_document, qwen_document):
            st.warning("The following records are explicitly labeled synthetic fixtures for functional checks, not coursework experiment evidence.")
        if metrics and compatible:
            for metric in metrics:
                frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
            metric = st.selectbox("Evaluation metric", metrics, format_func=lambda m: m.replace("ndcg", "NDCG").replace("recall", "Recall"), key=f"metric_name_{domain}")
            aggregates = frame.groupby("model")[metric].agg(["mean", "std", "count"]).reset_index()
            if family == "Three Qwen objectives":
                aggregates = aggregates.set_index("model")
                aggregates = aggregates.reindex([model for model in QWEN_MODES if model in aggregates.index]).reset_index()
            aggregates["method"] = aggregates["model"].map(lambda m: MODEL_SHORT.get(m, m))
            fig = px.bar(aggregates, x="method", y="mean", error_y="std", color="method", color_discrete_sequence=COLORS, color_discrete_map={MODEL_SHORT[model]: OBJECTIVES[model][-1] for model in QWEN_MODES}, labels={"method": "", "mean": metric}, template="plotly_white")
            fig.update_layout(showlegend=False, height=370, margin=dict(l=10, r=15, t=20, b=15), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans, Noto Sans SC, sans-serif", color="#24334d"))
            st.plotly_chart(fig, width="stretch", key=f"metric_chart_{domain}")
            summary = aggregates.copy()
            summary["score"] = summary.apply(lambda row: f'{row["mean"]:.5f} ± {row["std"]:.5f} (n={int(row["count"])})' if row["count"] > 1 else f'{row["mean"]:.5f} (n=1)', axis=1)
            st.dataframe(summary[["method", "score"]], hide_index=True, width="stretch")
            st.caption("Bars show means and error bars show sample standard deviations. A single seed is labeled score (n=1). Qwen repeats the full training pipeline, so selected LoRA and cache hashes may vary by seed. Numerical differences do not establish statistical significance or causality.")
            if family == "Three Qwen objectives" and split == "test":
                from shoplens.visual_charts import seed_metric_figure, paired_delta_figure
                seed_view, paired_view = st.tabs(["Actual results by seed", "Paired changes vs. Frozen"])
                colors = {model: OBJECTIVES[model][-1] for model in QWEN_MODES}
                with seed_view:
                    st.plotly_chart(seed_metric_figure(frame.to_dict("records"), metric, MODEL_SHORT, colors),
                                    width="stretch", key=f"seed_scatter_{domain}")
                with paired_view:
                    st.plotly_chart(paired_delta_figure(frame.to_dict("records"), metric, MODEL_SHORT, colors),
                                    width="stretch", key=f"paired_delta_{domain}")
        ordered = [c for c in ("model", "split", "seed", *metrics, "n_users", "n_targets") if c in frame]
        st.dataframe(frame[ordered], hide_index=True, width="stretch")
        st.download_button("Download evaluation CSV", frame.to_csv(index=False).encode("utf-8-sig"), file_name=f"shoplens_{domain}_{split}_metrics.csv", mime="text/csv", key=f"metric_csv_{domain}")
        low = [{"model": row["model"], "seed": row.get("seed"), **row["low_interaction"]}
               for row in rows if row.get("split", "unspecified") == split and isinstance(row.get("low_interaction"), dict)]
        if low:
            with st.expander("Low-interaction product analysis"):
                if isinstance(low, list):
                    st.dataframe(pd.DataFrame(low), width="stretch", hide_index=True)
                else:
                    st.json(low)
                st.caption("Groups use distinct positive-feedback users during training. Thresholds, eligible user counts and target counts are retained in the result records.")
    if family != "Six original baselines":
        _training_panel(artifact_dir)
    st.divider()
    st.markdown("**Artifact status across collections**")
    states = []
    for name, (english, display_name, _) in DOMAINS.items():
        data_path = root / "data" / "processed" / name
        model_path = root / "artifacts" / name
        result_rows = _metric_rows(_json(model_path / "results.json"))
        new_rows = _metric_rows(_json(root / "artifacts" / "qwen" / name / "results.json"))
        new_rows = [row for row in new_rows if row["model"] in QWEN_MODES and row.get("split") == "test" and not row.get("run_protocol", {}).get("evaluation_subset")]
        identities = {(row["model"], row.get("seed")) for row in new_rows}
        states.append({"Collection": display_name, "Product catalog": "Ready" if (data_path / "items.jsonl").exists() else "Pending", "Baseline evaluation records": len(result_rows), "Formal Qwen records": f"{len(identities)} / 9", "Qwen full-pipeline seeds": "42 / 43 / 44"})
    st.dataframe(pd.DataFrame(states), hide_index=True, width="stretch")
    with st.expander("Data protocol, sources & features"):
        st.markdown("**Data audit / audit.json**")
        st.json(audit or {"status": "No data audit is available yet"})
        st.markdown("**Feature record / features.json**")
        st.json(features or {"status": "No feature record is available yet"})
        if result_document:
            st.markdown("**Full evaluation record / results.json**")
            st.json(result_document)


def main() -> None:
    st.set_page_config(page_title="ShopLens · Multimodal Recommendation Lab", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    root = _root()
    domain = _sidebar(root)
    workspace = st.session_state.get("shoplens_workspace", "Shopping Studio")
    processed = root / "data" / "processed" / domain
    artifacts = root / "artifacts" / domain
    audit = _json(processed / "audit.json")
    features = _json(artifacts / "features.json")
    synthetic = _synthetic(audit, features)
    if workspace == "Shopping Studio":
        st.markdown('<div class="sl-hero"><div class="sl-eyebrow">SHOPLENS · SHOPPING STUDIO</div><span class="sl-tag">DESCRIBE</span><span class="sl-tag">EXPLORE</span><span class="sl-tag">REFINE</span><span class="sl-tag">DECIDE</span><h1>Your next find starts with a brief.</h1><p>Tell us what you need, add a reference photo, and shape the results with your feedback. Compare real products and take your shortlist with you.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sl-hero"><div class="sl-eyebrow">EE6405 · LANGUAGE-DRIVEN RECOMMENDATION</div><span class="sl-tag">QWEN3 EMBEDDING</span><span class="sl-tag">LORA OBJECTIVES × 3</span><span class="sl-tag">FROZEN CLIP VISION</span><h1><span class="sl-title-phrase">One set of preferences.</span> <span class="sl-title-phrase">Three ways to learn.</span></h1><p>Should text models learn semantic similarity or preference order? Choose real products, compare three training objectives, and explore the evidence alongside formal evaluation.</p></div>', unsafe_allow_html=True)
    if synthetic:
        st.warning("Synthetic test artifacts are loaded for end-to-end checks. They do not represent real Amazon products, users or model performance.")
    if not (processed / "items.jsonl").is_file():
        _empty_state(root, domain, "Data for this collection is not ready")
        if workspace == "Research lab":
            _metrics_tab(root, domain, audit, features, synthetic)
        return
    signature = _signature(processed / "items.jsonl", artifacts, root / "artifacts" / "qwen" / domain)
    try:
        items = sorted(_items(str(processed), signature), key=lambda item: str(item["item_id"]))
        models = _available_models(artifacts)
    except Exception as exc:
        _show_error(exc)
        return
    if not items:
        _empty_state(root, domain, "The product catalog is empty")
        return
    catalog = {str(item["item_id"]): item for item in items}
    likes_key = f"shoplens_likes_{domain}"
    if likes_key not in st.session_state:
        st.session_state[likes_key] = []
    liked = st.session_state[likes_key]
    liked[:] = [item_id for item_id in liked if item_id in catalog]
    if workspace == "Shopping Studio":
        from shoplens.studio_ui import render_studio
        render_studio(root, domain, items, catalog, models, liked, signature)
        st.markdown('<div class="sl-footer">ShopLens · Local shopping workspace · Studio retrieval and feedback are interactive diagnostics, separate from the formal research results. Preferences and reference photos stay in this session.</div>', unsafe_allow_html=True)
        return
    with st.container(key="shoplens_metrics"):
        count_columns = st.columns(4)
        count_columns[0].metric("Collection", DOMAINS[domain][1])
        count_columns[1].metric("Products to explore", f"{len(items):,}")
        count_columns[2].metric("Models ready", str(len(models)))
        count_columns[3].metric("Your preferences", f"{len(liked)} items")
    if _is_subset(audit):
        st.caption("This is a data subset. Metrics apply only to the sample scope recorded in the audit.")
    qwen_ready = sum(model in models for model in QWEN_MODES)
    if qwen_ready < 3:
        st.info(f"Qwen seed 42 exports ready: {qwen_ready}/3. Browse products and use the available baselines. The comparison defaults to Qwen once all three objectives are ready.")
    preference, atlas, compare, lab, evidence, experiment = st.tabs([
        "01  Preferences", "02  Product atlas", "03  Model comparison", "04  Preference lab", "05  Recommendation evidence", "06  Experiment dashboard"])
    with preference:
        _catalog_tab(root, domain, items, catalog, models, processed, artifacts, signature, liked)
    from shoplens.visual_ui import atlas as render_atlas, preference_lab
    visual_colors = {model: OBJECTIVES[model][-1] for model in QWEN_MODES}
    with atlas:
        render_atlas(root, domain, models, liked, signature, MODEL_SHORT, visual_colors,
                     lambda item_id, key: _card(root, catalog[item_id], DOMAINS[domain][2], liked, key))
    with compare:
        _compare_tab(root, domain, catalog, models, processed, artifacts, signature, liked)
    with lab:
        _history(root, domain, catalog, liked, "lab")
        preference_lab(root, domain, models, liked, signature, MODEL_SHORT, visual_colors)
    with evidence:
        _explain_tab(root, domain, catalog, models, processed, artifacts, signature, liked)
    with experiment:
        _metrics_tab(root, domain, audit, features, synthetic)
    st.markdown('<div class="sl-footer">ShopLens · Local multimodal recommendation research · Recommendations and metrics use the current artifacts. Preferences stay in this session.</div>', unsafe_allow_html=True)
