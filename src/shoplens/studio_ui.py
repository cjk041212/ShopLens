"""Task-oriented local shopping UI over real catalog vectors and session feedback.

Encoding is explicit, on Search only. Feedback reuses those vectors, and this
heuristic Studio ranker is kept separate from formal Qwen experiment results.
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import math
import re
import warnings
from importlib import import_module
from itertools import islice
from pathlib import Path

import streamlit as st
from PIL import Image


EXAMPLES = {
    "baby": "Find a stroller for everyday outings. I prefer a simple, compact style.",
    "sports": "Find a yoga mat for practicing at home. I prefer a simple, minimal style.",
    "clothing": "Find running shoes for everyday exercise. I prefer a simple style.",
}
IMAGE_WORDS = {"baby": "stroller", "sports": "yoga", "clothing": "running"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MODES = ("qwen_frozen", "qwen_semantic", "qwen_preference")
MODEL_LABELS = dict(zip(MODES, ("Frozen Qwen", "Semantic alignment LoRA", "User preference LoRA")))
RANKING_POLICY = "per-objective-query-candidate-retrieval-v2"
STUDIO_CSS = """
<style>
.studio-heading {font-size:30px;font-weight:700;letter-spacing:-.8px;color:#17243c;margin:0 0 8px;}
.studio-lead {color:#65758e;line-height:1.7;margin-bottom:20px;max-width:850px;}
.studio-step {font-size:11px;letter-spacing:.7px;text-transform:uppercase;color:#6680a5;margin:0 0 9px;}
.studio-score {font-size:11px;color:#63758e;margin:5px 0 10px;}
.studio-brief {background:#edf3ff;border:1px solid #dfe8fa;border-radius:14px;padding:18px;margin-bottom:15px;}
.studio-brief h3 {font-size:17px;margin:0 0 9px;}
.studio-brief p {font-size:13px;line-height:1.7;color:#526984;margin:0;}
.studio-source {font-size:11px;color:#8290a5;line-height:1.55;}
@media(max-width:1100px) {
 [class*="st-key-shoplens_studio_layout_"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {flex-wrap:wrap !important;flex-direction:column-reverse;}
 [class*="st-key-shoplens_studio_layout_"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {flex:1 1 100% !important;width:100% !important;min-width:0 !important;}
 [class*="st-key-studio_model_comparison_"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {flex-wrap:wrap !important;}
 [class*="st-key-studio_model_comparison_"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {flex:1 1 100% !important;width:100% !important;min-width:0 !important;}
}
@media(max-width:900px) {
 [class*="st-key-studio_result_grid_"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {flex-wrap:wrap !important;}
 [class*="st-key-studio_result_grid_"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {flex:1 1 100% !important;width:100% !important;min-width:0 !important;}
 .studio-heading {font-size:25px;}
}
</style>
"""


def _encoder_api():
    return import_module("shoplens.studio_encoders")


def _studio_api():
    return import_module("shoplens.studio")


def _model_api():
    return import_module("shoplens.studio_models")


def _title(item):
    return str(item.get("title") or item.get("item_id") or "Untitled product")


def _image_path(root, item):
    raw = item.get("image_path")
    if not isinstance(raw, str) or not raw:
        return None
    root = Path(root).resolve()
    path = (root / raw).resolve()
    return path if path.is_relative_to(root) and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and path.is_file() else None


def _validated_image(payload):
    """Check actual image bytes; no upload is ever written to a local file."""
    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_IMAGE_BYTES:
        raise ValueError("Use a nonempty JPG, PNG or WebP image no larger than 8 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"} or image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Use a JPG, PNG or WebP image with at most 16 million pixels.")
                image.verify()
    except (OSError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("The image could not be decoded safely. Try another JPG, PNG or WebP file.") from exc
    return payload


def _price(item):
    raw = (item.get("metadata") or {}).get("price", item.get("price"))
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if math.isfinite(raw) and raw >= 0 else None
    if isinstance(raw, str) and re.fullmatch(r"\s*\$?\s*\d+(?:,\d{3})*(?:\.\d{1,2})?\s*", raw):
        return float(raw.replace("$", "").replace(",", "").strip())
    return None


def _metadata(item):
    metadata = item.get("metadata") or {}
    price = _price(item)
    categories = metadata.get("categories", item.get("categories", []))
    if isinstance(categories, list):
        categories = " / ".join(str(part) for part in categories[0]) if categories and isinstance(categories[0], list) else " / ".join(map(str, categories))
    return {"Historical catalog price (USD)": f"${price:.2f}" if price is not None else "Unknown",
            "Brand": str(metadata.get("brand") or item.get("brand") or "Unknown"),
            "Catalog categories": str(categories or "Unknown")}


def _category_options(items):
    paths = set()
    for item in items:
        metadata = item.get("metadata") or {}
        categories = metadata.get("categories", []) if isinstance(metadata, dict) else []
        for path in categories if isinstance(categories, list) else []:
            if isinstance(path, list) and path and all(isinstance(part, str) and part.strip() for part in path):
                paths.add(" / ".join(path))
    return sorted(paths)


def _fingerprint(query, avoid, budget, image_bytes, text_weight, signature, required_category=None):
    return hashlib.sha256(json.dumps({"query": query.strip(), "avoid": avoid.strip(), "budget": budget,
                                      "image": hashlib.sha256(image_bytes).hexdigest() if image_bytes else None,
                                      "text_weight": text_weight, "signature": signature,
                                      "required_category": required_category}, sort_keys=True, default=str).encode()).hexdigest()


def _feedback(state, item_id, kind):
    for name in ("positive", "negative", "hidden"):
        state[name] = [value for value in state[name] if value != item_id]
    if kind in ("positive", "negative", "hidden"):
        state[kind].append(item_id)


def _feedback_token(state):
    return tuple(tuple(state[key]) for key in ("positive", "negative", "hidden"))


@st.cache_resource(show_spinner=False, max_entries=3)
def _catalog_vectors(root, domain, signature):
    del signature
    return _encoder_api().load_catalog(root, domain)


def _rank_token(state, statuses):
    artifacts = {mode: {key: statuses.get(mode, {}).get(key)
                        for key in ("ready", "error", "artifact_signature")} for mode in MODES}
    return hashlib.sha256(json.dumps({"policy": RANKING_POLICY, "request": state.get("fingerprint"),
                                      "feedback": _feedback_token(state), "artifacts": artifacts},
                                     sort_keys=True, default=str).encode()).hexdigest()


def _encoding_current(state, statuses):
    """Never reuse a base-only query, or a query from a replaced adapter."""
    if state.get("encoding_policy") != RANKING_POLICY:
        return False
    for field in ("query_vectors_by_model", "avoid_vectors_by_model"):
        if not isinstance(state.get(field), dict) or set(state[field]) != set(MODES):
            return False
    metadata = state.get("query_encoding_metadata", {})
    return all(not statuses.get(mode, {}).get("ready") or
               metadata.get(mode, {}).get("artifact_signature") == statuses.get(mode, {}).get("artifact_signature")
               for mode in MODES)


def _rank(state, root, domain, signature, statuses=None):
    bundle = _catalog_vectors(str(root), domain, signature)
    request = state["request"]
    statuses = statuses if statuses is not None else _model_api().model_status(root, domain)
    if not _encoding_current(state, statuses):
        raise RuntimeError("The query encoder or candidate artifacts changed. Run Find products to encode this request again.")
    rows_by_model, metadata_by_model, errors = {}, {}, {}
    for mode in MODES:
        try:
            if not statuses.get(mode, {}).get("ready"):
                raise RuntimeError(statuses.get(mode, {}).get("error") or "This seed42 model export is unavailable.")
            candidates = _model_api().candidate_catalog(root, domain, mode)
            if list(candidates["item_ids"]) != list(bundle["item_ids"]):
                raise ValueError("This objective's candidate catalog differs from the shared product order.")
            learned = _model_api().personalization(root, domain, mode, state["positive"])
            expected_signature = state["query_encoding_metadata"][mode]["artifact_signature"]
            if any(record.get("artifact_signature") != expected_signature
                   for record in (candidates["metadata"], learned["metadata"])):
                raise RuntimeError("This objective's artifacts changed while ranking. Run Find products again.")
            rows_by_model[mode] = _studio_api().rank_products(
                bundle["items"], candidates["text"], bundle["image"],
                query_vector=state["query_vectors_by_model"][mode], image_vector=state.get("image_vector"),
                positive_ids=state["positive"], negative_ids=state["negative"], hidden_ids=state["hidden"],
                text_weight=request["text_weight"], avoid_vector=state["avoid_vectors_by_model"][mode],
                avoid_terms=request["avoid"], max_price=request["budget"],
                required_category=request.get("required_category"), k=12,
                personalization_scores=learned["scores"],
            )
            metadata_by_model[mode] = {**learned["metadata"],
                                       "request_retrieval": {"query_encoding": state["query_encoding_metadata"][mode],
                                                             "candidate_catalog": candidates["metadata"],
                                                             "query_present": bool(request["query"]),
                                                             "avoid_present": bool(request["avoid"]),
                                                             "reference_image_present": state.get("image_vector") is not None,
                                                             "scope": "Exploratory request retrieval; not the formal history-based evaluation."}}
        except (ValueError, RuntimeError, OSError, ImportError, KeyError) as exc:
            rows_by_model[mode] = []
            errors[mode] = str(exc)
    # Commit all groups together. A failed group has no old rows and no fallback.
    state["previous_by_model"] = state.get("rows_by_model", {})
    state.update(rows_by_model=rows_by_model, model_metadata=metadata_by_model, model_errors=errors,
                 ranking_policy=RANKING_POLICY, ranking_token=_rank_token(state, statuses))
    state["ranked_feedback"] = _feedback_token(state)


def _activate_model(state, mode, reranked=False):
    previous_mode = state.get("displayed_model")
    if previous_mode and previous_mode != mode:
        reference = state.get("rows_by_model", {}).get(previous_mode, [])
        state["rank_reference_label"] = f"vs. {MODEL_LABELS[previous_mode]}"
    elif reranked:
        reference = state.get("previous_by_model", {}).get(mode, [])
        state["rank_reference_label"] = "vs. previous ranking" if reference else None
    else:
        reference = None
    if reference is not None:
        state["previous_ranks"] = {str(row["item_id"]): int(row["rank"]) for row in reference}
    state["rows"] = state.get("rows_by_model", {}).get(mode, [])
    state["displayed_model"] = mode


def _search(state, root, domain, signature, request, image_bytes, fingerprint):
    encoders = _encoder_api()
    status = encoders.status(root)
    if (request["query"] or request["avoid"]) and not status.get("query_ready"):
        raise RuntimeError(status.get("qwen", {}).get("error") or "The local Qwen query encoder is not ready. Prepare its pinned local weights first.")
    if image_bytes and not status.get("image_ready"):
        raise RuntimeError(status.get("clip", {}).get("error") or "The local CLIP image encoder is not ready. Prepare its local weights first.")
    encoding_before = encoders.query_encoding_metadata(root, domain)
    query_vectors = encoders.encode_queries(root, domain, request["query"]) if request["query"] else dict.fromkeys(MODES)
    avoid_vectors = encoders.encode_queries(root, domain, request["avoid"]) if request["avoid"] else dict.fromkeys(MODES)
    image_vector = encoders.encode_image(root, image_bytes) if image_bytes else None
    encoding_metadata = encoders.query_encoding_metadata(root, domain)
    if encoding_before != encoding_metadata:
        raise RuntimeError("Encoder or adapter identity changed during request encoding. Run Find products again; no mixed-version results are displayed.")
    if set(query_vectors) != set(MODES) or set(avoid_vectors) != set(MODES) or set(encoding_metadata) != set(MODES):
        raise ValueError("All three objectives require their own query encoding and provenance; no base-only fallback is used.")
    # Commit a search only after encoding succeeds. On failure the old request
    # never becomes eligible to display results for the new controls.
    state.update(request=request, query_vectors_by_model=query_vectors, avoid_vectors_by_model=avoid_vectors,
                 query_encoding_metadata=encoding_metadata, encoding_policy=RANKING_POLICY,
                 image_vector=image_vector, fingerprint=fingerprint, rows=[], rows_by_model={},
                 displayed_model=None, previous_ranks={}, rank_reference_label=None)
    _rank(state, root, domain, signature)


def _add_shortlist(state, item_id):
    if item_id not in state["shortlist"]:
        state["shortlist"].append(item_id)


def _product_image(root, item):
    path = _image_path(root, item)
    if path:
        st.image(str(path), width="stretch")
    else:
        st.caption("No local product photo is available.")


def _feedback_controls(state, root, domain, catalog):
    count = sum(len(state[key]) for key in ("positive", "negative", "hidden"))
    if not count:
        return
    with st.expander(f"Your feedback · {len(state['positive'])} positive / {len(state['negative'])} negative / {len(state['hidden'])} hidden"):
        st.caption("More/Less like this changes vector-based preference feedback. Hide excludes that exact product. Reference products remain available here and can be shortlisted.")
        options = list(dict.fromkeys(state["positive"] + state["negative"] + state["hidden"]))
        item_id = st.selectbox("Manage a feedback product", options, format_func=lambda item_id: _title(catalog[item_id]), key=f"studio_manage_{domain}")
        role = next(key for key in ("positive", "negative", "hidden") if item_id in state[key])
        st.caption(f"Current feedback: {role} · {item_id}")
        buttons = st.columns(3)
        if buttons[0].button("Undo feedback", key=f"studio_undo_{domain}_{item_id}", width="stretch"):
            _feedback(state, item_id, None)
            st.rerun()
        if buttons[1].button("Shortlist reference", key=f"studio_save_reference_{domain}_{item_id}", width="stretch"):
            _add_shortlist(state, item_id)
            st.rerun()
        if buttons[2].button("Reset all feedback", key=f"studio_reset_{domain}", width="stretch"):
            for key in ("positive", "negative", "hidden"):
                state[key] = []
            st.rerun()


def _result_cards(state, root, domain, catalog):
    rows = state.get("rows", [])
    if not rows:
        st.info("No eligible products remain in the query's semantic candidate pool. Relax the budget, edit the request, or reset feedback. A budget excludes unknown historical prices; soft avoid preferences only change ranking.")
        return
    st.caption("Studio first retrieves a semantic candidate pool, then reranks it with image and feedback signals. Scores belong to this interactive policy, separate from formal Qwen evaluation and live shopping offers.")
    if state.get("rank_reference_label"):
        st.caption(f"Rank movement {state['rank_reference_label']}. Only observed Top-12 positions are compared; missing positions are unknown.")
    with st.container(key=f"studio_result_grid_{domain}"):
        for start in range(0, len(rows), 2):
            for column, row in zip(st.columns(2), rows[start:start + 2]):
                item_id = str(row["item_id"])
                item = catalog[item_id]
                with column, st.container(border=True):
                    before = state.get("previous_ranks", {}).get(item_id)
                    delta = before - int(row["rank"]) if before is not None else None
                    change = (f" · {'↑' if delta > 0 else '↓'} {abs(delta)}" if delta else " · unchanged" if before is not None
                              else " · entered Top-12" if state.get("rank_reference_label") else "")
                    st.markdown(f"**#{row['rank']}{change}**")
                    _product_image(root, item)
                    st.markdown(f'<div class="sl-product-title">{html.escape(_title(item))}</div>', unsafe_allow_html=True)
                    metadata = _metadata(item)
                    st.caption(f"{metadata['Brand']} · {metadata['Historical catalog price (USD)']} historical catalog price · {item_id}")
                    st.markdown(f'<div class="studio-score">Studio score {float(row["score"]):.4f} · within this request only</div>', unsafe_allow_html=True)
                    description = str(item.get("description") or item.get("text") or "No source description is available.")
                    st.write(description[:220] + ("…" if len(description) > 220 else ""))
                    first, second = st.columns(2)
                    if first.button("More like this", key=f"studio_more_{domain}_{item_id}", width="stretch"):
                        _feedback(state, item_id, "positive")
                        st.rerun()
                    if second.button("Less like this", key=f"studio_less_{domain}_{item_id}", width="stretch"):
                        _feedback(state, item_id, "negative")
                        st.rerun()
                    first, second = st.columns(2)
                    if first.button("Hide this", key=f"studio_hide_{domain}_{item_id}", width="stretch"):
                        _feedback(state, item_id, "hidden")
                        st.rerun()
                    if second.button("✓ Shortlisted" if item_id in state["shortlist"] else "Shortlist", key=f"studio_shortlist_{domain}_{item_id}", width="stretch"):
                        _add_shortlist(state, item_id)
                        st.rerun()
                    with st.expander("Source evidence & ranking signals"):
                        for evidence in row.get("evidence", []):
                            st.write(str(evidence))
                        for reason in row.get("reasons", []):
                            st.caption(str(reason))
                        st.caption("Source excerpts and computed similarities are evidence, not generated claims or causal explanations. Product suitability still needs your judgment.")
                        st.json(row.get("components", {}))


def _result_html(root, selected, request, final_choice):
    cards = []
    for item in selected:
        picture = ""
        path = _image_path(root, item)
        if path and path.stat().st_size <= MAX_IMAGE_BYTES:
            try:
                payload = _validated_image(path.read_bytes())
                mime = "image/png" if payload.startswith(b"\x89PNG") else "image/webp" if payload.startswith(b"RIFF") else "image/jpeg"
                picture = f'<img alt="Product photo" src="data:{mime};base64,{base64.b64encode(payload).decode()}">'
            except ValueError:
                pass
        metadata = _metadata(item)
        chosen = str(item["item_id"]) == final_choice
        cards.append(f'<article>{picture}<small>{"FINAL CHOICE" if chosen else "SHORTLIST"}</small><h2>{html.escape(_title(item))}</h2><p>{html.escape(str(item["item_id"]))}</p>'
                     + "".join(f'<p><b>{html.escape(key)}:</b> {html.escape(value)}</p>' for key, value in metadata.items())
                     + f'<p>{html.escape(str(item.get("description") or item.get("text") or "No source description available.")[:1500])}</p></article>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ShopLens shortlist</title>'
            '<style>body{font:16px system-ui,sans-serif;background:#f3f6fb;color:#20334e;margin:0;padding:32px;max-width:1000px;margin:auto}h1{font-size:36px}p{line-height:1.6}article{background:white;border:1px solid #dee6f2;border-radius:18px;padding:24px;margin:24px 0;overflow:auto}img{width:200px;height:180px;object-fit:contain;float:right;max-width:45%}small{color:#3674a9;letter-spacing:1px}footer{font-size:12px;color:#60728e}</style>'
            '<h1>ShopLens · Your shortlist</h1>'
            f'<p>{html.escape(request.get("query", ""))}</p><p>Soft avoid preferences: {html.escape(request.get("avoid") or "None")}</p><p>Source category: {html.escape(request.get("required_category") or "All categories")}</p>'
            f'<p>Selected objective: {html.escape(str(request.get("selected_model_label") or "Not recorded"))}; seed42. Current ranking evidence: {"available" if request.get("ranking_current") else "omitted"}.</p>'
            '<p>Request retrieval is exploratory and does not establish formal recommendation quality or performance gains.</p>'
            f'<details><summary>Retrieval provenance</summary><pre style="white-space:pre-wrap">{html.escape(json.dumps(request.get("request_retrieval"), ensure_ascii=False, indent=2))}</pre></details>'
            + "".join(cards)
            + '<footer>Amazon Reviews 2014 catalog evidence. Prices are historical USD metadata, not current offers. No live stock or suitability is guaranteed. Studio retrieval and feedback are separate from formal model evaluation. This file was generated locally; images are embedded, with no external network resources.</footer></html>')


def _shortlist_panel(state, root, domain, catalog, current_results=True):
    st.markdown("### Your shortlist")
    if not state["shortlist"]:
        st.caption("Save 2–3 candidates, compare the source evidence, then choose your favorite.")
        return
    for item_id in list(state["shortlist"]):
        columns = st.columns([4, 1])
        columns[0].write(_title(catalog[item_id]))
        if columns[1].button("×", help="Remove from shortlist", key=f"studio_remove_shortlist_{domain}_{item_id}"):
            state["shortlist"].remove(item_id)
            if state.get("final_choice") == item_id:
                state["final_choice"] = None
            st.rerun()
    choice_key = f"studio_final_choice_{domain}"
    if st.session_state.get(choice_key) not in ["__none__", *state["shortlist"]]:
        st.session_state[choice_key] = state.get("final_choice") or "__none__"
    choice = st.selectbox("Final choice", ["__none__", *state["shortlist"]],
                          format_func=lambda item_id: "Choose one product…" if item_id == "__none__" else _title(catalog[item_id]), key=f"studio_final_choice_{domain}")
    if st.button("Confirm final choice", disabled=choice == "__none__", key=f"studio_confirm_{domain}", width="stretch"):
        state["final_choice"] = choice
        st.rerun()
    if state.get("final_choice"):
        st.success(f"Chosen: {_title(catalog[state['final_choice']])}")
    request = state.get("request") or {"query": "", "avoid": "", "budget": None}
    mode = state.get("selected_model", MODES[0])
    metadata = state.get("model_metadata", {}).get(mode) if current_results else None
    export_request = {**request, "selected_model": mode, "selected_model_label": MODEL_LABELS[mode],
                      "seed": 42, "ranking_current": current_results, "ranking_policy": RANKING_POLICY,
                      "request_retrieval": metadata.get("request_retrieval") if metadata else None}
    brief = {**export_request, "final_choice": state.get("final_choice"), "reference_image": "Session-only; not exported"}
    selected = [catalog[item_id] for item_id in state["shortlist"]]
    source = {"dataset": "Amazon Reviews 2014", "domain": domain, "metadata": "Prepared local catalog", "price": "Historical metadata, not a live quote",
              "current_objective_at_export": mode, "selected_seed42_model": metadata,
              "request_retrieval": export_request["request_retrieval"],
              "ranking_policy": RANKING_POLICY, "feedback": {key: list(state[key]) for key in ("positive", "negative", "hidden")},
              "selection_note": "Saved products may have been shortlisted under different objectives. The selected objective describes this export's current ranking only."}
    kwargs = dict(recommendations=state.get("rows", []) if current_results else [], source=source)
    with st.expander("Save your decision"):
        if not current_results:
            st.caption("Saved selections are retained. Outdated or unavailable ranking rows are omitted from the export; refresh the search to include current ranking evidence.")
        st.download_button("Download shortlist JSON", _studio_api().shortlist_json(selected, state["shortlist"], brief, **kwargs), file_name=f"shoplens_{domain}_shortlist.json", mime="application/json", key=f"studio_json_{domain}", width="stretch")
        st.download_button("Download shortlist CSV", _studio_api().shortlist_csv(selected, state["shortlist"], brief, **kwargs), file_name=f"shoplens_{domain}_shortlist.csv", mime="text/csv", key=f"studio_csv_{domain}", width="stretch")
        st.download_button("Save result card (HTML)", _result_html(root, selected, export_request, state.get("final_choice")), file_name=f"shoplens_{domain}_result.html", mime="text/html", key=f"studio_html_{domain}", width="stretch")


def _rank_changes(reference, selected):
    before = {str(row["item_id"]): int(row["rank"]) for row in reference}
    after = {str(row["item_id"]): int(row["rank"]) for row in selected}
    shared = set(before) & set(after)
    return {"before": before, "after": after, "shared": len(shared),
            "entered": sorted(set(after) - set(before), key=after.get),
            "exited": sorted(set(before) - set(after), key=before.get),
            "moved": sum(before[item_id] != after[item_id] for item_id in shared),
            "jaccard": len(shared) / len(set(before) | set(after)) if before or after else None}


def _compare_models(state, root, domain, catalog):
    st.markdown("### Same request. Three text models.")
    st.caption("The request, image, constraints and feedback are the same. Each objective uses its own actual query encoder and candidate text vectors, so its candidate pool can differ. This is exploratory request retrieval, not a formal quality evaluation; raw scores are not compared.")
    if not state["positive"] and state["request"].get("query"):
        st.info("Every text model retrieves your request directly; no positive history is required. More like this adds optional learned history-based personalization. Image-only requests share the frozen CLIP encoder and can produce identical lists.")
    elif not state["positive"]:
        st.info("This image-only request uses the shared frozen CLIP encoder, so model lists can coincide. Positive preference feedback is optional; no empty user history is invented.")
    groups = state.get("rows_by_model", {})
    errors = state.get("model_errors", {})
    reference = groups.get(MODES[0], [])
    if MODES[0] in errors:
        st.info("The Frozen reference is unavailable; no baseline rank changes or overlaps are inferred.")
    else:
        order = list(dict.fromkeys(str(row["item_id"]) for mode in MODES for row in groups.get(mode, [])))
        ranks = {mode: {str(row["item_id"]): int(row["rank"]) for row in groups.get(mode, [])} for mode in MODES}
        table = [{"Product": _title(catalog[item_id]), "ASIN": item_id,
                  **{MODEL_LABELS[mode]: ranks[mode].get(item_id) for mode in MODES}} for item_id in order]
        for mode in MODES[1:]:
            if mode in errors:
                continue
            change = _rank_changes(reference, groups.get(mode, []))
            st.markdown(f"**{MODEL_LABELS[mode]} vs. Frozen:** {len(change['entered'])} entered Top-12 · {len(change['exited'])} left Top-12 · {change['moved']} shared items changed rank")
        # Keep all three rank columns visible beside long source product titles.
        rank_table = [{"Product": row["Product"],
                       **{label: row[MODEL_LABELS[mode]] for label, mode in zip(("Frozen", "Semantic", "Preference"), MODES)}}
                      for row in table]
        st.dataframe(rank_table, hide_index=True, width="stretch",
                     column_config={"Product": st.column_config.TextColumn(width="medium"),
                                    **{label: st.column_config.NumberColumn(width="small", format="%d")
                                       for label in ("Frozen", "Semantic", "Preference")}})
        st.caption("Ranks use this exact request and feedback; a smaller number means a higher position. Changes use Frozen as the fixed reference. Missing positions are outside the observed Top-12, not an assumed rank 13.")
    st.markdown("#### Browse each model’s Top 3")
    with st.container(key=f"studio_model_comparison_{domain}"):
        for column, mode in zip(st.columns(3), MODES):
            with column:
                st.markdown(f"**{MODEL_LABELS[mode]}**")
                metadata = state.get("model_metadata", {}).get(mode, {})
                st.caption(f"seed42 · selected epoch {metadata.get('selected_epoch', 'unavailable')}")
                if mode in errors:
                    st.error(f"Unavailable: {errors[mode]}")
                    continue
                rows = groups.get(mode, [])
                if mode == MODES[0]:
                    st.caption(f"Frozen reference · {len(rows)} observed Top-12 items")
                elif MODES[0] not in errors:
                    change = _rank_changes(reference, rows)
                    overlap = f"{change['jaccard']:.0%}" if change["jaccard"] is not None else "undefined"
                    st.caption(f"Shared with Frozen: {change['shared']} / {len(rows)} current items (Frozen has {len(reference)}). Jaccard: {overlap}. Changed positions: {change['moved']}.")
                if not rows:
                    st.caption("No eligible products in this model's candidate pool.")
                for row in rows[:3]:
                    item_id = str(row["item_id"])
                    st.markdown(f"**#{int(row['rank'])}**")
                    _product_image(root, catalog[item_id])
                    st.markdown(f'<div class="sl-product-title">{html.escape(_title(catalog[item_id]))}</div>', unsafe_allow_html=True)
                    if st.button("✓ Shortlisted" if item_id in state["shortlist"] else "Shortlist", key=f"studio_model_save_{domain}_{mode}_{item_id}", width="stretch"):
                        _add_shortlist(state, item_id)
                        st.rerun()


def _compare(state, root, domain, catalog):
    if len(state["shortlist"]) < 2:
        return
    st.markdown("### Compare your candidates")
    selected = st.multiselect("Products to compare", state["shortlist"], default=state["shortlist"][:3], max_selections=3,
                              format_func=lambda item_id: _title(catalog[item_id]), key=f"studio_compare_{domain}")
    if len(selected) < 2:
        st.caption("Choose two or three shortlisted products to compare.")
        return
    for column, item_id in zip(st.columns(len(selected)), selected):
        item = catalog[item_id]
        with column, st.container(border=True):
            _product_image(root, item)
            st.markdown(f'<div class="sl-product-title">{html.escape(_title(item))}</div>', unsafe_allow_html=True)
            st.caption(item_id)
            for label, value in _metadata(item).items():
                st.write(f"{label}: {value}")
            st.write(str(item.get("description") or item.get("text") or "No source description is available.")[:1200])
    st.caption("Comparison uses original catalog evidence. Prices are historical USD values; absent attributes are unknown. There is no live price or stock lookup.")


def render_studio(root, domain, items, catalog, models, liked, signature):
    del models
    root = Path(root)
    st.markdown(STUDIO_CSS, unsafe_allow_html=True)
    store_key = f"shoplens_studio_{domain}"
    if store_key not in st.session_state:
        st.session_state[store_key] = {"positive": list(dict.fromkeys(item_id for item_id in liked if item_id in catalog)),
                                       "negative": [], "hidden": [], "shortlist": [], "rows": [],
                                       "request": None, "fingerprint": None, "final_choice": None}
    state = st.session_state[store_key]
    state.setdefault("selected_model", MODES[0])
    draft = state.get("draft") or state.get("request") or {}
    resuming = bool(draft) and f"studio_query_{domain}" not in st.session_state
    for field in ("positive", "negative", "hidden", "shortlist"):
        state[field] = [item_id for item_id in state[field] if item_id in catalog]
    if state.get("final_choice") not in state["shortlist"]:
        state["final_choice"] = None
    for suffix, default in (("query", draft.get("query", "")), ("avoid", draft.get("avoid", "")),
                            ("budget", draft.get("budget") or 0.0),
                            ("category", draft.get("required_category") or "__all__"),
                            ("image_source", draft.get("image_source", "No image")),
                            ("model", state["selected_model"])):
        st.session_state.setdefault(f"studio_{suffix}_{domain}", default)
    with st.form(f"studio_brief_form_{domain}"):
        raw_brief = st.text_area("Tell us what you are looking for", value=draft.get("brief") or EXAMPLES.get(domain, "Find a product for everyday use."), max_chars=2000, height=100, key=f"studio_brief_{domain}")
        submitted = st.form_submit_button("Interpret brief", type="primary")
    if submitted:
        parsed = _studio_api().parse_brief(raw_brief)
        state["parsed"] = parsed
        st.session_state[f"studio_query_{domain}"] = str(parsed.get("query") or "")
        st.session_state[f"studio_avoid_{domain}"] = str(parsed.get("avoid") or "")
        st.session_state[f"studio_budget_{domain}"] = float(parsed.get("max_price") or 0)
    st.caption("Brief interpretation uses transparent keyword and budget rules, not an LLM. Review and edit the extracted request before searching.")
    query = st.text_input("Search request", placeholder="Interpret the brief above, or enter a request directly…", key=f"studio_query_{domain}", max_chars=2000)
    with st.expander("Refine request & constraints"):
        avoid = st.text_input("Avoid (soft preference)", placeholder="For example: pink, large logos", key=f"studio_avoid_{domain}", max_chars=1000)
        budget_value = st.number_input("Maximum historical catalog price (USD; 0 = no limit)", min_value=0.0, step=5.0, key=f"studio_budget_{domain}")
        category = st.selectbox("Source catalog category", ["__all__", *_category_options(items)],
                                format_func=lambda value: "All source categories" if value == "__all__" else value,
                                key=f"studio_category_{domain}")
        required_category = None if category == "__all__" else category
        st.caption("Category filtering matches the original metadata path. Catalog labels can be incomplete or incorrect; inspect product text before deciding.")
        st.caption("Avoid terms lower scores; they are not a hard product guarantee. A price limit filters historical catalog metadata and excludes unknown prices. No live price, stock or unsupported attribute constraints are guaranteed.")
        for note in state.get("parsed", {}).get("notes", []):
            st.caption(str(note))
        if liked and st.button("Use selections from Research lab", key=f"studio_shared_{domain}"):
            for item_id in liked:
                if item_id in catalog:
                    _feedback(state, item_id, "positive")
            st.rerun()
    image_bytes = None
    image_error = None
    with st.expander("Add a visual reference (optional)"):
        image_source = st.radio("Reference image source", ["No image", "Upload image", "Use catalog example"], horizontal=True, key=f"studio_image_source_{domain}")
        try:
            if image_source == "Upload image":
                upload = st.file_uploader("Upload JPG, PNG or WebP", type=["jpg", "jpeg", "png", "webp"], max_upload_size=8, key=f"studio_image_{domain}")
                if upload is not None:
                    image_bytes = _validated_image(upload.getvalue())
                    state["retained_reference_active"] = False
                elif (resuming or state.get("retained_reference_active")) and state.get("uploaded_reference_bytes"):
                    image_bytes = state["uploaded_reference_bytes"]
                    state["retained_reference_active"] = True
                    st.caption("Using your reference saved in this session. Upload a replacement or choose No image to remove it.")
            elif image_source == "Use catalog example":
                matching = [item for item in items if IMAGE_WORDS.get(domain, "") in _title(item).lower() and item.get("image_path")]
                examples = [str(item["item_id"]) for item in islice((item for item in matching if _image_path(root, item)), 12)]
                if not examples:
                    examples = [str(item["item_id"]) for item in islice((item for item in items if _image_path(root, item)), 12)]
                if examples:
                    example = st.selectbox("Catalog reference photo", examples, format_func=lambda item_id: _title(catalog[item_id]), key=f"studio_image_example_{domain}")
                    image_bytes = _validated_image(_image_path(root, catalog[example]).read_bytes())
                else:
                    st.info("No local catalog photos are available for this collection.")
            if image_bytes:
                st.image(image_bytes, width=180)
            if image_source != "Upload image":
                state["retained_reference_active"] = False
            state["uploaded_reference_bytes"] = image_bytes if image_source == "Upload image" else None
            st.caption("The reference stays in this local session. No uploaded image is written to disk or sent to an external service.")
        except (ValueError, OSError) as exc:
            image_error = str(exc)
            st.error(image_error)
        text_weight = st.slider("Text weight (the remainder uses the reference image)", min_value=0.0, max_value=1.0, value=draft.get("text_weight", .65), step=.05, disabled=image_bytes is None, key=f"studio_text_weight_{domain}")
        st.caption("With no reference image, only available text and preference signals are used.")
    request = {"query": query.strip(), "avoid": avoid.strip(), "budget": budget_value or None,
               "text_weight": text_weight, "required_category": required_category}
    state["draft"] = {**request, "brief": raw_brief, "image_source": image_source}
    fingerprint = _fingerprint(query, avoid, request["budget"], image_bytes, text_weight, signature, required_category)
    mode = st.radio("Learned recommendation objective", MODES, format_func=MODEL_LABELS.get,
                    horizontal=True, key=f"studio_model_{domain}")
    state["selected_model"] = mode
    st.caption("seed42 exports · Same request and feedback across all three objectives. Switching models reuses the search encodings.")
    try:
        statuses = _model_api().model_status(root, domain)
    except (ValueError, RuntimeError, OSError, ImportError, KeyError) as exc:
        statuses = {candidate: {"ready": False, "error": str(exc)} for candidate in MODES}
    reranked = False
    if st.button("Find products", type="primary", disabled=bool(image_error) or not (query.strip() or image_bytes or state["positive"]), key=f"studio_search_{domain}", width="stretch"):
        try:
            with st.spinner("Encoding your request locally and searching the real catalog…"):
                _search(state, root, domain, signature, request, image_bytes, fingerprint)
                reranked = True
        except (ValueError, RuntimeError, OSError, ImportError, KeyError) as exc:
            state["fingerprint"] = None
            st.error(f"Search is unavailable: {exc}")
            st.caption("No placeholder recommendations are used. Prepare the local encoder weights and catalog exports, then try again.")
    current = (state.get("fingerprint") == fingerprint and state.get("request") is not None
               and _encoding_current(state, statuses))
    if current and state.get("ranking_token") != _rank_token(state, statuses):
        try:
            with st.spinner("Updating recommendations from your feedback…"):
                _rank(state, root, domain, signature, statuses)
                reranked = True
        except (ValueError, RuntimeError, OSError, ImportError, KeyError) as exc:
            current = False
            st.error(f"Feedback could not be applied: {exc}")
    if current:
        _activate_model(state, mode, reranked)
    active_ready = current and mode not in state.get("model_errors", {})
    with st.container(key=f"shoplens_studio_layout_{domain}"):
        results_column, brief_column = st.columns([3, 1], gap="large")
        with results_column:
            discover, compare, model_comparison = st.tabs(["Discover", "Compare shortlist", "Compare models"])
            with discover:
                if current:
                    st.markdown("### Find your next favorite")
                    st.caption(f"Current objective: {MODEL_LABELS[mode]} · seed42")
                    _feedback_controls(state, root, domain, catalog)
                    if mode in state.get("model_errors", {}):
                        st.error(f"{MODEL_LABELS[mode]} is unavailable: {state['model_errors'][mode]}")
                    else:
                        metadata = state.get("model_metadata", {}).get(mode, {})
                        if request["query"]:
                            st.caption("Request retrieval is active with this objective's text encoder and candidate vectors. Preference feedback is optional.")
                        elif not metadata.get("active"):
                            st.caption("Image retrieval uses shared CLIP features. More like this adds optional personalization.")
                        if metadata.get("active"):
                            st.caption(f"Learned user head active · {metadata.get('history_count', '?')} recent positive references")
                        _result_cards(state, root, domain, catalog)
                else:
                    if state.get("request"):
                        st.info("Your request, reference or query-encoding policy changed. Click Find products to refresh; previous results are hidden.")
                    else:
                        st.markdown('<div class="sl-empty"><b>Start with a need, finish with a decision.</b><br>Interpret your brief or enter a search request, then click Find products.<br>Use feedback to refine the results, and save 2–3 products to compare.</div>', unsafe_allow_html=True)
            with compare:
                if len(state["shortlist"]) < 2:
                    st.info("Shortlist at least two products to compare their original catalog evidence.")
                _compare(state, root, domain, catalog)
            with model_comparison:
                if current:
                    _compare_models(state, root, domain, catalog)
                else:
                    st.info("Run Find products for the current request before comparing objectives. Previous model rows are hidden.")
        with brief_column:
            stage = "4 · Decision saved" if state.get("final_choice") else "3 · Compare & decide" if state["shortlist"] else "2 · Refine the results" if current else "1 · Describe your need"
            st.markdown(f'<div class="studio-brief"><div class="studio-step">{html.escape(stage)}</div><h3>Your brief</h3><p>{html.escape(query.strip() or "Your search request will appear here.")}</p></div>', unsafe_allow_html=True)
            if avoid.strip():
                st.caption(f"Soft avoid: {avoid.strip()}")
            if request["budget"]:
                st.caption(f"Historical price limit: ${request['budget']:.2f} USD")
            if required_category:
                st.caption(f"Source category: {required_category}")
            if image_bytes:
                st.image(image_bytes, width="stretch")
            st.caption("All suggestions use real local catalog records. Metadata may be incomplete or outdated.")
            st.caption(f"Selected objective: {MODEL_LABELS[mode]} · seed42")
            _shortlist_panel(state, root, domain, catalog, active_ready)
    with st.expander("How this Studio works"):
        st.write("Each selected Frozen or LoRA text encoder embeds the request and soft avoid text against its own 1024-dimensional candidate text vectors. Each model retrieves its own top 200 text-similar candidates; image-only queries use the shared frozen CLIP image features. Boundary ties are retained. Images and feedback rerank within each pool; budgets and exclusions do not fill it with lower-relevance products. This does not guarantee relevance. Long text is truncated to 256 tokens.")
        st.caption("More/Less like this supplies explicit Studio feedback; all positive, negative and hidden reference products are excluded from further suggestions. Your shared Research lab preferences are never changed.")
        st.caption("Request retrieval is an exploratory use of the trained text adapters, separate from the formal history-based recommendation evaluation. Positive references can additionally activate each objective's selected seed42 history/candidate towers and user head. Signals enter the Studio policy as within-catalog percentiles. No performance gains are assumed, and this interaction does not train a model or change formal results.")
