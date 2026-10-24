"""Interactive views over existing seed42 exports; no model training or encoding."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import visual_charts as charts
from . import visual_data as data

MODES = ("qwen_frozen", "qwen_semantic", "qwen_preference")


def _unit(root, domain, mode):
    return Path(root) / "artifacts/qwen" / domain / mode / "seed42"


def _scorer(root, domain, mode):
    from .qwen_inference import load_scorer
    return load_scorer(Path(root) / "data/processed" / domain, _unit(root, domain, mode), mode)


def _ready(root, domain, models):
    return [mode for mode in MODES if mode in models
            and (_unit(root, domain, mode) / "serving.npz").is_file()]


@st.cache_data(show_spinner=False, max_entries=3)
def _projection(root, domain, signature):
    del signature
    return data.catalog_projection(_scorer(root, domain, "qwen_frozen"))


@st.cache_data(show_spinner=False, max_entries=12)
def _map(root, domain, mode, liked, k, sample_size, signature):
    return data.build_item_map(_scorer(root, domain, "qwen_frozen"), _scorer(root, domain, mode),
                               liked, k=k, sample_size=sample_size,
                               projection=_projection(root, domain, signature))


@st.cache_data(show_spinner=False, max_entries=24)
def _relations(root, domain, mode, liked, k, signature):
    del signature
    from .qwen_inference import recommend
    top = recommend(Path(root) / "data/processed" / domain, _unit(root, domain, mode), mode, list(liked), k=k)
    return data.history_candidate_similarities(_scorer(root, domain, mode), liked, [row["item_id"] for row in top])


@st.cache_data(show_spinner=False, max_entries=12)
def _removals(root, domain, mode, liked, k, signature):
    del signature
    return data.leave_one_out(_scorer(root, domain, mode), liked, k=k)


def _map_figure(points, variance, color, coordinates=None):
    figure = go.Figure()
    styles = {"background": ("Catalog sample", "#CAD5E5", 5, "circle", .48),
              "recommended": ("Current model Top-K", color, 13, "diamond", .95),
              "history": ("Selected preferences", "#EBA044", 16, "star", 1.0)}
    for role, (label, point_color, size, symbol, opacity) in styles.items():
        subset = points[points["role"] == role]
        if subset.empty:
            continue
        details = [[str(row.item_id), html.escape(str(row.title)),
                    f"#{int(row.rank)}" if pd.notna(row.rank) else "—"]
                   for row in subset.itertuples()]
        figure.add_trace(go.Scatter(
            x=subset["x"], y=subset["y"], mode="markers", name=label,
            marker=dict(color=point_color, size=size, symbol=symbol, opacity=opacity,
                        line=dict(width=1 if role != "background" else 0, color="white")),
            customdata=details,
            hovertemplate="%{customdata[1]}<br>ASIN: %{customdata[0]}<br>Recommendation rank: %{customdata[2]}<extra>%{fullData.name}</extra>",
        ))
    figure.update_layout(template="plotly_white", height=540, dragmode="pan", clickmode="event+select", hovermode="closest", uirevision="frozen-reference",
                         modebar=dict(orientation="v", bgcolor="rgba(255,255,255,.8)"),
                         title=dict(text="One fixed map, three recommendation objectives", font_size=17),
                         paper_bgcolor="white", plot_bgcolor="white", margin=dict(l=30, r=15, t=65, b=45),
                         legend=dict(orientation="h", y=1.03, x=0),
                         font=dict(family="Avenir Next, PingFang SC, sans-serif", color="#334764"))
    figure.update_xaxes(title=f"PC1 · {variance[0]:.1%} variance", gridcolor="#EEF2F7", zeroline=False)
    figure.update_yaxes(title=f"PC2 · {variance[1]:.1%} variance", gridcolor="#EEF2F7", zeroline=False)
    if coordinates is not None:
        low, high = coordinates.min(axis=0), coordinates.max(axis=0)
        padding = [(max(float(high[i] - low[i]), 1e-6) * .05) for i in range(2)]
        figure.update_xaxes(range=[float(low[0]) - padding[0], float(high[0]) + padding[0]])
        figure.update_yaxes(range=[float(low[1]) - padding[1], float(high[1]) + padding[1]])
    return figure


def _reset_map_selection(domain):
    st.session_state[f"atlas_plot_revision_{domain}"] = st.session_state.get(f"atlas_plot_revision_{domain}", 0) + 1
    st.session_state.pop(f"atlas_last_click_{domain}", None)


def atlas(root, domain, models, liked, signature, labels, colors, render_product):
    st.subheader("Product atlas")
    st.caption("Each point is a real product. Switch models to compare recommendations, then click or select points to inspect photos, read product text and add preferences.")
    ready = _ready(root, domain, models)
    if "qwen_frozen" not in ready:
        st.info("The atlas requires a complete Frozen seed42 export. It will be available once the artifacts are ready.")
        return
    if st.button("Open product atlas", type="primary", key=f"open_atlas_{domain}"):
        st.session_state[f"atlas_open_{domain}"] = True
    if not st.session_state.get(f"atlas_open_{domain}"):
        st.markdown('<div class="sl-empty"><b>Explore the product catalog</b><br>Gray points show a catalog sample, gold stars mark your preferences, and colored diamonds show the current model’s recommendations.<br>The atlas reads local exports without loading the Qwen language model.</div>', unsafe_allow_html=True)
        return
    controls = st.columns([2, 1, 1])
    mode = controls[0].selectbox("Recommendation model", ready, format_func=lambda m: labels[m], key=f"atlas_model_{domain}")
    k = controls[1].selectbox("Map Top-K", [5, 10, 20], index=1, key=f"atlas_k_{domain}")
    sample_size = controls[2].selectbox("Background products", [250, 500, 1000], index=1, key=f"atlas_sample_{domain}")
    try:
        with st.spinner("Loading local vectors and locating products…"):
            result = _map(str(root), domain, mode, tuple(liked), k, sample_size, signature)
        points, projection = result["points"], result["projection"]
        summary = st.columns(3)
        summary[0].metric("Products on map", f"{len(points):,}")
        summary[1].metric("Recommendations", len(result["top_items"]))
        summary[2].metric("Variance retained in 2D", f"{sum(projection['explained_variance_ratio']):.1%}")
        st.caption(f"Catalog: {result['catalog_size']:,} products. The map includes a fixed sample of {result['background_sample_size']} products, all selections and the current Top-K. Rankings use the full eligible catalog. Stars show all selections; only the 20 most recent enter the user representation.")
        if not liked:
            st.info("No preferences selected yet. The map shows only a catalog sample; inspect a product to add it to your preferences.")
        with st.container(key=f"shoplens_atlas_{domain}"):
            chart_column, item_column = st.columns([3, 1])
            with chart_column:
                event = st.plotly_chart(_map_figure(points, projection["explained_variance_ratio"], colors[mode], projection.get("coordinates")),
                                        width="stretch", key=f"atlas_plot_{domain}_{st.session_state.get(f'atlas_plot_revision_{domain}', 0)}",
                                        on_select="rerun", selection_mode=("points", "box", "lasso"),
                                        config={"displaylogo": False, "modeBarButtonsToRemove": ["toImage", "zoomIn2d", "zoomOut2d", "autoScale2d"]})
            options = list(result["top_items"]["item_id"]) + list(points["item_id"])
            options = list(dict.fromkeys(options))
            selected = event.get("selection", {}).get("points", [])
            if selected:
                clicked = str(selected[0].get("customdata", [""])[0])
                click_key = f"atlas_last_click_{domain}"
                if clicked in options and st.session_state.get(click_key) != clicked:
                    st.session_state[f"atlas_item_{domain}"] = clicked
                    st.session_state[click_key] = clicked
            if st.session_state.get(f"atlas_item_{domain}") not in options:
                st.session_state[f"atlas_item_{domain}"] = options[0]
            titles = points.set_index("item_id")["title"].to_dict()
            with item_column:
                item_id = st.selectbox("Inspect product", options,
                                       format_func=lambda i: str(titles.get(i, i)), key=f"atlas_item_{domain}",
                                       on_change=_reset_map_selection, args=(domain,))
                render_product(item_id, f"atlas_product_{domain}_{item_id}")
                st.caption("Drag to pan and click to inspect. Use the chart tools to switch to box or lasso selection.")
        st.caption("Coordinates use a fixed PCA of the Frozen candidate tower. Switching models changes highlights, not coordinates; independently trained towers are not treated as aligned spaces. The 2D projection loses information and does not replace full-dimensional similarity or final ranking.")
        with st.expander("Inspect mapped products and scores"):
            st.dataframe(points[["item_id", "title", "role", "active_history", "rank", "score"]], hide_index=True, width="stretch")
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        st.error(f"Unable to build the atlas from the current export: {exc}")


def preference_lab(root, domain, models, liked, signature, labels, colors):
    del colors
    st.subheader("Preference lab")
    st.caption("Explore similarities between preference and recommendation vectors, then remove one history item and recompute the current model’s scores and rankings.")
    ready = _ready(root, domain, models)
    if not ready or not liked:
        st.info("Prepare the Qwen seed42 exports and select at least one liked product to open the preference lab.")
        return
    controls = st.columns([2, 1])
    mode = controls[0].selectbox("Experiment model", ready, format_func=lambda m: labels[m], key=f"lab_model_{domain}")
    k = controls[1].selectbox("Recommendations to inspect", [5, 10, 15], index=1, key=f"lab_k_{domain}")
    try:
        relation = _relations(str(root), domain, mode, tuple(liked), k, signature)
        history, candidates, matrix = relation["history"], relation["candidates"], relation["matrix"]
        if candidates.empty:
            st.info("No candidates remain outside the selected products.")
            return
        target = st.selectbox("Focus on a recommendation", ["__all__", *candidates["item_id"]],
                              format_func=lambda i: "All current recommendations" if i == "__all__" else str(candidates.set_index("item_id").loc[i, "title"]),
                              key=f"lab_focus_{domain}_{mode}_{k}")
        shown = candidates if target == "__all__" else candidates[candidates["item_id"] == target]
        values = matrix.loc[:, shown["item_id"]].to_numpy()
        view = st.radio("Relationship view", ["Similarity heatmap", "Preference network"], horizontal=True, key=f"lab_view_{domain}")
        if view == "Similarity heatmap":
            figure = charts.evidence_heatmap(history.to_dict("records"), shown.to_dict("records"), values)
        else:
            figure = charts.network_figure(history.to_dict("records"), shown.to_dict("records"), values, max_edges=60)
        st.plotly_chart(figure, width="stretch", key=f"lab_relation_{domain}")
        st.caption(f"Using the current model’s {len(history)} most recent active history items. Cosines use full 128-dimensional tower vectors, retaining positive and negative values. Edges are not attention, modality weights or a causal decomposition of the final score.")
        st.divider()
        st.markdown("### What changes when one preference is removed?")
        st.caption("Keep the current history of up to 20 items fixed and remove one item at a time, without backfilling older selections. All originally selected products remain excluded from candidates. Δ score = after removal − original, interpreted within the current model only.")
        can_remove = len(history) > 1
        if not can_remove:
            st.info("At least two active history items are required. Removing the only item would create an unsupported empty history; no artificial zero change is shown.")
        token = (mode, tuple(liked), k, signature)
        if st.button("Run removal experiment", type="primary", disabled=not can_remove, key=f"run_removal_{domain}"):
            st.session_state[f"removal_{domain}"] = (token, _removals(str(root), domain, mode, tuple(liked), k, signature))
        stored = st.session_state.get(f"removal_{domain}")
        if not stored or stored[0] != token:
            if can_remove:
                st.caption("Click Run to recompute each removal with the exported user network. Run again after changing preferences or the model.")
            return
        result = stored[1]
        if not result["available"]:
            return
        summary, effects = result["summary"], result["effects"]
        pivot = effects.pivot(index="removed_item_id", columns="target_item_id", values="score_delta")
        pivot = pivot.reindex(index=history["item_id"], columns=candidates["item_id"])
        hover = [[[html.escape(str(source.title)), str(source.item_id),
                   html.escape(str(target.title)), str(target.item_id)]
                  for target in candidates.itertuples()] for source in history.itertuples()]
        figure = go.Figure(go.Heatmap(
            z=pivot.to_numpy(), x=list(range(len(candidates))), y=list(range(len(history))), zmid=0,
            customdata=hover,
            colorscale=[[0, "#C96859"], [.5, "#F7F9FC"], [1, "#347CDB"]],
            colorbar=dict(title="Δ score"),
            hovertemplate="Removed: %{customdata[0]} (%{customdata[1]})<br>Candidate: %{customdata[2]} (%{customdata[3]})<br>Δ score: %{z:+.5f}<extra></extra>"))
        figure.update_layout(template="plotly_white", height=max(340, 230 + 24 * len(history)),
                             title="Remove one history item: score changes for the original Top-K", margin=dict(l=20, r=10, t=60, b=95),
                             xaxis_title="Original recommended products", yaxis_title="Removed history product")
        figure.update_xaxes(tickmode="array", tickvals=list(range(len(candidates))), ticktext=[str(title)[:22] for title in candidates["title"]])
        figure.update_yaxes(tickmode="array", tickvals=list(range(len(history))), ticktext=[str(title)[:24] for title in history["title"]], autorange="reversed")
        st.plotly_chart(figure, width="stretch", key=f"removal_scores_{domain}")
        overlap = go.Figure(go.Bar(x=summary["topk_overlap_ratio"], y=list(range(len(summary))),
                                  orientation="h", marker_color="#347CDB",
                                  customdata=[[html.escape(str(row.title)), str(row.removed_item_id)] for row in summary.itertuples()],
                                  hovertemplate="Removed: %{customdata[0]} (%{customdata[1]})<br>Original Top-K retained: %{x:.0%}<extra></extra>"))
        overlap.update_layout(template="plotly_white", height=max(250, 100 + 28 * len(history)),
                              title="How much of the original Top-K remains?", margin=dict(l=15, r=15, t=60, b=40),
                              xaxis=dict(title="Share of original Top-K retained", range=[0, 1.05], tickformat=".0%"), yaxis_title="")
        overlap.update_yaxes(tickmode="array", tickvals=list(range(len(summary))), ticktext=[str(title)[:30] for title in summary["title"]], autorange="reversed")
        st.plotly_chart(overlap, width="stretch", key=f"removal_overlap_{domain}")
        st.caption("This is a sensitivity check for the current session and seed42 model. It is not a new formal test result or a causal attribution.")
        with st.expander("Inspect recomputed scores and full-catalog ranks"):
            st.dataframe(summary, hide_index=True, width="stretch")
            st.dataframe(effects, hide_index=True, width="stretch")
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        st.error(f"Unable to run the preference experiment with the current export: {exc}")
