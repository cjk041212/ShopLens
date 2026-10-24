"""Keep Plotly lazy classes stable across isolated Streamlit module fixtures.

Several UI fixtures temporarily patch ``sys.modules``. Restoring that dictionary
removes lazily imported child modules while Plotly's parent modules retain their
cached classes. Load the graph types before any fixture takes its snapshot, so
later validators see the same Annotation/Shape classes. This diagnostic warm-up
is entirely in memory and neither reads nor creates experiment artifacts.
"""


def pytest_sessionstart(session):
    import plotly.express as px
    import plotly.graph_objects as go

    from shoplens import visual_charts as charts

    history = [{"item_id": "DIAGNOSTIC-H", "title": "Synthetic history"}]
    candidates = [{"item_id": "DIAGNOSTIC-C", "title": "Synthetic candidate"}]
    results = {mode: [{"item_id": "DIAGNOSTIC-C", "rank": 1}] for mode in charts.MODES}
    records = [{"model": mode, "seed": 42, "split": "test", "domain": "diagnostic fixture", "ndcg@10": .01}
               for mode in charts.MODES]
    figures = [
        charts.rank_flow_figure(results, {"DIAGNOSTIC-C": candidates[0]}),
        charts.overlap_figure(results),
        charts.evidence_heatmap(history, candidates, [[-.1]]),
        charts.network_figure(history, candidates, [[-.1]]),
        charts.seed_metric_figure(records, "ndcg@10"),
        charts.paired_delta_figure(records, "ndcg@10"),
        go.Figure(go.Scattergl(x=[0], y=[0], marker={"line": {"width": 1, "color": "white"}})),
        px.bar(x=["diagnostic"], y=[.01], error_y=[.001], template="plotly_white"),
        px.line(x=[1, 2], y=[.01, .02], markers=True, template="plotly_white"),
    ]
    for figure in figures:
        figure.to_json()
