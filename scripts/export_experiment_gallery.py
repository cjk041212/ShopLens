#!/usr/bin/env python3
"""Export the audited 27-run experiment gallery as portable offline evidence.

Usage: python scripts/export_experiment_gallery.py --output dist/experiment-gallery
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plotly.offline import get_plotlyjs

from shoplens.experiment_charts import (paired_delta_figure, paired_forest_figure,
                                        performance_figure, training_figure)
from shoplens.experiment_data import METRICS, load_experiment_summary


def _csv(rows):
    stream = io.StringIO(newline="")
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return stream.getvalue()


def export_gallery(source_root, output_dir, runtime_root=None):
    document = load_experiment_summary(source_root, runtime_root)
    provenance = document["provenance"]
    figure_source = f"Formal snapshot · seeds 42/43/44 · {provenance['source_sha256'][:12]}"
    sections = []
    config = {"displaylogo": False, "responsive": True,
              "toImageButtonOptions": {"format": "svg", "filename": "shoplens-experiment", "scale": 2}}

    def section(identifier, title, description, figure):
        sections.append(f'<section class="chart-card" id="{html.escape(identifier, quote=True)}">'
                        f'<div class="section-copy"><span class="eyebrow">RECORDED EXPERIMENTS</span>'
                        f'<h2>{html.escape(title)}</h2><p>{html.escape(description)}</p></div>'
                        + figure.to_html(full_html=False, include_plotlyjs=False, config=config,
                                         div_id=f"figure-{identifier}") + '</section>')

    for metric in METRICS:
        section(metric.replace("@", "-"), f"{metric.upper()} across domains",
                "Held-out full-catalog history recommendation. Each point summarizes three independent pipeline seeds; error bars show sample SD.",
                performance_figure(document["results"], metric=metric, source=figure_source))
    section("paired-seeds", "Every paired seed, visible",
            "Recorded NDCG@10 differences versus Frozen, matched by complete pipeline seed. A negative difference favors Frozen; the CSV also retains Semantic-to-Preference comparisons.",
            paired_delta_figure(document["paired_comparisons"], source=figure_source))
    section("paired-summary", "How stable is the change?",
            "Mean paired difference versus Frozen ± sample SD, with three recorded paired seeds. These intervals are not confidence intervals or significance tests.",
            paired_forest_figure(document["paired_comparisons"], source=figure_source))
    section("validation", "Checkpoint selection in two epochs",
            "Seed42 validation NDCG@10 after adaptation epochs 1 and 2. Checkpoints use validation, never held-out test results. Warmup is a separate stage.",
            training_figure(document["training"], source=figure_source))

    links = [("ndcg-10", "NDCG@10"), ("recall-10", "Recall@10"), ("ndcg-20", "NDCG@20"),
             ("recall-20", "Recall@20"), ("paired-seeds", "Paired seeds"),
             ("paired-summary", "Uncertainty"), ("validation", "Validation")]
    navigation = ''.join(f'<a href="#{identifier}">{html.escape(label)}</a>' for identifier, label in links)
    caveats = ''.join(f'<li>{html.escape(text)}</li>' for text in provenance["caveats"])
    source_label, source_hash = html.escape(provenance["source_relative"]), html.escape(provenance["source_sha256"])
    winners = {domain: max((row for row in document["results"] if row["domain"] == domain),
                           key=lambda row: row["ndcg@10"])["model"] for domain in ("baby", "sports", "clothing")}
    if set(winners.values()) == {"qwen_frozen"}:
        main_result = "Frozen has the highest mean held-out NDCG@10 in all three domains. Fine-tuning changes recommendations, but these recorded runs do not establish a consistent improvement over the frozen encoder."
    else:
        labels = {"qwen_frozen": "Frozen", "qwen_semantic": "Semantic", "qwen_preference": "Preference"}
        main_result = "Highest mean held-out NDCG@10 by domain: " + "; ".join(f"{domain.title()}: {labels[model]}" for domain, model in winners.items()) + ". These descriptive comparisons do not establish statistical significance."
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ShopLens | Experiment Gallery</title>
<style>
:root{font-family:Inter,"Avenir Next",Arial,sans-serif;color:#17243b;background:#f4f7fb;scroll-behavior:smooth}
*{box-sizing:border-box}body{margin:0}main{max-width:1280px;margin:auto;padding:32px 24px 64px}
.hero{padding:44px;border-radius:28px;color:#fff;background:linear-gradient(125deg,#152742,#24496a 70%,#176e77);box-shadow:0 20px 45px #233c5f19}
.eyebrow{letter-spacing:.16em;font-size:11px;font-weight:700;opacity:.72}.hero h1{font-size:clamp(34px,5vw,58px);line-height:1.05;letter-spacing:-.045em;margin:18px 0}
.hero p{font-size:17px;line-height:1.65;max-width:760px;color:#d8e6f2}.pills{display:flex;gap:10px;flex-wrap:wrap;margin:26px 0 0}
.pill{border:1px solid #ffffff38;border-radius:999px;padding:10px 15px;font-size:13px;background:#ffffff0c}
nav{display:flex;gap:8px;flex-wrap:wrap;padding:22px 0}nav a,.download a{color:#24577e;text-decoration:none;font-size:13px;font-weight:600;background:#fff;border:1px solid #dce5ef;padding:10px 15px;border-radius:999px}nav a:hover,.download a:hover{background:#eaf3fb}
.insight{padding:20px 25px;border-left:4px solid #278a87;border-radius:8px;background:#e7f3f2;margin:0 0 26px;line-height:1.65;font-size:14px}
.chart-card{border:1px solid #e0e7ef;border-radius:22px;background:#fff;margin:0 0 28px;overflow:hidden;box-shadow:0 10px 30px #20365305;scroll-margin-top:16px}
.section-copy{padding:26px 30px 0}.section-copy .eyebrow{color:#47758c}h2{font-size:24px;letter-spacing:-.025em;margin:10px 0}p{line-height:1.65}.section-copy p{font-size:14px;color:#62738a;margin:0;max-width:960px}
.download{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0 24px}.notes{padding:28px 32px;border-radius:22px;background:#eaf0f7;font-size:13px;line-height:1.8;color:#53657d}.notes h2{color:#233e5f}code{overflow-wrap:anywhere;font-size:12px}.notes ul{padding-left:20px}
@media(max-width:650px){main{padding:15px 10px 35px}.hero{padding:28px 22px;border-radius:20px}.section-copy{padding:20px 16px 0}.notes{padding:20px}.hero p{font-size:15px}h2{font-size:21px}}
</style><script>''' + get_plotlyjs() + '''</script></head><body><main>
<header class="hero"><span class="eyebrow">SHOPLENS / EXPERIMENT GALLERY</span>
<h1>The experiment, beyond a score.</h1>
<p>Explore how frozen embeddings, semantic alignment and preference training behave across three product domains — with paired seeds, honest uncertainty and traceable checkpoints.</p>
<div class="pills"><span class="pill">27 complete pipeline runs</span><span class="pill">3 domains × 3 objectives × 3 seeds</span><span class="pill">Works offline</span></div></header>
<nav aria-label="Experiment sections">''' + navigation + '''</nav>
<div class="insight"><strong>The main result:</strong> ''' + html.escape(main_result) + '''</div>
''' + ''.join(sections) + '''<div class="download"><a href="metrics.csv" download>Metric table · CSV</a><a href="paired_differences.csv" download>Paired seeds · CSV</a><a href="training_curves.csv" download>Validation curves · CSV</a><a href="evidence.json" download>Evidence and source · JSON</a></div>
<footer class="notes"><span class="eyebrow">SCOPE AND PROVENANCE</span><h2>What these figures can support</h2><ul>''' + caveats + '''</ul><p>Source: <code>''' + source_label + '''</code><br>SHA256: <code>''' + source_hash + '''</code></p><p>The original report declared its 27-unit audit complete. This export checks identities, finite values, paired statistics and validation checkpoint consistency; it does not rerun the experiments.</p></footer>
</main></body></html>'''
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(page, encoding="utf-8")
    for filename, rows in (("metrics.csv", document["aggregate_rows"]),
                           ("paired_differences.csv", document["paired_rows"]),
                           ("training_curves.csv", document["trace_rows"])):
        (output / filename).write_text(_csv(rows), encoding="utf-8", newline="")
    (output / "evidence.json").write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"files": ["index.html", "metrics.csv", "paired_differences.csv", "training_curves.csv", "evidence.json"],
            "source_sha256": provenance["source_sha256"], "runs": 27, "figures": len(sections)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runtime-root", type=Path, help="Explicit deliverables fallback if the public summary is absent")
    parser.add_argument("--output", type=Path, default=Path("dist/experiment-gallery"))
    args = parser.parse_args()
    print(json.dumps(export_gallery(args.source_root, args.output, args.runtime_root), indent=2))


if __name__ == "__main__":
    main()
