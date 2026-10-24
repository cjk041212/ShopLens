#!/usr/bin/env python3
"""Generate the final English report only after the measured study is complete.

Usage:
    python scripts/make_english_report.py --root /path/to/NLP-Coursework --check-only
    python scripts/make_english_report.py --root /path/to/NLP-Coursework

No training, evaluation, network access or source-artifact changes occur here.
The final command writes deliverables/ShopLens_report_en.md atomically. Missing,
synthetic, diagnostic, unmatched or incompatible evidence causes a refusal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shoplens.report_analysis import _primary, _same_protocol, _validate

DOMAINS = ("baby", "sports", "clothing")
MODES = ("popularity", "mf", "tfidf", "text", "multimodal", "adapted")
SEEDS = (42, 43, 44)
METRICS = ("ndcg@10", "recall@10", "ndcg@20", "recall@20")
DOMAIN_NAMES = {"baby": "Baby", "sports": "Sports", "clothing": "Clothing"}
MODEL_NAMES = {
    "popularity": "Popularity", "mf": "BPR + fold-in", "tfidf": "TF-IDF",
    "text": "MiniLM", "multimodal": "CLIP + MiniLM", "adapted": "CLIP + adapted MiniLM",
}


class IncompleteStudy(ValueError):
    """The evidence cannot support a complete final report."""


@dataclass
class Evidence:
    root: Path
    primary: dict
    audits: dict
    adaptations: dict
    budgets: dict
    benchmark: dict


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise IncompleteStudy(f"Missing required evidence: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise IncompleteStudy(f"Unreadable evidence: {path}") from exc
    if not isinstance(value, dict):
        raise IncompleteStudy(f"Expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    if not path.is_file():
        raise IncompleteStudy(f"Missing source file for provenance verification: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value, name: str, *, metric: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise IncompleteStudy(f"Missing or nonfinite measured number: {name}")
    if value < 0 or (metric and value > 1):
        raise IncompleteStudy(f"Measured number outside its valid range: {name}")
    return float(value)


def validate(root: Path) -> Evidence:
    """Require exactly 54 primary records and nine independent MF checks."""
    root = root.resolve()
    primary = _primary(root)  # Shared source/scope/cohort/protocol/feature validation.
    missing = []
    for domain in DOMAINS:
        groups = primary.get(domain, {})
        if set(groups) - set(MODES):
            raise IncompleteStudy(f"Unexpected main methods in {domain}: {sorted(set(groups)-set(MODES))}")
        for mode in MODES:
            found = set(groups.get(mode, {}))
            missing.extend(f"{domain}/{mode}/seed{seed}" for seed in set(SEEDS) - found)
            if found - set(SEEDS):
                raise IncompleteStudy(f"Unexpected primary seeds: {domain}/{mode}/{sorted(found)}")
    if missing:
        complete = 54 - len(missing)
        raise IncompleteStudy(f"Primary matrix incomplete ({complete}/54). Missing: {', '.join(sorted(missing))}")
    if sum(len(rows) for groups in primary.values() for rows in groups.values()) != 54:
        raise IncompleteStudy("The final primary matrix must contain exactly 54 test records")

    audits, adaptations = {}, {}
    for domain in DOMAINS:
        processed = root / "data" / "processed" / domain
        audit = audits[domain] = read_json(processed / "audit.json")
        if audit.get("synthetic") is not False or audit.get("scope") != "full-input":
            raise IncompleteStudy(f"Final course report requires the identified full-input real domain: {domain}")
        reference = primary[domain]["text"][42]
        for filename, key in (("items.jsonl", "items_sha256"), ("events.jsonl", "events_sha256")):
            if sha256(processed / filename) != reference[key]:
                raise IncompleteStudy(f"Current source fingerprint differs from test evidence: {domain}/{filename}")
        for key in ("items", "users", "events"):
            current = audit.get("counts", {}).get(key)
            if not isinstance(current, int) or current <= 0 or current != reference["data_audit"].get("counts", {}).get(key):
                raise IncompleteStudy(f"Data audit count mismatch: {domain}/{key}")
        for mode in MODES:
            for row in primary[domain][mode].values():
                epochs = row.get("training_configuration", {}).get("epochs")
                expected = 0 if mode in {"popularity", "tfidf"} else 5
                if epochs != expected:
                    raise IncompleteStudy(f"Primary fixed-budget mismatch: {domain}/{mode}/{row['seed']}")
                low = row.get("low_interaction", {})
                if low.get("max_training_positive_users") != 5:
                    raise IncompleteStudy(f"Low-interaction threshold differs from <=5: {domain}/{mode}")
                if not isinstance(low.get("n_targets"), int) or low["n_targets"] <= 0:
                    raise IncompleteStudy(f"No measured sparse-item cohort: {domain}/{mode}")
                if (low.get("n_users"), low["n_targets"]) != (
                    reference["low_interaction"].get("n_users"), reference["low_interaction"].get("n_targets")
                ):
                    raise IncompleteStudy(f"Sparse-item cohorts differ: {domain}/{mode}")
                for metric in METRICS:
                    finite(low.get(metric), f"{domain}/{mode}/low/{metric}", metric=True)

        meta = adaptations[domain] = read_json(root / "artifacts" / domain / "text_adaptation" / "adaptation.json")
        if meta.get("real_data") is not True or meta.get("seed") != 42 or meta.get("epochs") != 2:
            raise IncompleteStudy(f"Adaptation provenance/settings incomplete: {domain}")
        if meta.get("base_model") != "sentence-transformers/all-MiniLM-L6-v2" or meta.get("batch_size") != 32:
            raise IncompleteStudy(f"Unexpected text adaptation model/batch: {domain}")
        if meta.get("trainable_layers") != "last two Transformer blocks" or meta.get("max_pairs") != 30000:
            raise IncompleteStudy(f"Unexpected adaptation layers/pair budget: {domain}")
        if meta.get("validation_protocol") != "item-disjoint partition of training-period reviews":
            raise IncompleteStudy(f"Unverified adaptation validation partition: {domain}")
        history = meta.get("history", [])
        if [entry.get("epoch") for entry in history] != [1, 2]:
            raise IncompleteStudy(f"Expected both observed adaptation epochs: {domain}")
        for entry in history:
            finite(entry.get("train_loss"), f"{domain}/adapt/train_loss")
            finite(entry.get("validation_loss"), f"{domain}/adapt/validation_loss")
        if any(not isinstance(meta.get(key), int) or meta[key] <= 0 for key in ("training_pairs", "validation_pairs")):
            raise IncompleteStudy(f"Missing adaptation pair counts: {domain}")
        if sha256(processed / "reviews_train.jsonl") != meta.get("source_reviews_sha256"):
            raise IncompleteStudy(f"Adaptation training-review fingerprint mismatch: {domain}")
        for row in primary[domain]["adapted"].values():
            recorded = row.get("features_metadata", {}).get("adaptation", {})
            for key in ("source_reviews_sha256", "seed", "epochs", "history"):
                if recorded.get(key) != meta.get(key):
                    raise IncompleteStudy(f"Downstream adapted features use different adaptation: {domain}/{key}")

    checks = read_json(root / "experiments" / "baseline_budget.json").get("checks")
    if not isinstance(checks, list) or len(checks) != 9:
        raise IncompleteStudy(f"Independent MF sensitivity matrix incomplete ({len(checks or [])}/9)")
    budgets = {domain: {} for domain in DOMAINS}
    for record in checks:
        domain, seed = record.get("domain"), record.get("seed")
        if domain not in DOMAINS or seed not in SEEDS or seed in budgets[domain]:
            raise IncompleteStudy("Invalid or duplicate independent MF domain/seed")
        row = record.get("result", {})
        _validate(row)
        if row.get("model") != "mf" or row.get("seed") != seed or record.get("epochs") != 20:
            raise IncompleteStudy(f"Unexpected independent MF check: {domain}/{seed}")
        if row.get("training_configuration", {}).get("epochs") != 20:
            raise IncompleteStudy(f"Extended model lacks the measured 20-epoch budget: {domain}/{seed}")
        _same_protocol(primary[domain]["mf"][seed], row, same_model=True)
        curve = record.get("validation_curve", [])
        if len(curve) != 20:
            raise IncompleteStudy(f"Missing 20-point validation curve: {domain}/{seed}")
        for value in curve:
            finite(value, f"{domain}/{seed}/MF validation", metric=True)
        selected = curve.index(max(curve)) + 1
        if record.get("selected_epoch") != selected or row.get("selected_epoch") != selected:
            raise IncompleteStudy(f"MF checkpoint differs from validation-only selection: {domain}/{seed}")
        budgets[domain][seed] = record
    if any(set(records) != set(SEEDS) for records in budgets.values()):
        raise IncompleteStudy("Independent MF sensitivity matrix must cover every domain and seed")

    benchmark = read_json(root / "artifacts" / "local_benchmark.json")
    records = benchmark.get("records", [])
    indexed = {}
    for row in records:
        key = (row.get("domain"), row.get("model"))
        if key in indexed or key[0] not in DOMAINS or key[1] not in MODES:
            raise IncompleteStudy("Duplicate or unexpected local benchmark record")
        if (row.get("device") != "cpu" or row.get("n_liked_items") != 3
                or not isinstance(row.get("repeats"), int) or row["repeats"] < 20):
            raise IncompleteStudy(f"Incomplete actual CPU benchmark protocol: {key}")
        if row.get("n_catalog_items") != audits[key[0]]["counts"]["items"]:
            raise IncompleteStudy(f"Local benchmark uses a different catalogue: {key}")
        for name in ("cold_model_load_and_recommend_ms", "warm_median_ms", "warm_p95_ms"):
            finite(row.get(name), f"local benchmark/{key}/{name}")
        if row["warm_p95_ms"] < row["warm_median_ms"]:
            raise IncompleteStudy(f"Invalid benchmark quantiles: {key}")
        indexed[key] = row
    if set(indexed) != {(domain, mode) for domain in DOMAINS for mode in MODES}:
        raise IncompleteStudy(f"Local CPU benchmark matrix incomplete ({len(indexed)}/18)")
    finite(benchmark.get("process_peak_rss_mib"), "local process peak RSS")
    if not benchmark.get("platform") or not benchmark.get("architecture") or benchmark.get("torch_threads") != 4:
        raise IncompleteStudy("Missing local platform or CPU-thread provenance")
    return Evidence(root, primary, audits, adaptations, budgets, benchmark)


def avg_sd(values, *, signed=False, digits=6) -> str:
    values = list(values)
    if not values:
        raise IncompleteStudy("Cannot format an unmeasured score")
    mean = statistics.mean(values)
    if abs(mean) < 1e-12:
        mean = 0.0
    prefix = "+" if signed else ""
    center = f"{mean:{prefix}.{digits}f}"
    return f"{center}±{statistics.stdev(values):.{digits}f}" if len(values) > 1 else center


def values(evidence, domain, mode, metric="ndcg@10", subgroup=None):
    return [(row[subgroup] if subgroup else row)[metric]
            for row in evidence.primary[domain][mode].values()]


def paired(evidence, domain, before, after, *, subgroup=None):
    first, second = evidence.primary[domain][before], evidence.primary[domain][after]
    return [((second[seed][subgroup] if subgroup else second[seed])["ndcg@10"]
             - (first[seed][subgroup] if subgroup else first[seed])["ndcg@10"]) for seed in SEEDS]


def render(evidence: Evidence) -> str:
    """Render already-validated evidence. This function performs no file writes."""
    lines = [
        "# ShopLens: Review-Adapted Language Representations for Multimodal Product Recommendation", "",
        "## Abstract", "",
        "ShopLens investigates how language representations affect personalized product ranking. "
        "The study compares a traditional TF-IDF representation, pretrained MiniLM sentence embeddings, "
        "and MiniLM adapted using training-period customer reviews. Frozen CLIP image embeddings provide "
        "an additional modality. Three Amazon product domains support a complete comparison of six "
        "methods with three downstream training seeds. An independent MF budget check separates the "
        "fixed primary training budget from longer optimization. The report combines ranking metrics, "
        "paired differences, sparse-item outcomes and measured local CPU inference. These experiments "
        "assess the implemented pipeline under a specific temporal protocol. They do not establish a "
        "general benefit from adding images or reducing a language-alignment loss.", "",
        "## 1. Research question and language methods", "",
        "Customer reviews describe products through experience, while catalogue descriptions emphasize "
        "specifications and promotional language. Our language question is whether adapting an encoder "
        "to connect these expressions improves subsequent recommendation. A separate comparison tests "
        "whether visual features add useful information beyond the text representation. This structure "
        "keeps NLP central: textual feature extraction, sentence representations and contrastive "
        "fine-tuning supply the main controlled changes. CLIP remains frozen throughout.", "",
        "TF-IDF uses unigram and bigram product features with history cosine similarity. MiniLM-L6-v2 "
        "encodes product titles and descriptions into 384-dimensional vectors [3]. Long descriptions "
        "use 224-wordpiece chunks followed by normalized aggregation. Reviews never enter catalogue "
        "text. CLIP ViT-B/32 supplies a separate 512-dimensional image representation [2]. The learned "
        "ranker projects these inputs into a 128-dimensional product space. A small network converts "
        "the mean representation of recent liked products into a user vector. User–item dot products "
        "then rank the catalogue. A new session can use this same history encoder without learning "
        "a new user identifier.", "",
        "## 2. Data and temporal evaluation", "",
        "We use the Baby, Sports and Outdoors, and Clothing, Shoes and Jewelry domains of Amazon "
        "Reviews 2014 [1]. The table reports the final retained data, including readable product text "
        "and decodable photographs. Filtering can change the original five-core property. These are "
        "three domain benchmarks from one source, rather than independently collected corpora.", "",
        "| Domain | Products | Users | Events | Train / validation / test events |",
        "|---|---:|---:|---:|---:|",
    ]
    for domain in DOMAINS:
        count = evidence.audits[domain]["counts"]
        splits = count["splits"]
        lines.append(f"| {DOMAIN_NAMES[domain]} | {count['items']:,} | {count['users']:,} | {count['events']:,} | {splits['train']:,} / {splits['validation']:,} / {splits['test']:,} |")
    lines += ["",
        "Global timestamp quantiles at 0.8 and 0.9 define each domain’s training, validation and test "
        "intervals. Timestamp ties can change the exact record proportions. Ratings of four or five "
        "provide positive feedback. Every target uses only strictly earlier positive interactions, "
        "with at most twenty recent distinct products forming its history. Ranking excludes all "
        "previously liked products and skips repeat targets. Unobserved products are training "
        "contrasts, not verified dislikes.", "",
        "Evaluation ranks the full retained catalogue. Model weights remain fixed, while earlier "
        "validation or test interactions may enter later histories under the prequential protocol. "
        "Each target has one relevant product. We calculate Recall and NDCG at ten and twenty, average "
        "within each user, and then average equally across users. Data fingerprints, evaluated cohorts "
        "and candidate protocols match across methods and seeds. Static metadata snapshots cannot "
        "reconstruct when historical descriptions first became available.", "",
        "## 3. Training and adaptation", "",
        "The primary learned recommenders use five training epochs, AdamW and BPR pairwise ranking "
        "loss [4]. Validation NDCG@10 selects the checkpoint. Popularity uses training counts and "
        "TF-IDF requires no iterative optimization. The behavioural baseline learns BPR item factors, "
        "then estimates a user vector through ridge history fold-in with regularization one. This "
        "inference rule differs from standard evaluation using learned user-ID embeddings. The primary "
        "table is a fixed-budget comparison, not an exhaustive tuning exercise.", "",
        "Language adaptation updates MiniLM’s final two Transformer blocks for two epochs, with "
        "batch size thirty-two and at most thirty thousand positive training-period review–description "
        "pairs. A symmetric contrastive objective aligns paired texts. Each batch retains at most "
        "one pair per product. Adaptation validation holds out product identities within training-period "
        "pairs. The encoder seed remains forty-two. Seeds forty-two, forty-three and forty-four repeat "
        "downstream recommendation training only, so their variation excludes adaptation uncertainty.", "",
        "| Domain | Training / validation pairs | Epoch 1 validation loss | Epoch 2 validation loss |",
        "|---|---:|---:|---:|",
    ]
    for domain in DOMAINS:
        meta = evidence.adaptations[domain]
        history = meta["history"]
        lines.append(f"| {DOMAIN_NAMES[domain]} | {meta['training_pairs']:,} / {meta['validation_pairs']:,} | {history[0]['validation_loss']:.6f} | {history[1]['validation_loss']:.6f} |")
    lines += ["",
        "These losses measure semantic alignment after each adaptation epoch. We did not record "
        "a pre-adaptation validation baseline. An improvement between epochs therefore concerns this "
        "auxiliary objective and cannot establish that personalized ranking improved. Recommendation "
        "requires a separate downstream comparison.", "",
        "## 4. Complete primary results", "",
        "The complete six-method by three-domain matrix contains fifty-four test records. Cells show "
        "the mean and sample standard deviation over three downstream seeds. The four metric columns "
        "retain the same user-macro aggregation. Deterministic baselines can legitimately have zero "
        "between-seed deviation. No missing run or failed evaluation contributes a zero score.", "",
        "| Domain | Method | NDCG@10 | Recall@10 | NDCG@20 | Recall@20 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for domain in DOMAINS:
        for mode in MODES:
            cells = [avg_sd(values(evidence, domain, mode, metric)) for metric in METRICS]
            lines.append(f"| {DOMAIN_NAMES[domain]} | {MODEL_NAMES[mode]} | " + " | ".join(cells) + " |")
    lexical_leads = [DOMAIN_NAMES[domain] for domain in DOMAINS
                     if statistics.mean(values(evidence, domain, "tfidf")) > max(
                         statistics.mean(values(evidence, domain, mode)) for mode in ("text", "multimodal", "adapted"))]
    if lexical_leads:
        lines += ["", f"Under the primary budget, TF-IDF has higher mean NDCG@10 than all three neural variants in {', '.join(lexical_leads)}. "
                  "This is a numerical comparison without a significance test."]
    lines += ["",
        "Matched-seed differences below subtract the earlier method from the later method. They "
        "compare image addition and language adaptation separately. Positive, tied and negative "
        "counts preserve variation that an overall mean can conceal. Three optimization seeds "
        "are not independent samples of the user population and do not provide a significance test.", "",
        "| Domain | Later minus earlier | ΔNDCG@10 mean±SD | Improved / tied / lower |",
        "|---|---|---:|---:|",
    ]
    adaptation_wins = 0
    for domain in DOMAINS:
        for before, after, name in (("text", "multimodal", "Image addition"), ("multimodal", "adapted", "Review adaptation")):
            delta = paired(evidence, domain, before, after)
            wins, losses = sum(v > 0 for v in delta), sum(v < 0 for v in delta)
            if after == "adapted":
                adaptation_wins += wins
            lines.append(f"| {DOMAIN_NAMES[domain]} | {name} | {avg_sd(delta, signed=True)} | {wins} / {3-wins-losses} / {losses} |")
    if 0 < adaptation_wins < 9:
        interpretation = f"Adaptation improves NDCG@10 in {adaptation_wins} of nine matched domain–seed comparisons. The mixed directions do not support a stable recommendation gain."
    elif adaptation_wins == 0:
        interpretation = "Adaptation improves none of the nine matched comparisons. This configuration supplies no observed downstream gain."
    else:
        interpretation = "Adaptation gives positive differences in all nine matched comparisons under this budget. This observation alone does not establish generalization beyond the measured setting."
    lines += ["", interpretation, "",
        "## 5. Products with few interactions", "",
        "The sparse group contains target products with at most five distinct positive training users, "
        "including zero-interaction products. It is not a dedicated strictly unseen-item cold-start "
        "experiment. The same cohort and threshold apply across methods. Its NDCG@10 results are:", "",
        "| Domain | TF-IDF | MiniLM | CLIP + MiniLM | CLIP + adapted MiniLM |",
        "|---|---:|---:|---:|---:|",
    ]
    sparse_failures = []
    for domain in DOMAINS:
        cells = [avg_sd(values(evidence, domain, mode, subgroup="low_interaction")) for mode in ("tfidf", "text", "multimodal", "adapted")]
        lines.append(f"| {DOMAIN_NAMES[domain]} | " + " | ".join(cells) + " |")
        baseline = statistics.mean(values(evidence, domain, "tfidf", subgroup="low_interaction"))
        neural = [statistics.mean(values(evidence, domain, mode, subgroup="low_interaction")) for mode in ("text", "multimodal", "adapted")]
        if baseline > max(neural):
            sparse_failures.append(DOMAIN_NAMES[domain])
    if sparse_failures:
        lines += ["", f"TF-IDF exceeds all three neural variants in this group for {', '.join(sparse_failures)}. "
                  "This negative result prevents interpreting an aggregate change as evidence that the "
                  "neural models solved sparse-product recommendation."]
    else:
        lines += ["", "These group-specific outcomes constrain interpretation of aggregate metrics. "
                  "They do not by themselves establish a cold-start solution."]
    lines += ["",
        "## 6. Independent MF budget sensitivity", "",
        "MF validation curves continued rising near the five-epoch limit, motivating a separate "
        "twenty-epoch check for every domain and seed. This preserves the original primary table. "
        "Validation NDCG selects each extended checkpoint, with no selection by test performance. "
        "The table compares identical seeds, data and fold-in inference:", "",
        "| Domain | Primary MF5 NDCG@10 | Independent MF20 NDCG@10 | Paired Δ mean±SD | Selected epochs |",
        "|---|---:|---:|---:|---|",
    ]
    boundary = []
    for domain in DOMAINS:
        checks = evidence.budgets[domain]
        extended = [checks[seed]["result"]["ndcg@10"] for seed in SEEDS]
        differences = [checks[seed]["result"]["ndcg@10"] - evidence.primary[domain]["mf"][seed]["ndcg@10"] for seed in SEEDS]
        epochs = ",".join(str(checks[seed]["selected_epoch"]) for seed in SEEDS)
        lines.append(f"| {DOMAIN_NAMES[domain]} | {avg_sd(values(evidence, domain, 'mf'))} | {avg_sd(extended)} | {avg_sd(differences, signed=True)} | {epochs} |")
        if any(checks[seed]["selected_epoch"] == 20 for seed in SEEDS):
            boundary.append(DOMAIN_NAMES[domain])
    budget_limit = (f"Boundary checkpoints remain selected in {', '.join(boundary)}, so the longer run does not demonstrate convergence. " if boundary else "Checkpoint selection occurred before the extended limit in these recorded curves. ")
    lines += ["", budget_limit + "The check describes budget sensitivity rather than a new recommendation method. "
              "Neither five nor twenty epochs establishes exhaustive baseline tuning.", "",
        "## 7. Local deployment and measured performance", "",
        "The Streamlit interface supports product selection, matched-history model comparisons, "
        "rank changes and inspection of source descriptions and feature similarities. Similarities "
        "are diagnostic evidence, not causal explanations. Deployment reuses cached product vectors "
        "and small ranking networks, so the large encoders need not remain loaded.", "",
    ]
    bench = evidence.benchmark
    repetitions = sorted({row["repeats"] for row in bench["records"]})
    warm_repeats = (f"{repetitions[0]} warm repetitions per method" if len(repetitions) == 1
                    else f"Between {repetitions[0]} and {repetitions[-1]} warm repetitions per method")
    lines.append(f"Measured platform: {bench['platform']}, {bench['architecture']}, PyTorch {bench['torch']}, "
                 f"four CPU threads. The measurement process reached {bench['process_peak_rss_mib']:.1f} MiB peak RSS. "
                 "Each request selects three catalogue products and returns ten recommendations. "
                 f"{warm_repeats} reuse the loaded scorer. Cold timing includes model loading "
                 "and feature-checksum verification after Python and PyTorch imports.")
    lines += ["", "| Domain | Method | Cold load + rank (ms) | Warm median (ms) | Warm P95 (ms) |",
              "|---|---|---:|---:|---:|"]
    indexed = {(row["domain"], row["model"]): row for row in bench["records"]}
    for domain in DOMAINS:
        for mode in MODES:
            row = indexed[domain, mode]
            lines.append(f"| {DOMAIN_NAMES[domain]} | {MODEL_NAMES[mode]} | {row['cold_model_load_and_recommend_ms']:.2f} | {row['warm_median_ms']:.3f} | {row['warm_p95_ms']:.3f} |")
    lines += ["",
        "These API timings exclude browser rendering, network transfer, encoder execution and "
        "application startup. Peak RSS belongs to the measurement process, not the whole desktop. "
        "The figures describe this workload and host, rather than a latency guarantee for arbitrary "
        "catalogues or devices.", "",
        "## 8. Interpretation and limitations", "",
        "The experiments distinguish semantic alignment from recommendation. A smaller contrastive "
        "validation loss concerns paired language representations, while user ranking also depends "
        "on interaction history, negative sampling and the scoring network. The sparse-item results "
        "and MF budget check constrain claims about model complexity.", "",
        "The evaluation measures recorded positive-feedback prediction within one historical Amazon "
        "source. It cannot establish causal purchase effects, online satisfaction or robustness to "
        "future catalogue shifts. Three downstream seeds measure a limited part of training variation. "
        "Adding images also adds projection and fusion parameters; this is not a parameter-matched ablation, so differences cannot be attributed to image content alone. "
        "Further work should examine additional encoder seeds and validation-led hyperparameter "
        "budgets before extending conclusions. The delivered code, audits and run records support "
        "reproduction of this specific comparison.", "",
        "## References", "",
        "[1] [Amazon Reviews 2014, UCSD public data archive](https://mcauleylab.ucsd.edu/public_datasets/data/amazon/datasets.html).", "",
        "[2] [Radford et al., Learning Transferable Visual Models From Natural Language Supervision, ICML 2021](https://proceedings.mlr.press/v139/radford21a.html). "
        "[Official CLIP implementation](https://github.com/openai/CLIP).", "",
        "[3] [Sentence Transformers, all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).", "",
        "[4] [Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback, UAI 2009](https://arxiv.org/abs/1205.2618).", "",
        "[5] [Streamlit documentation](https://docs.streamlit.io/).", "",
        f"Evidence snapshot: {datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y-%m-%d %H:%M:%S')} UTC+8. "
        "Sources: data/processed/*/audit.json, artifacts/*/results.json, "
        "artifacts/*/text_adaptation/adaptation.json, experiments/baseline_budget.json "
        "and artifacts/local_benchmark.json.", "",
    ]
    return "\n".join(lines)


def word_count(markdown: str) -> int:
    """Count alphanumeric words, excluding Markdown-only delimiter tokens."""
    return sum(bool(re.search(r"[A-Za-z0-9]", word)) for word in markdown.split())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check-only", action="store_true", help="Validate evidence and word count without writing a report")
    args = parser.parse_args(argv)
    try:
        evidence = validate(args.root)
        report = render(evidence)
        count = word_count(report)
        if not 1200 <= count <= 1700:
            raise IncompleteStudy(f"Report word count {count} is outside the intended 1200–1700 range")
        destination = evidence.root / "deliverables" / "ShopLens_report_en.md"
        if args.check_only:
            print(json.dumps({"status": "ready", "primary_records": 54, "budget_checks": 9,
                              "benchmark_records": 18, "words": count, "writes": False}))
            return 0
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=destination.parent, delete=False, encoding="utf-8") as stream:
            stream.write(report)
            temporary = Path(stream.name)
        temporary.replace(destination)
        print(json.dumps({"status": "written", "words": count, "path": str(destination)}))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Refusing final report: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
