"""Load audited experiment evidence without loading models or inventing results.

The report is a historical audit snapshot. This reader validates its completeness,
identities and numerical consistency; it does not repeat training or byte-audit
the unavailable seed43/44 model exports.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path

DOMAINS = ("baby", "sports", "clothing")
MODELS = ("qwen_frozen", "qwen_semantic", "qwen_preference")
SEEDS = (42, 43, 44)
METRICS = ("ndcg@10", "recall@10", "ndcg@20", "recall@20")
PAIRS = tuple((MODELS[i], MODELS[j]) for i in range(3) for j in range(i + 1, 3))


class ExperimentDataError(ValueError):
    """A report cannot support a complete, numerically consistent gallery."""


def _require(condition, message):
    if not condition:
        raise ExperimentDataError(message)


def _number(value, label, *, nonnegative=False):
    _require(type(value) in (int, float) and math.isfinite(value), f"Invalid finite number: {label}")
    _require(not nonnegative or value >= 0, f"Negative value: {label}")
    return float(value)


def _integer(value, label, *, minimum=0):
    _require(type(value) is int and value >= minimum, f"Invalid integer: {label}")
    return value


def _score(value, label):
    value = _number(value, label, nonnegative=True)
    _require(value <= 1, f"Metric outside [0, 1]: {label}")
    return value


def _same(actual, expected, label):
    _require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12), f"Inconsistent {label}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path):
    raw = path.read_bytes()
    def reject_constant(_):
        raise ExperimentDataError("Nonfinite JSON constant")
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object,
                           parse_constant=reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ExperimentDataError("Experiment source is not valid UTF-8 JSON") from error
    _require(isinstance(value, dict), "Experiment source must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _seed_list(row, label):
    _require(row.get("seeds") == list(SEEDS), f"{label} requires independent seeds 42/43/44")
    if "n_seeds" in row:
        _require(type(row["n_seeds"]) is int and row["n_seeds"] == 3, f"{label} requires n_seeds=3")


def _metrics(row, label):
    _require(isinstance(row, dict), f"Malformed metrics: {label}")
    result = {"n_seeds": 3, "seeds": list(SEEDS)}
    _seed_list(row, label)
    _require(row.get("n_seeds") == 3, f"{label} requires n_seeds=3")
    for metric in METRICS:
        result[metric] = _score(row.get(metric), f"{label}/{metric}")
        result[metric + "_std"] = _number(row.get(metric + "_std"), f"{label}/{metric}_std", nonnegative=True)
    return result


def _training_record(domain, model, seed, evidence):
    curve = evidence.get("adaptation_curve")
    _require(isinstance(curve, list) and len(curve) == 2, f"{domain}/{model}/seed{seed}: missing two-epoch adaptation curve")
    _require(all(isinstance(row, dict) for row in curve), "Malformed adaptation epoch")
    _require([row.get("epoch") for row in curve] == [1, 2], "Adaptation epochs must be 1 and 2")
    values = [_score(row.get("validation_ndcg@10"), "adaptation validation NDCG@10") for row in curve]
    _require(evidence.get("adaptation_validation_ndcg@10") == values, "Validation curve disagrees with epoch records")
    selected = evidence.get("selected_epoch")
    _require(type(selected) is int and selected == values.index(max(values)) + 1, "Selected epoch must maximize validation NDCG@10 (first on tie)")
    samples = evidence.get("sample_metadata", {})
    _require(isinstance(samples, dict), "Missing training sample metadata")
    _require(samples.get("seed") == seed, "Training evidence has the wrong seed identity")
    return {"domain": domain, "model": model, "seed": seed, "epochs": [1, 2],
            "validation_ndcg": values, "selected_epoch": selected,
            "adaptation_cases": _integer(samples.get("sampled_adaptation_cases"), "adaptation cases", minimum=1),
            "validation_cases": _integer(samples.get("validation_cases"), "validation cases", minimum=1),
            "bpr_loss": [_number(row.get("bpr_loss"), "BPR loss", nonnegative=True) for row in curve],
            "semantic_loss": [_number(row.get("semantic_loss"), "semantic loss", nonnegative=True) for row in curve]}


def load_experiment_summary(source_root, runtime_root=None):
    """Return safe chart records from an explicit, complete report snapshot.

    ``submission/results_qwen_summary.json`` takes precedence. Only an explicitly
    supplied runtime root enables the original ``deliverables`` fallback.
    No model weights, review text or per-user records are read.
    """
    source = Path(source_root) / "submission" / "results_qwen_summary.json"
    relative = "submission/results_qwen_summary.json"
    if not source.is_file() and runtime_root is not None:
        source = Path(runtime_root) / "deliverables" / "results_qwen_summary.json"
        relative = "deliverables/results_qwen_summary.json"
    if not source.is_file():
        raise FileNotFoundError("Formal results_qwen_summary.json is missing; no demonstration metrics are synthesized.")
    report, digest = _read_json(source)
    verification = report.get("verification", {})
    _require(isinstance(verification, dict), "Malformed formal verification")
    baseline_preservation = verification.get("baseline_preservation", {})
    _require(isinstance(baseline_preservation, dict), "Malformed baseline preservation evidence")
    _require(report.get("schema_version") == 1, "Unsupported experiment summary schema")
    _require(report.get("complete") is True and verification.get("complete") is True, "Formal audit is incomplete")
    for container, key in ((report, "raw_record_count"), (verification, "expected_records"),
                           (verification, "observed_identities"), (verification, "summarized_records")):
        _require(type(container.get(key)) is int and container[key] == 27, f"Audit requires {key}=27")
    _require(verification.get("missing_units") == [] and verification.get("errors") == [], "Audit contains missing units or errors")
    expected_units = {f"{domain}/{model}/seed{seed}" for domain in DOMAINS for model in MODELS for seed in SEEDS}
    evidence = verification.get("unit_evidence", {})
    _require(isinstance(evidence, dict) and set(evidence) == expected_units, "Audit must identify all 27 unique domain/model/seed units")

    rows = report.get("results", [])
    _require(isinstance(rows, list) and len(rows) == 9, "Expected nine aggregate groups")
    groups, results, aggregate_rows = {}, [], []
    for row in rows:
        _require(isinstance(row, dict), "Malformed aggregate row")
        domain, model = row.get("domain"), row.get("model")
        key = (domain, model)
        _require(domain in DOMAINS and model in MODELS and key not in groups, "Invalid or duplicate aggregate identity")
        normal = {"domain": domain, "model": model, **_metrics(row, f"{domain}/{model}")}
        low = row.get("low_interaction", {})
        normal["low_interaction"] = {**_metrics(low, "low-interaction cohort"),
                                     "max_training_positive_users": _integer(low.get("max_training_positive_users"), "cohort threshold", minimum=1),
                                     "n_users": _integer(low.get("n_users"), "cohort users", minimum=1),
                                     "n_targets": _integer(low.get("n_targets"), "cohort targets", minimum=1)}
        _require(normal["low_interaction"]["max_training_positive_users"] == 5, "Expected low-interaction threshold of five training-positive users")
        results.append(normal)
        groups[key] = normal
        for cohort, metrics in (("overall", normal), ("low_interaction", normal["low_interaction"])):
            aggregate_rows.append({"domain": domain, "model": model, "cohort": cohort,
                                   **{key: value for key, value in metrics.items() if key != "low_interaction"}})
    _require(set(groups) == {(domain, model) for domain in DOMAINS for model in MODELS}, "Aggregate grid is incomplete")

    comparisons = report.get("paired_comparisons", [])
    _require(isinstance(comparisons, list) and len(comparisons) == 9, "Expected nine paired comparisons")
    pairs, paired, paired_rows = set(), [], []
    for row in comparisons:
        _require(isinstance(row, dict), "Malformed paired comparison")
        domain, before, after = row.get("domain"), row.get("before"), row.get("after")
        key = (domain, before, after)
        _require(domain in DOMAINS and (before, after) in PAIRS and key not in pairs, "Invalid or duplicate paired comparison")
        _require(row.get("metric") == "ndcg@10", "Paired report only supports NDCG@10")
        _seed_list(row, "paired comparison")
        differences = row.get("differences", [])
        _require(isinstance(differences, list) and len(differences) == 3, "Expected one paired difference per seed")
        differences = [_number(value, "paired difference") for value in differences]
        mean = _number(row.get("mean_difference"), "mean difference")
        std = _number(row.get("std_difference"), "difference SD", nonnegative=True)
        _same(mean, statistics.mean(differences), "paired mean")
        _same(std, statistics.stdev(differences), "paired sample SD")
        _same(mean, groups[(domain, after)]["ndcg@10"] - groups[(domain, before)]["ndcg@10"], "paired versus aggregate mean")
        normal = {"domain": domain, "before": before, "after": after, "metric": "ndcg@10", "seeds": list(SEEDS),
                  "differences": differences, "mean_difference": mean, "std_difference": std}
        for label, expected in (("wins", sum(value > 1e-12 for value in differences)),
                                ("ties", sum(abs(value) <= 1e-12 for value in differences)),
                                ("losses", sum(value < -1e-12 for value in differences))):
            _require(row.get(label) == expected, f"Inconsistent paired {label}")
            normal[label] = expected
        paired.append(normal)
        pairs.add(key)
        paired_rows.extend({"domain": domain, "before": before, "after": after, "metric": "ndcg@10", "seed": seed,
                            "difference": difference} for seed, difference in zip(SEEDS, differences))
    _require(pairs == {(domain, before, after) for domain in DOMAINS for before, after in PAIRS}, "Paired comparison grid is incomplete")
    comparison_index = {(row["domain"], row["before"], row["after"]): row for row in paired}
    for domain in DOMAINS:
        frozen_semantic = comparison_index[(domain, MODELS[0], MODELS[1])]["differences"]
        frozen_preference = comparison_index[(domain, MODELS[0], MODELS[2])]["differences"]
        semantic_preference = comparison_index[(domain, MODELS[1], MODELS[2])]["differences"]
        for left, right, direct in zip(frozen_semantic, semantic_preference, frozen_preference):
            _same(left + right, direct, "same-seed paired comparison cycle")

    training_all, trace_rows, unit_rows = [], [], []
    for domain in DOMAINS:
        for model in MODELS:
            for seed in SEEDS:
                unit = f"{domain}/{model}/seed{seed}"
                record = evidence[unit]
                _require(isinstance(record, dict), "Malformed unit evidence")
                training = _training_record(domain, model, seed, record)
                training_all.append(training)
                hashes = record.get("source_evidence_sha256", {})
                _require(set(hashes) == {"training.json", f"models/{model}.json", "complete.json", "results.json"}, "Missing source evidence digests")
                _require(all(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) for value in hashes.values()), "Malformed source SHA256")
                unit_rows.append({"domain": domain, "model": model, "seed": seed,
                                  "selected_epoch": training["selected_epoch"], "source_evidence_sha256": hashes.copy()})
                for index, epoch in enumerate(training["epochs"]):
                    trace_rows.append({"domain": domain, "model": model, "seed": seed, "phase": "adaptation", "epoch": epoch,
                                       "validation_ndcg@10": training["validation_ndcg"][index],
                                       "bpr_loss": training["bpr_loss"][index], "semantic_loss": training["semantic_loss"][index],
                                       "selected": epoch == training["selected_epoch"]})
    provenance = {
        "source_relative": relative, "source_sha256": digest, "schema_version": 1,
        "raw_record_count": 27, "aggregate_groups": 9, "paired_comparisons": 9, "seeds": list(SEEDS),
        "summary_structural_checks_passed": True,
        "historical_audit_complete": True,
        "historical_cache_bytes_verified": verification.get("local_cache_bytes_verified") is True,
        "historical_baseline_preserved": baseline_preservation.get("unchanged") is True,
        "live_training_or_full_export_audit_repeated": False,
        "caveats": [
            "Formal metrics evaluate history-to-item full-catalog recommendation, not free-text Shopping Studio requests.",
            "Error bars are sample standard deviations across three complete pipeline seeds (42/43/44), not confidence intervals or significance tests.",
            "Frozen freezes Qwen weights; its recommendation head is still trained.",
            "Training curves are validation NDCG@10 for two adaptation epochs; they are separate from held-out test metrics.",
            "Seed-specific absolute test scores are not present in this aggregate snapshot; paired seed differences are recorded directly.",
            "Clothing LoRA candidate training used MATH SDPA; Baby/Sports retain automatic SDPA. Cross-domain conclusions are conditional on these configurations.",
            "Source hashes identify historical evidence; this gallery does not rerun training or re-audit all model bytes.",
        ],
    }
    return {"results": results, "paired_comparisons": paired,
            "training": [row for row in training_all if row["seed"] == 42],
            "training_traces": [row for row in training_all if row["seed"] == 42],
            "training_all": training_all, "aggregate_rows": aggregate_rows, "paired_rows": paired_rows,
            "trace_rows": trace_rows, "evidence": unit_rows, "provenance": provenance}


def load_training_traces(runtime_root):
    """Read optional real seed42 adaptation traces; reject a partial grid.

    The complete report loader already exposes all 27 historical traces. This
    helper is for explicitly comparing a supplied local runtime snapshot.
    """
    base = Path(runtime_root) / "artifacts" / "qwen"
    paths = [(domain, model, base / domain / model / "seed42" / "training.json") for domain in DOMAINS for model in MODELS]
    if not any(path.is_file() for _, _, path in paths):
        return []
    _require(all(path.is_file() for _, _, path in paths), "Runtime has a partial seed42 training grid")
    traces = []
    for domain, model, path in paths:
        record, digest = _read_json(path)
        _require(record.get("model") == model and record.get("seed") == 42 and record.get("diagnostic") is False,
                 "Runtime trace identity is wrong or diagnostic")
        trace = _training_record(domain, model, 42, record)
        trace["source_sha256"] = digest
        trace["source_relative"] = f"artifacts/qwen/{domain}/{model}/seed42/training.json"
        traces.append(trace)
    return traces
