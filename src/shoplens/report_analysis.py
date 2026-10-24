"""Measured, seed-matched analysis without writing or changing result artifacts."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

DOMAINS = ("baby", "sports", "clothing")
MAIN_MODELS = ("text", "multimodal", "adapted")
LABELS = {"text": "MiniLM 文本", "multimodal": "CLIP＋MiniLM", "adapted": "CLIP＋适配 MiniLM"}
METRICS = ("ndcg@10", "recall@10", "ndcg@20", "recall@20")
PROTOCOL_KEYS = ("candidate_set", "history", "max_history", "evaluation", "averaging",
                 "positive_feedback", "repeat_targets", "ties", "n_catalog_items")


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Missing/nonfinite metric: {name}")
    if not 0 <= value <= 1:
        raise ValueError(f"Metric outside [0,1]: {name}")
    return float(value)


def _scope(row):
    audit = row["data_audit"]
    scope = audit.get("scope")
    if scope not in {"full-input", "subset"}:
        raise ValueError("Result lacks an explicit full-input/subset data scope")
    # A common, explicitly identified real subset may be reported; it must never
    # be pooled with a full-input run or a different subset.
    subset = audit.get("subset", {}) if scope == "subset" else {}
    return scope, json.dumps(subset, sort_keys=True)


def _validate(row):
    if row.get("split") != "test":
        raise ValueError("Analysis accepts test records only")
    if not isinstance(row.get("seed"), int) or isinstance(row.get("seed"), bool):
        raise ValueError("Result lacks an integer training seed")
    audit = row.get("data_audit", {})
    if audit.get("synthetic") is not False:
        raise ValueError("Synthetic or unverified data cannot enter measured analysis")
    for doc in (audit, row.get("features_metadata", {})):
        if any(doc.get(key) is True for key in ("synthetic", "is_synthetic", "fixture")):
            raise ValueError("Synthetic/fixture record cannot enter measured analysis")
    protocol = row.get("run_protocol", {})
    if protocol.get("evaluation_subset") is not False:
        raise ValueError("Diagnostic or unspecified evaluation subset cannot enter measured analysis")
    if any(key not in protocol for key in PROTOCOL_KEYS):
        raise ValueError("Incomplete candidate/history evaluation protocol")
    if row.get("training_configuration", {}).get("max_examples") is not None:
        raise ValueError("Diagnostic training-example subset cannot enter primary analysis")
    for key in ("items_sha256", "events_sha256"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"Missing data fingerprint: {key}")
    for key in ("n_users", "n_targets"):
        if not isinstance(row.get(key), int) or row[key] <= 0:
            raise ValueError(f"No valid evaluated cohort: {key}")
    for key in METRICS:
        _number(row.get(key), key)
    _scope(row)


def _same_protocol(left, right, *, same_model=False):
    for key in ("items_sha256", "events_sha256", "n_users", "n_targets"):
        if left.get(key) != right.get(key):
            raise ValueError(f"Incompatible measured records: {key}")
    if _scope(left) != _scope(right):
        raise ValueError("Cannot mix full-input and subset runs or different data subsets")
    for key in PROTOCOL_KEYS:
        if left["run_protocol"][key] != right["run_protocol"][key]:
            raise ValueError(f"Incompatible candidate/history protocol: {key}")
    if same_model:
        if left.get("selected_features_sha256") != right.get("selected_features_sha256"):
            raise ValueError("Same-model seeds use different selected features")
        if left["model"] == "mf" and left["run_protocol"].get("mf_inference") != right["run_protocol"].get("mf_inference"):
            raise ValueError("MF checks use different fold-in inference")


def _summary(values, *, signed=False):
    mean = statistics.mean(values)
    if abs(mean) < 1e-12:
        mean = 0.0
    formatted = f"{mean:+.6f}" if signed else f"{mean:.6f}"
    if len(values) > 1:
        return f"{formatted} ± {statistics.stdev(values):.6f}"
    return formatted + "（单次，无标准差）"


def _primary(root):
    domains = {}
    for domain in DOMAINS:
        path = root / "artifacts" / domain / "results.json"
        if not path.exists():
            continue
        groups = defaultdict(dict)
        rows = [row for row in json.loads(path.read_text(encoding="utf-8")).get("results", [])
                if row.get("split") == "test"]
        for row in rows:
            _validate(row)
            if row["seed"] in groups[row["model"]]:
                raise ValueError(f"Duplicate seed record: {domain}/{row['model']}/{row['seed']}")
            if rows:
                _same_protocol(rows[0], row)
            group = groups[row["model"]]
            if group:
                _same_protocol(next(iter(group.values())), row, same_model=True)
            group[row["seed"]] = row
        if groups:
            domains[domain] = dict(groups)
    return domains


def _paired_section(domains):
    lines = ["## 同种子配对比较", "",
             "差值为后一个方法减前一个方法的 NDCG@10，按相同训练种子配对，再计算差值均值与样本标准差。"
             "只使用两种方法都存在的种子，不把缺失种子补成零。三个训练种子不足以给出统计显著性结论，以下也不进行显著性检验。", "",
             "| 数据域 | 对照（后−前） | 匹配种子 | ΔNDCG@10 均值 ± 样本标准差 | 改善／持平／下降 |",
             "|---|---|---|---:|---:|"]
    comments, count = [], 0
    for domain, groups in domains.items():
        for before, after in (("text", "multimodal"), ("multimodal", "adapted")):
            first, second = groups.get(before, {}), groups.get(after, {})
            seeds = sorted(set(first) & set(second))
            if not seeds:
                continue
            differences = [second[seed]["ndcg@10"] - first[seed]["ndcg@10"] for seed in seeds]
            positive = sum(value > 0 for value in differences)
            negative = sum(value < 0 for value in differences)
            equal = len(seeds) - positive - negative
            label = f"{LABELS[after]} − {LABELS[before]}"
            lines.append(f"| {domain} | {label} | {', '.join(map(str, seeds))} | {_summary(differences, signed=True)} | {positive}／{equal}／{negative} |")
            mean = statistics.mean(differences)
            if abs(mean) < 1e-12:
                mean = 0.0
            direction = "增加" if mean > 0 else "下降" if mean < 0 else "不变"
            consistency = ("各匹配种子的方向一致" if positive == len(seeds) or negative == len(seeds)
                           else "种子间没有呈现一致的改善方向")
            comments.append(f"- {domain}：{label}的配对均值{direction}，{consistency}；结论仅对应当前数据划分、训练预算与这些种子。")
            if set(first) != set(second):
                comments.append(f"- {domain} 的该对照尚有未匹配种子，仅比较上述 {len(seeds)} 个共同种子。")
            if _scope(first[seeds[0]])[0] == "subset":
                comments.append(f"- {domain} 使用审计记录中的真实数据子集，不能称为完整原始域结果。")
            count += 1
    if not count:
        lines.append("当前没有完整的同种子方法对照。")
    lines.extend(["", *comments, ""])
    return lines


def _low_section(domains):
    lines = ["## 低交互商品检查", "",
             "低交互组指训练期不同正反馈用户数不超过 5 的目标商品，包含零交互商品。"
             "仍按组内用户宏平均统计，不是专门构造的严格新商品冷启动实验。", "",
             "| 数据域 | 方法 | 种子数 | 有效用户／目标 | NDCG@10 | Recall@10 | NDCG@20 | Recall@20 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    count, comments = 0, []
    for domain, groups in domains.items():
        cohort = None
        low_means = {}
        for mode in MAIN_MODELS:
            rows = list(groups.get(mode, {}).values())
            if not rows or any("low_interaction" not in row for row in rows):
                continue
            lows = [row["low_interaction"] for row in rows]
            for low in lows:
                if low.get("max_training_positive_users") != 5:
                    raise ValueError("Low-interaction groups must use the same <=5 threshold")
                identity = (low.get("n_users"), low.get("n_targets"))
                if any(not isinstance(value, int) or value < 0 for value in identity):
                    raise ValueError("Low-interaction group lacks nonnegative cohort counts")
                if identity[0] > identity[1] or (identity[1] == 0 and identity[0] != 0):
                    raise ValueError("Invalid low-interaction user/target counts")
                if cohort is None:
                    cohort = identity
                elif identity != cohort:
                    raise ValueError("Low-interaction methods/seeds use different evaluated cohorts")
                if identity[1]:
                    for key in METRICS:
                        _number(low.get(key), "low_interaction/" + key)
            if cohort[1]:
                cells = [_summary([low[key] for low in lows]) for key in METRICS]
                low_means[mode] = statistics.mean(low["ndcg@10"] for low in lows)
            else:
                cells = ["无有效目标"] * len(METRICS)
            lines.append(f"| {domain} | {LABELS[mode]} | {len(rows)} | {cohort[0]}／{cohort[1]} | " + " | ".join(cells) + " |")
            count += 1
        if all(mode in low_means for mode in MAIN_MODELS):
            values = "、".join(f"{LABELS[mode]} {low_means[mode]:.6f}" for mode in MAIN_MODELS)
            comments.append(f"- {domain}：三种主要方法的低交互 NDCG@10 均值分别为{values}。总体指标的变化不能直接解释成低交互商品受益。")
            baseline = list(groups.get("tfidf", {}).values())
            if baseline and all(row.get("low_interaction", {}).get("max_training_positive_users") == 5 for row in baseline):
                if any((row["low_interaction"].get("n_users"), row["low_interaction"].get("n_targets")) != cohort for row in baseline):
                    raise ValueError("TF-IDF low-interaction reference uses a different cohort")
                for row in baseline:
                    _number(row["low_interaction"].get("ndcg@10"), "TF-IDF low-interaction ndcg@10")
                reference = statistics.mean(row["low_interaction"]["ndcg@10"] for row in baseline)
                if reference > max(low_means.values()):
                    comments.append(f"- {domain}：同组 TF-IDF 的 NDCG@10 均值为 {reference:.6f}，高于上述三种神经模型；当前结果不支持神经模型已解决低交互推荐。")
    if not count:
        lines.append("当前没有可核对阈值与有效目标数的低交互结果。")
    lines.extend(["", *comments, ""])
    return lines


def _budget_section(root, domains):
    lines, checks = [], defaultdict(list)
    for domain in DOMAINS:
        for path in sorted((root / "artifacts" / "budget_check" / domain).glob("seed*/budget_check.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("domain") != domain or record.get("seed") != record.get("result", {}).get("seed"):
                raise ValueError(f"Budget check identity mismatch: {path}")
            extended = record["result"]
            _validate(extended)
            if extended.get("model") != "mf":
                raise ValueError("Budget sensitivity check must contain an MF result")
            original = domains.get(domain, {}).get("mf", {}).get(record["seed"])
            if original is None:
                continue
            _same_protocol(original, extended, same_model=True)
            primary_epochs = original.get("training_configuration", {}).get("epochs")
            epochs = record.get("epochs")
            if (not isinstance(primary_epochs, int) or not isinstance(epochs, int)
                    or epochs <= primary_epochs
                    or extended.get("training_configuration", {}).get("epochs") != epochs):
                raise ValueError("Budget check must explicitly extend the primary MF epoch limit")
            selected = record.get("selected_epoch")
            if not isinstance(selected, int) or not 1 <= selected <= epochs or selected != extended.get("selected_epoch"):
                raise ValueError("Budget check has an invalid selected checkpoint epoch")
            checks[domain].append((record["seed"], primary_epochs, epochs, selected, original, extended))
    if not checks:
        return lines
    lines = ["## MF 训练预算敏感性检查", "",
             "该检查由主实验中 MF 验证曲线在预算末端仍上升而追加，使用验证集选择 checkpoint。"
             "它保留为独立预算检查，不替换主表；增加最大 epoch 不保证已收敛。只比较同种子、同数据与同候选协议的记录。", "",
             "| 数据域 | 种子 | 主预算／扩展预算 | 主预算 NDCG@10 | 扩展预算 NDCG@10 | 差值 | 扩展预算选中 epoch |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    comments = []
    for domain, rows in checks.items():
        seen = set()
        for seed, original_epochs, epochs, selected, original, extended in rows:
            if seed in seen:
                raise ValueError("Duplicate MF budget seed")
            seen.add(seed)
            difference = extended["ndcg@10"] - original["ndcg@10"]
            lines.append(f"| {domain} | {seed} | {original_epochs}／{epochs} | {original['ndcg@10']:.6f} | {extended['ndcg@10']:.6f} | {difference:+.6f} | {selected} |")
        differences = [extended["ndcg@10"] - original["ndcg@10"] for _, _, _, _, original, extended in rows]
        comments.append(f"- {domain}：{len(rows)} 个匹配种子的预算变化 ΔNDCG@10 为 {_summary(differences, signed=True)}。"
                        "这描述训练预算敏感性，不是方法本身的改进，也不构成显著性结论。")
        if any(selected == epochs for _, _, epochs, selected, _, _ in rows):
            comments.append(f"- {domain} 的部分扩展检查仍在最大 epoch 选中最好验证点，不能宣称 MF 已充分调优。")
    lines.extend(["", *comments, ""])
    return lines


def build_analysis(root) -> str:
    """Return Markdown based only on measured records, raising on unsafe pooling."""
    root = Path(root)
    domains = _primary(root)
    if not domains:
        return "\n## 进一步结果分析\n\n当前没有通过来源与协议校验的正式测试记录。\n"
    lines = ["", "## 进一步结果分析", "",
             "以下分析由逐种子实测记录生成；匹配种子不代表独立的数据抽样，不能由三个训练种子的波动推断用户总体置信区间。", ""]
    lines.extend(_paired_section(domains))
    lines.extend(_low_section(domains))
    lines.extend(_budget_section(root, domains))
    return "\n".join(lines)
