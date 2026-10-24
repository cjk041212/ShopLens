# ShopLens

A local multimodal shopping and recommendation lab built with **Qwen3-Embedding, LoRA and CLIP**.

[Experiment report](reports/ShopLens_Qwen_report_en.md) · [Presentation](reports/ShopLens_Qwen_presentation.pptx) · [Data and model credits](ATTRIBUTION.md)

## Quick start

Use **Python 3.12**. The pinned setup was exercised on macOS arm64 / M4 CPU; Linux, Windows and MPS have not received equivalent validation.

```bash
git clone --depth 1 https://github.com/cjk041212/ShopLens.git
cd ShopLens
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m streamlit run app.py --server.address 127.0.0.1 \
  --server.headless true --browser.gatherUsageStats false
```

Open **http://127.0.0.1:8501**. **Experiment gallery** and **Benchmark lab** work directly from the recorded results and included photos, without loading model weights.

For live shopping and custom image + text retrieval, prepare the included runtime and official base encoders once, then start the app again:

```bash
python scripts/install_assets.py --repo .
python scripts/prepare_studio_models.py --root . --include-clip
python scripts/check_install.py --root . --require-weights
```

**No retraining or Release package is needed.** Trained heads, LoRA adapters, fusion weights, catalogs, photos and feature caches are in Git. Official Qwen/CLIP base weights download separately during preparation; searches never download them automatically. Allow roughly **1.6 GB for the checkout**, **1.9 GB for runtime restoration**, plus dependencies and base encoders. The installer verifies checksums and refuses to overwrite different existing files.

## Explore

| Workspace | What it does |
| --- | --- |
| **Shopping Studio** | Enter a request, add an optional photo, compare Frozen / Semantic / Preference results, refine and export a shortlist. Model arena follows products across the three lists. |
| **Research lab** | Explore product maps, history-based recommendations and descriptive input-removal diagnostics. |
| **Experiment gallery** | Inspect the original 27 runs, paired seed effects and validation curves. |
| **Benchmark lab** | Explore ESCI search labels, FashionIQ image + language retrieval, history/exposure diagnostics and local CPU timings. Custom fashion edits run four retrieval methods locally. |

The shopping catalog has **47,801 products**. Three model columns and Top-12 lists are display choices. Prices are historical; avoid terms are soft preferences. Custom requests have no ground-truth accuracy score.

## Repository

```text
app.py            Application entry point
src/shoplens/     Models, training, inference and interface
scripts/          Setup, experiments, checks and figure exports
tests/            Automated checks
assets/           Runtime data, benchmark photos/features and source manifests
models/           Trained Qwen heads/adapters and FashionIQ fusion weights
results/          Recorded measurements, protocols and query-level evidence
reports/          One experiment report, one PPT and their figures
```

## Evidence

Three dataset releases support different tasks: **Amazon Reviews 2014, Amazon ESCI and FashionIQ**. They can share underlying products; their scores should not be averaged together.

- The original **27-run recommendation study** has Frozen's highest mean NDCG@10 in all three domains; it does not establish a consistent LoRA gain.
- **ESCI:** 500 official test queries, 9,766 judged candidates, four title-ranking methods. This is candidate-list transfer, not full-catalog search.
- **FashionIQ:** 850 train / 82 dev / 287 evaluation queries after the disclosed cross-split photo repair. Learned fusion improves fixed fusion, but text-only remains better at Recall@10.
- **Diagnostics and timings:** fixed checkpoints and cohorts; CPU ranking-stage speedups do not imply browser or complete-application speedups.

The [report](reports/ShopLens_Qwen_report_en.md) contains exact metrics, protocols, limitations and source links. The **20-slide presentation** follows the application demo, model methods and measured results. The source passed **367 tests and 165 subtests**.

<details>
<summary>Optional checks, offline figures and experiment reproduction</summary>

Verify portable benchmark files without inference or network:

```bash
python scripts/check_install.py --root . --benchmarks-only
```

Export self-contained interactive figures and their source data:

```bash
python scripts/export_experiment_gallery.py --output dist/experiment-gallery
python scripts/export_benchmarks.py --output dist/benchmark-gallery
```

Open the generated `index.html` files. Plotly is embedded for offline viewing.

To reproduce measurements, use a separate checkout to preserve the published outputs. These commands perform real encoding or training and may overwrite result files:

```bash
python scripts/run_esci_benchmark.py --root . --data-dir .cache/esci \
  --model-cache-manifest data/studio_models/manifest.json --device cpu
python scripts/run_fashion_experiment.py --root .
git fetch --depth 1 origin eb53149346207b2500d8e160682e3894fb30125d
python scripts/benchmark_studio.py --root .
```

Recommendation-diagnostic reproduction additionally requires separately prepared chronological interactions, excluded from Git:

```bash
python scripts/run_recommendation_diagnostics.py --runtime-root <prepared-project>
```

Raw reviews/user histories and other seeds' full checkpoints are not part of the demo. The recorded views require none of them. Runtime checksums and provenance remain in `assets/runtime/manifest.json` and the benchmark manifests under `assets/benchmarks/` and `results/`.

</details>
