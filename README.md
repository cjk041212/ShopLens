# ShopLens

**Explore shopping requests, image + language retrieval, recommendation behavior and local inference cost with real models and measured evidence.**

- **Shopping Studio:** describe a need, optionally add a reference image, refine results with feedback, and save a shortlist.
- **Model arena:** follow a product across three actual recommendation lists, inspect shared and exclusive candidates, and compare observed ranks.
- **Research lab:** explore product maps, history-to-item recommendations, and descriptive input-removal diagnostics.
- **Experiment gallery:** inspect recorded metrics, matched seed effects, and validation checkpoint curves; export SVG/PNG figures, interactive HTML, CSV, and source JSON.
- **Benchmark lab:** inspect 500 labeled ESCI search requests, FashionIQ reference-image modifications, same-user history/exposure diagnostics and measured local inference stages. Edit a FashionIQ modification to run four retrieval methods locally.

Frozen Qwen, Semantic alignment LoRA, and User preference LoRA each use their corresponding text encoder and candidate vectors. CLIP image features are shared. The catalog contains **47,801 products** across Baby, Sports & Outdoors, and Clothing & Accessories. Displayed Top-3/Top-12 lists are display limits, not catalog sizes.

## Repository layout

| Path | Contents |
| --- | --- |
| `app.py` | Streamlit entry point |
| `src/shoplens/` | Recommendation models, training, inference and interface code |
| `scripts/` | Runtime setup, experiment runners, checks and figure export commands |
| `tests/` | Automated checks |
| `reports/` | One experiment report, the presentation, figures and historical interface screenshots |
| `results/qwen/` | Recorded 27-run experiment summary in JSON and CSV |
| `results/esci/`, `results/fashion/` | Separate measured search / image-language protocols, metrics and query examples |
| `results/diagnostics/`, `results/performance/` | Fixed-checkpoint recommendation analysis and real local CPU timing samples |
| `examples/` | Example exported shopping shortlist |
| `models/qwen/` | Nine seed42 trained recommendation heads and six LoRA adapters |
| `models/fashion/` | Trained lightweight CLIP fusion checkpoint |
| `assets/benchmarks/` | Portable bounded benchmark inputs, real FashionIQ photos, features and source checksums |
| `assets/runtime/` | Versioned catalog/photo/cache bytes in bounded Git data chunks, plus their manifest |
| `metadata/` | Runtime inventory and material checksums |

The checkout includes the complete selected shopping runtime, approximately 1.53 GB before Git overhead. Installation restores its original `data/` and `artifacts/` paths locally; these restored directories stay out of Git. Allow additional space for the restored assets, dependencies and official base encoders.

## Install the source

Use **Python 3.12**. The pinned recipe and local CPU encoding were exercised on macOS arm64, including an M4 with 16 GB memory; a fresh Linux/Windows installation and equivalent MPS behavior are not established by that check.

```bash
git clone --depth 1 https://github.com/cjk041212/ShopLens.git
cd ShopLens
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

**To view recorded experiments only**, start the app and select **Experiment gallery** or **Benchmark lab**. These routes read the versioned result files and benchmark photographs without loading encoders or restoring the shopping runtime:

```bash
python -m streamlit run app.py \
  --server.address 127.0.0.1 --server.port 8501 \
  --browser.gatherUsageStats false
```

Open **http://127.0.0.1:8501/**. The server binds to loopback.

Optionally verify all new portable benchmark records, the fusion checkpoint/features and all 3,043 FashionIQ photos without inference or model downloads:

```bash
python scripts/check_install.py --root . --benchmarks-only
```

## Prepare the shopping runtime

Shopping Studio and Research lab use the assets already included in this Git checkout. **No Release archive or separate runtime-package download is required, and no retraining is needed.** The tracked runtime contains all 47,801 catalog photos, matching catalogs and feature caches, nine seed42 recommendation heads/exports, and six LoRA adapters. Other seeds' full checkpoints, raw reviews and interaction histories are excluded.

```bash
python scripts/install_assets.py --repo .
python scripts/prepare_studio_models.py --root . --include-clip
python scripts/check_install.py --root . --require-weights
```

`install_assets.py --repo .` uses only local repository files. It verifies the manifest, model files, all data chunks and restored-file checksums before installing; existing different files are never overwritten. A fresh restore needs approximately 1.9 GB of free space beyond the checkout, excluding dependencies and base encoders; the installer checks the destination filesystem before writing. The inventory is in [metadata/runtime-assets.json](metadata/runtime-assets.json), with per-file provenance in `assets/runtime/manifest.json`.

The preparation helper separately downloads official pinned **Qwen/Qwen3-Embedding-0.6B** and **openai/clip-vit-base-patch32** base weights from Hugging Face. These upstream base weights are not duplicated in this repository. This is a download, not training; no external inference API key is needed. For an already prepared supported cache, use `--offline` on the preparation helper. Searches do not download weights automatically.

FashionIQ's **Try my own modification** uses the bundled image features and trained fusion checkpoint plus the prepared local CLIP text encoder. Select a recorded reference image, enter a new modification, and press **Run my modification**. All four methods rank the same category pool, excluding the reference. A new custom request has no labeled target or accuracy score; the recorded paired-target ranks apply only to the original dataset captions. Base-model preparation is required for this live action, while recorded views remain usable offline.

Start the app with the Streamlit command above. Prices are historical USD metadata; unknown prices are excluded by budget filters. Avoid terms are soft preferences. Similarity and input-removal views are descriptive, not causal explanations.

## Experiment and offline figures

The recorded study has **27 complete pipeline runs**: three domains × three objectives × seeds 42/43/44. The domains share one Amazon Reviews 2014 source. **Frozen has the highest mean held-out NDCG@10 in all three domains**; these runs do not establish a consistent LoRA improvement. Frozen keeps Qwen weights fixed while training the recommendation head.

Formal metrics evaluate full-catalog history-to-item recommendation. Studio free-text retrieval is an exploratory interaction, and changed rankings do not prove improved relevance. Error bars show sample standard deviations across three seeds, not confidence intervals. Training views show the two measured adaptation validation points; checkpoints use validation, never test scores.

Generate the portable seven-figure gallery and its exact data:

```bash
python scripts/export_experiment_gallery.py --output dist/experiment-gallery
```

Open `dist/experiment-gallery/index.html`. Plotly is embedded for offline use; keep companion CSV/JSON files beside the HTML to retain its download links. The figure pack is optional and can be generated locally. Installing tracked runtime assets or exporting charts does not rerun the historical study.

## Automatic benchmark extension

The new measurements use **three dataset releases**: Amazon Reviews 2014, Amazon ESCI, and FashionIQ. Baby/Sports/Clothing remain three domains of the first source. These releases can share underlying Amazon products; separate annotations do not imply disjoint products or pretraining. Do not compare numeric scores across these different tasks or combine them into one average.

- **ESCI:** 500 deterministic official English test queries, the complete official per-query candidate lists (2–40 products), title-only reranking with TF-IDF and the three matching Qwen encoders. Clothing seed42 adapters are fixed in advance. No ESCI fine-tuning or test-based best-adapter selection occurs. Declared gains are E=1, S=.1, C=.01, I=0; the official qrel-script S/C discrepancy is reported separately.
- **FashionIQ:** final 850 train / 82 internal-dev / 287 official-validation queries. Unavailable images are omitted without replacement. A disclosed integrity amendment removes byte-identical photos from lower-priority train/dev splits while preserving the complete fixed evaluation set/gallery; final phase pools have no cross-split ID or JPEG-SHA overlap. Category evaluation pools contain 281 / 292 / 293 images before excluding each reference item. CLIP stays frozen; the 262,784-parameter fusion projection is trained and selected on internal dev only. Image-only, text-only and fixed-fusion baselines are retained. These are bounded available-image scores, not full-gallery leaderboard scores.
- **Recommendation diagnostics:** fixed seed42 and 1,000 deterministic users per domain; identical targets at history caps 1/3/5/10/20. All earlier positives remain excluded even with short visible histories. Popularity uses TRAIN users only. Only aggregates and public catalog products are distributed, with no raw user histories.
- **Local performance:** real query/image encoders, three selected heads, cache boundaries and complete-row before/after ranker parity. CPU timings are environment-specific; exact repeat cache hits do not represent new-query forward latency.

Recorded metrics and the trained fusion model ship in Git; viewing them requires no retraining. Export ten interactive charts with their exact source JSON:

```bash
python scripts/export_benchmarks.py --output dist/benchmark-gallery
```

Open `dist/benchmark-gallery/index.html`. Plotly is embedded; this export requires no network or inference.

Optional reproduction commands (they perform new downloads/encoding or training and can overwrite the declared output paths; use a separate checkout when preserving the published measurements):

```bash
python scripts/run_esci_benchmark.py --root . --data-dir .cache/esci \
  --model-cache-manifest data/studio_models/manifest.json --device cpu
python scripts/run_fashion_experiment.py --root .
# Only needed for the historical before/after timing comparison in a shallow clone:
git fetch --depth 1 origin eb53149346207b2500d8e160682e3894fb30125d
python scripts/benchmark_studio.py --root .
```

ESCI's runner verifies the full examples-file SHA and fetches only required product Parquet columns with bounded HTTP ranges. It records the selected inputs and range hashes; it does not claim to verify the complete upstream 1.1 GB product file. FashionIQ retains missing-image counts and source revisions in its manifest. The performance comparison requires the public baseline commit named in its script; for a shallow clone, fetch that commit before running it.

Recommendation-diagnostic reproduction additionally requires prepared chronological interactions, which are intentionally excluded from the public runtime. Point `python scripts/run_recommendation_diagnostics.py --runtime-root <prepared-project>` at separately prepared source events and selected exports. The included aggregate diagnostic JSON and its UI remain usable without those private local preparation files.

## Report and presentation

- [English experiment report](reports/ShopLens_Qwen_report_en.md)
- [PowerPoint presentation](reports/ShopLens_Qwen_presentation.pptx)
- Recorded results: [JSON](results/qwen/results_qwen_summary.json) · [CSV](results/qwen/results_qwen_summary.csv)

See [ATTRIBUTION.md](ATTRIBUTION.md) for dataset/model sources, upstream credit and rights status.
