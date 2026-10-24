# ShopLens

**Compare frozen and fine-tuned text models through an interactive multimodal shopping interface and a recorded experiment gallery.**

- **Shopping Studio:** describe a need, optionally add a reference image, refine results with feedback, and save a shortlist.
- **Model arena:** follow a product across three actual recommendation lists, inspect shared and exclusive candidates, and compare observed ranks.
- **Research lab:** explore product maps, history-to-item recommendations, and descriptive input-removal diagnostics.
- **Experiment gallery:** inspect recorded metrics, matched seed effects, and validation checkpoint curves; export SVG/PNG figures, interactive HTML, CSV, and source JSON.

Frozen Qwen, Semantic alignment LoRA, and User preference LoRA each use their corresponding text encoder and candidate vectors. CLIP image features are shared. The catalog contains **47,801 products** across Baby, Sports & Outdoors, and Clothing & Accessories. Displayed Top-3/Top-12 lists are display limits, not catalog sizes.

## Repository layout

| Path | Contents |
| --- | --- |
| `app.py` | Streamlit entry point |
| `src/shoplens/` | Recommendation models, training, inference and interface code |
| `scripts/` | Runtime setup, experiment runners, checks and figure export commands |
| `tests/` | Automated checks |
| `reports/` | One experiment report, the presentation and their screenshots |
| `results/qwen/` | Recorded 27-run experiment summary in JSON and CSV |
| `examples/` | Example exported shopping shortlist |
| `models/qwen/` | Nine seed42 trained recommendation heads and six LoRA adapters |
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

**To view recorded experiments only**, start the app and select **Experiment gallery**. This route reads `results/qwen/results_qwen_summary.json` and needs neither model weights nor the shopping runtime assets:

```bash
python -m streamlit run app.py \
  --server.address 127.0.0.1 --server.port 8501 \
  --browser.gatherUsageStats false
```

Open **http://127.0.0.1:8501/**. The server binds to loopback.

## Prepare the shopping runtime

Shopping Studio and Research lab use the assets already included in this Git checkout. **No Release archive or separate runtime-package download is required, and no retraining is needed.** The tracked runtime contains all 47,801 catalog photos, matching catalogs and feature caches, nine seed42 recommendation heads/exports, and six LoRA adapters. Other seeds' full checkpoints, raw reviews and interaction histories are excluded.

```bash
python scripts/install_assets.py --repo .
python scripts/prepare_studio_models.py --root . --include-clip
python scripts/check_install.py --root . --require-weights
```

`install_assets.py --repo .` uses only local repository files. It verifies the manifest, model files, all data chunks and restored-file checksums before installing; existing different files are never overwritten. A fresh restore needs approximately 1.9 GB of free space beyond the checkout, excluding dependencies and base encoders; the installer checks the destination filesystem before writing. The inventory is in [metadata/runtime-assets.json](metadata/runtime-assets.json), with per-file provenance in `assets/runtime/manifest.json`.

The preparation helper separately downloads official pinned **Qwen/Qwen3-Embedding-0.6B** and **openai/clip-vit-base-patch32** base weights from Hugging Face. These upstream base weights are not duplicated in this repository. This is a download, not training; no external inference API key is needed. For an already prepared supported cache, use `--offline` on the preparation helper. Searches do not download weights automatically.

Start the app with the Streamlit command above. Prices are historical USD metadata; unknown prices are excluded by budget filters. Avoid terms are soft preferences. Similarity and input-removal views are descriptive, not causal explanations.

## Experiment and offline figures

The recorded study has **27 complete pipeline runs**: three domains × three objectives × seeds 42/43/44. The domains share one Amazon Reviews 2014 source. **Frozen has the highest mean held-out NDCG@10 in all three domains**; these runs do not establish a consistent LoRA improvement. Frozen keeps Qwen weights fixed while training the recommendation head.

Formal metrics evaluate full-catalog history-to-item recommendation. Studio free-text retrieval is an exploratory interaction, and changed rankings do not prove improved relevance. Error bars show sample standard deviations across three seeds, not confidence intervals. Training views show the two measured adaptation validation points; checkpoints use validation, never test scores.

Generate the portable seven-figure gallery and its exact data:

```bash
python scripts/export_experiment_gallery.py --output dist/experiment-gallery
```

Open `dist/experiment-gallery/index.html`. Plotly is embedded for offline use; keep companion CSV/JSON files beside the HTML to retain its download links. The figure pack is optional and can be generated locally. Installing tracked runtime assets or exporting charts does not rerun the historical study.

## Report and presentation

- [English experiment report](reports/ShopLens_Qwen_report_en.md)
- [PowerPoint presentation](reports/ShopLens_Qwen_presentation.pptx)
- Recorded results: [JSON](results/qwen/results_qwen_summary.json) · [CSV](results/qwen/results_qwen_summary.csv)

See [ATTRIBUTION.md](ATTRIBUTION.md) for dataset/model sources, upstream credit and rights status.
