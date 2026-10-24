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
| `metadata/` | Runtime download metadata and material checksums |

Local runtime data and model assets install under `data/` and `artifacts/`; these generated directories stay out of Git.

## Install the source

Use **Python 3.12**. The pinned recipe and local CPU encoding were exercised on macOS arm64, including an M4 with 16 GB memory; a fresh Linux/Windows installation and equivalent MPS behavior are not established by that check.

```bash
git clone https://github.com/cjk041212/ShopLens.git
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

Shopping Studio and Research lab additionally require the separate [runtime-v1 assets](https://github.com/cjk041212/ShopLens/releases/tag/runtime-v1) and pinned base encoders. The runtime archive is **1,457,809,593 bytes**. It includes catalogs/photos, feature caches, nine trained seed42 recommendation heads/exports and six LoRA adapters. These trained weights are in the runtime attachment, rather than tracked in Git; using them requires no retraining. Base weights, raw reviews and interaction histories are excluded. The download URL, checksum and scope are recorded in [metadata/runtime-assets.json](metadata/runtime-assets.json). Allow additional space for dependencies, downloaded weights and installer work files.

```bash
python scripts/install_assets.py \
  --url "https://github.com/cjk041212/ShopLens/releases/download/runtime-v1/shoplens-runtime-seed42-v1.tar.gz" \
  --sha256 2f1fc93175adedca6ffb987ecc42f50fcd4a0959679d187aa4c81b7a8ad904c9
python scripts/prepare_studio_models.py --root . --include-clip
python scripts/check_install.py --root . --require-weights
```

The preparation helper explicitly downloads official pinned **Qwen/Qwen3-Embedding-0.6B** and **openai/clip-vit-base-patch32** files. Searching does not download weights automatically or require an external inference API key. For an already prepared supported cache, use `--offline` on the preparation helper. A downloaded runtime archive can instead be installed with `--archive /actual/path/to/archive.tar.gz` and the same SHA256. Source, runtime assets and base weights are separate installation parts.

Start the app with the Streamlit command above. Prices are historical USD metadata; unknown prices are excluded by budget filters. Avoid terms are soft preferences. Similarity and input-removal views are descriptive, not causal explanations.

## Experiment and offline figures

The recorded study has **27 complete pipeline runs**: three domains × three objectives × seeds 42/43/44. The domains share one Amazon Reviews 2014 source. **Frozen has the highest mean held-out NDCG@10 in all three domains**; these runs do not establish a consistent LoRA improvement. Frozen keeps Qwen weights fixed while training the recommendation head.

Formal metrics evaluate full-catalog history-to-item recommendation. Studio free-text retrieval is an exploratory interaction, and changed rankings do not prove improved relevance. Error bars show sample standard deviations across three seeds, not confidence intervals. Training views show the two measured adaptation validation points; checkpoints use validation, never test scores.

Generate the portable seven-figure gallery and its exact data:

```bash
python scripts/export_experiment_gallery.py --output dist/experiment-gallery
```

Open `dist/experiment-gallery/index.html`. Plotly is embedded for offline use; keep companion CSV/JSON files beside the HTML to retain its download links. The figure pack is optional and can be generated locally; **runtime-v1** is the separate shopping runtime with its original checksum. Neither installing the demo assets nor exporting charts reruns the historical study.

## Report and presentation

- [English experiment report](reports/ShopLens_Qwen_report_en.md)
- [PowerPoint presentation](reports/ShopLens_Qwen_presentation.pptx)
- Recorded results: [JSON](results/qwen/results_qwen_summary.json) · [CSV](results/qwen/results_qwen_summary.csv)

See [ATTRIBUTION.md](ATTRIBUTION.md) for dataset/model sources, upstream credit and rights status.
