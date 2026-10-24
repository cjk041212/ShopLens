# ShopLens

**Describe a shopping need, compare three text models, and build a shortlist from real catalog evidence.**

ShopLens is a local Streamlit application for text-led, image-assisted shopping. Shopping Studio accepts a natural-language request and an optional reference photo, retrieves products with **Frozen Qwen**, **Semantic alignment LoRA**, and **User preference LoRA**, and lets you refine the results with feedback. A separate Research lab contains product maps, ranking comparisons, input-removal diagnostics, and recorded experiment views.

Repository: [cjk041212/ShopLens](https://github.com/cjk041212/ShopLens).

## What you can do

- Search Baby, Sports & Outdoors, or Clothing & Accessories with a request and optional image.
- Switch objectives without losing the request, reference image, feedback, shortlist, or final choice.
- Use **More like this**, **Less like this**, and **Hide this**, then undo feedback when needed.
- Compare original descriptions, photos, brands, categories, and known historical prices.
- Export your shortlist and final choice as JSON, CSV, or a self-contained HTML result card.

Each objective uses its own pinned Frozen/LoRA query encoder and corresponding candidate text vectors, so its candidate pool can differ. CLIP image features are shared. Positive references can additionally activate each objective's trained history/candidate towers and user network. This request retrieval is an **exploratory extension**, separate from the formal history-based recommendation evaluation; no quality or performance improvement is assumed.

The prepared catalog contains **47,801 products**: Baby **7,037**, Sports & Outdoors **18,273**, and Clothing & Accessories **22,491**. Compare models displays the first three recommendations per objective; Discover normally displays twelve results. These are display limits, not catalog sizes.

## Three parts to install

| Part | Delivery | Purpose |
| --- | --- | --- |
| Source | This repository | App, reusable model code, tests, and preparation/install helpers |
| Runtime assets | Separate checksum-verified archive | Three catalogs and photos, aligned caches, nine selected seed42 exports, six selected adapters, and compact recorded results |
| Base weights | Explicit official pinned Hugging Face downloads | Local request and image encoding |

**Cloning the source does not install catalogs, trained exports, or model weights.** The declared runtime payload is about 1.52 GB before compression; weights and Python packages need additional disk space. Runtime assets exclude raw reviews, interaction histories, user identifiers, private course materials, credentials, and base model weights. They are a demo installation, not a complete training-data or checkpoint archive.

Runtime assets are published in [runtime-v1](https://github.com/cjk041212/ShopLens/releases/tag/runtime-v1). The archive is **1,457,809,593 bytes** and its GitHub-reported SHA256 matches the identity in [runtime-assets.json](runtime-assets.json). The source and runtime package are separate downloads; base encoders are prepared explicitly from official pinned sources. See [INSTALL.md](INSTALL.md) for installation and environment limits.

Use **Python 3.12**. Exact pins reflect the exercised Python 3.12.14/macOS arm64 environment. NumPy 2.5.4 and SciPy 1.18.1 require Python 3.12, so this pinned recipe is not compatible with Python 3.11.

Obtain the source, then create an isolated environment:

~~~bash
git clone https://github.com/cjk041212/ShopLens.git
cd ShopLens
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
~~~

Install the published runtime archive with checksum verification:

~~~bash
python scripts/install_assets.py \
  --url "https://github.com/cjk041212/ShopLens/releases/download/runtime-v1/shoplens-runtime-seed42-v1.tar.gz" \
  --sha256 2f1fc93175adedca6ffb987ecc42f50fcd4a0959679d187aa4c81b7a8ad904c9
~~~

Prepare the base weights explicitly, check the installation, and start the app:

~~~bash
python scripts/prepare_studio_models.py --root . --include-clip
python scripts/check_install.py --root . --require-weights
python -m streamlit run app.py \
  --server.address 127.0.0.1 --server.port 8501 \
  --browser.gatherUsageStats false
~~~

Open **http://127.0.0.1:8501/**. The server listens on loopback. Searching does not download weights automatically or require an external inference API key.

## Try a complete task

1. Describe one product you want, for example: “Find a yoga mat for practicing at home; avoid pink.”
2. Select **Interpret brief** to review the transparent rule-based request, soft avoid terms, and historical budget. Edit them as needed.
3. Optionally upload a JPG/PNG/WebP or choose a catalog reference photo, then select **Find products**.
4. Switch objectives and inspect **Compare models**. Refine the lists and shortlist two or three products.
5. Compare source evidence, confirm a final choice, and save the result.

Prices are historical USD catalog metadata, not live offers. Budgets exclude unknown prices. Avoid terms are soft ranking preferences. Source-category filters follow metadata that can be incomplete or wrong. The app does not guarantee stock, suitability, current prices, or unsupported attributes. Similarity and removal diagnostics are descriptive evidence, not causal attribution.

## Evidence and reproducibility

The original research comparison contained **27 formal records**: three domains × three objectives × three full-pipeline seeds. Frozen had the highest mean NDCG@10 among the three objectives in each domain; the runs do not establish a consistent LoRA gain. All three domains come from the same Amazon 2014 source.

The runtime package selects seed42 for interaction and includes compact recorded aggregates. It does not contain every seed's complete checkpoints, raw reviews, or events. Training scripts need separately prepared data and compute; installing demo assets does not reproduce the 27 runs. Keep exploratory request-retrieval results separate from the formal history-based benchmark.

For development:

~~~bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
~~~

Tests use explicitly labeled synthetic fixtures where appropriate; fixture success is not evidence of recommendation quality. See [INSTALL.md](INSTALL.md) for environment limits, offline preparation, and troubleshooting, and [ATTRIBUTION.md](ATTRIBUTION.md) for upstream credit and rights status.


## Coursework materials

The English deck contains **26 slides: 18 main slides and 8 optional appendix slides**, including **15 real application screenshots**. The matching human narration has 848 spoken words; reserve time within the course video limit for actual UI operation and a brief Python walkthrough.

See [the Chinese submission checklist](提交清单.md) and [submission/](submission/) for public copies of the report, slides, narration, results, and UI evidence. The coursework source is the repository itself; use **Code → Download ZIP** for a source archive. Runtime assets and base weights remain separate installation steps. Video, signed group contributions, and the official AI declaration still require the group to complete them.
