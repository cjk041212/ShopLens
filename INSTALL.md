# Installation and operating notes

## Tested scope

The exercised local environment is **Python 3.12.14 on macOS arm64**, including an Apple M4 with 16 GB memory. CPU request encoding, selected LoRA switching, CLIP image encoding, and the Streamlit workflow have been exercised there. This does not establish a latency or memory guarantee for another machine.

The source has a Linux CPU path. A fresh Linux CPU installation is not established by the macOS run; these pins are not a platform-independent binary lock. **Windows is untested.** MPS is an opt-in code path, not the default or a claim of equivalent behavior.

Python 3.11 cannot install the exact recipe unchanged: measured NumPy 2.5.4 and SciPy 1.18.1 declare Python >=3.12. Use Python 3.12 rather than silently selecting different numerical-library versions.

## 1. Install dependencies

Run from the checkout root, in an isolated environment:

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip check
~~~

Runtime requirements explicitly include **peft** and **accelerate**, needed for actual LoRA switching. Installing only an older baseline dependency list is insufficient. Hugging Face Hub, Tokenizers, and Safetensors are pinned with Transformers to retain the exercised loading combination. Tests are separate in requirements-dev.txt; pytest 9 includes the subtest fixture used here.

On Linux CPU, select an appropriate official CPU build of PyTorch before the remaining requirements. For the recorded version, the intended command is:

~~~bash
python -m pip install "torch==2.14.1" \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
~~~

Use this only if the official index supplies that wheel for the chosen Python/architecture. CPU-index availability and a fresh Linux runtime are not asserted by the macOS inventory. General-index Linux PyTorch packages may pull CUDA runtime dependencies; the demo does not require a GPU. Do not silently substitute an untested PyTorch/Transformers/PEFT combination when reproducing an issue.

## 2. Install runtime assets

The source excludes large catalogs, photos, features, checkpoints, and base weights. The separate tar.gz archive has payload paths under runtime/ and an internal **runtime/manifest.json**, listing per-file paths, sizes, and SHA256. Its external publisher manifest supplies the **archive SHA256**; the archive cannot declare its own checksum.

The declared runtime scope contains:

- Three Amazon 2014 catalogs and local product photos.
- Aligned Qwen base and original CLIP-provenance feature caches.
- Frozen, Semantic alignment, and User preference seed42 exports in all domains, including six selected adapters.
- Compact recorded aggregate/evidence files needed by the demo.

It excludes raw reviews, chronological events/user histories, user identifiers, private course materials, credentials, base Qwen/CLIP weights, and other seeds' complete training checkpoints. The declared uncompressed payload is about **1.52 GB**. Allow additional space for the archive, Python environment, model downloads, and installer work files.

The actual archive and external manifest are available from [runtime-v1](https://github.com/cjk041212/ShopLens/releases/tag/runtime-v1). Install the public archive directly:

~~~bash
python scripts/install_assets.py \
  --url "https://github.com/cjk041212/ShopLens/releases/download/runtime-v1/shoplens-runtime-seed42-v1.tar.gz" \
  --sha256 2f1fc93175adedca6ffb987ecc42f50fcd4a0959679d187aa4c81b7a8ad904c9
~~~

For a separately downloaded local copy, use `--archive "/actual/path/to/shoplens-runtime-seed42-v1.tar.gz"` instead of `--url`. Add `--dry-run` to validate without installing payload files. Both modes require the archive checksum. The installer validates the complete archive and internal per-file manifest before installing; a mismatch is an error. Use `--root TARGET` only to install intentionally into another project copy.

The release API's archive size and digest were checked against the actual local package. Its external manifest is a companion inventory, not the payload itself. Review [ATTRIBUTION.md](ATTRIBUTION.md) and upstream terms before using or redistributing third-party payloads. Checksums establish integrity, not permission.

## 3. Prepare pinned base encoders

This is the explicit network step for public model files:

~~~bash
python scripts/prepare_studio_models.py --root . --include-clip
~~~

The helper uses official pinned repositories without an inference-service API key:

| Model | Pinned revision |
| --- | --- |
| Qwen/Qwen3-Embedding-0.6B | 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 |
| openai/clip-vit-base-patch32 | 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 |

Downloads go into data/studio_hf_cache/; the helper records prepared file hashes under data/studio_models/. Qwen must be present in the project cache. CLIP may also be discovered in the normal Hugging Face cache. Existing public files are reused when the pinned-file checks permit it.

For a machine whose pinned files are already in a supported local cache:

~~~bash
python scripts/prepare_studio_models.py --root . --include-clip --offline
~~~

Offline mode neither creates missing weights nor downloads them. Selected adapters come from the runtime package, not an unpinned remote adapter repository. Search does not fetch models in the background.

## 4. Check and start

~~~bash
python scripts/check_install.py --root . --require-weights
python -m streamlit run app.py \
  --server.address 127.0.0.1 --server.port 8501 \
  --browser.gatherUsageStats false
~~~

Open http://127.0.0.1:8501/. The server binds to loopback. No cloud inference endpoint, cluster connection, or scheduled job is needed.

The checker is read-only: it inspects real catalogs, selected exports, photo paths, and pinned-weight preparation. It does not load a language model, encode a new query, download files, train, or measure recommendation quality. Missing base weights are warnings by default; --require-weights makes them fatal. --json produces machine-readable results.

Studio encodes the request and optional image on **Find products**. Switching cached objective results does not re-encode. Feedback, shortlists, and uploaded reference bytes are session state; uploads are not written into project files. Changed adapters or requests require fresh encoding rather than mixing old queries and new candidates.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Missing peft or accelerate | Activate the project environment, install current requirements.txt, then run pip check. |
| Python 3.11 dependency failure | Recreate the environment with Python 3.12; the recorded numerical pins require it. |
| Missing catalog, exports, or adapters | Install the actual runtime archive. A source checkout or manifest alone has no payload. |
| Base model not prepared | Run the explicit helper with --include-clip and inspect its error. Offline mode needs existing pinned files. |
| Archive/model hash mismatch | Compare against the real publisher manifest/pinned source; do not edit hashes to make checks pass. |
| Model/feature/catalog provenance mismatch | Install one coherent asset package; mixed catalogs, feature orders, and exports are unsupported. |
| A new request shows a refresh message | Select Find products; edits or changed encoding/artifacts invalidate older results. |
| Historical budget returns few products | Unknown prices are excluded; the app does not fetch current offers. |
| Objectives show identical lists | This is valid, especially for shared image-only inputs; model use does not guarantee a change or gain. |
| Port 8501 is busy | Choose another local --server.port and use that browser URL. |

## Development and experiments

~~~bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
~~~

Tests include labeled synthetic fixtures, not actual training or product-quality evidence. Runtime assets contain seed42 demo exports and compact aggregates, not every checkpoint, raw review, and event needed for the historical 27-run matrix. Training/reproduction scripts require separately prepared data and an appropriate environment; a demo installation is not a training job or resource allocation.

