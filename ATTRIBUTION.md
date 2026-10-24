# Attribution and rights status

This file credits upstream models, data, and software. **No project-wide software license is declared in this distribution.** This attribution file does not supply a new license, an ownership claim, or third-party redistribution permission. Check the source owner's terms and each upstream provider's applicable terms before using or redistributing code, weights, adapters, descriptions, or photographs.

## Product data

Prepared catalogs use the Baby, Sports & Outdoors, and Clothing, Shoes & Jewelry domains of **Amazon Reviews 2014**, from the UCSD/McAuley dataset collection:

- [Amazon dataset source](https://mcauleylab.ucsd.edu/public_datasets/data/amazon/datasets.html)

These are three domains of one source. Titles, descriptions, brands, categories, historical prices, and photos remain third-party catalog content. A local photograph or metadata record is not newly authored ShopLens content. Attribution, a downloaded copy, or a checksum manifest does not state that redistribution permission has been obtained.

The repository tracks the selected runtime catalog/photo payload as bounded data chunks under `assets/runtime/`, with selected heads and adapters under `models/`. Its manifest records every restored path, checksum, scope and transformation. The local installer reconstructs catalogs, photos, aligned caches and selected exports without downloading a Release package. It excludes raw reviews, chronological events/user histories, user identifiers, private course documents, credentials, and base weights. Distribution of that runtime payload remains subject to the applicable source terms.

## Additional benchmark sources

- **Amazon Shopping Queries / ESCI:** [official dataset and ranking code](https://github.com/amazon-science/esci-data/tree/7916cdf6ab75a462e77f20ab40428a10923998d5). Its README declares Apache-2.0. The bounded public input subset retains official query/product identifiers, product titles and E/S/C/I judgments; source and selected-file hashes accompany the results. No product images are supplied by this benchmark. The declared primary gains follow `ranking/train.py`; the reversed S/C mapping in the qrel-export script is preserved as a separate sensitivity.
- **FashionIQ:** [official annotations and research dataset](https://github.com/XiaoxiaoGuo/fashion-iq/tree/f71b16c3b09ef03b02c0fed160fa072fb3b8aa16), [official-linked image URL metadata](https://github.com/hongwang600/fashion-iq-metadata/tree/3bb39d7d9a024d92f1b0fd0929b38d73dff24a0a). The annotation README names a Community Data License Agreement (CDLA) and links cdla.io without specifying its variant/version; no particular CDLA variant is asserted here. Captions, paired product identifiers and photographs are third-party dataset content. The portable subset preserves source revisions, per-image URLs and hashes, split membership and unavailable-image records in `assets/benchmarks/fashion/manifest.json`. Its trained fusion projection and frozen feature cache are derived experiment outputs. These subset files do not supply a new photograph license or claim provider endorsement; upstream dataset and content terms continue to apply.

Benchmark labels were already supplied by the dataset authors; this extension collected no new human annotations. User-history diagnostics publish aggregates only. New measurements are separated from the original Amazon Reviews 2014 experiment and do not establish online purchase, satisfaction or fairness outcomes.

## Models and derived checkpoints

| Component | Official source | Identity used by the demo |
| --- | --- | --- |
| Qwen3-Embedding-0.6B | [Qwen model repository](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3) | Qwen/Qwen3-Embedding-0.6B, revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 |
| CLIP ViT-B/32 | [OpenAI model repository](https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268), [original implementation](https://github.com/openai/CLIP), [paper](https://proceedings.mlr.press/v139/radford21a.html) | openai/clip-vit-base-patch32, revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 |
| MiniLM retained baseline | [Sentence Transformers repository](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | sentence-transformers/all-MiniLM-L6-v2 |

The explicit preparation helper obtains pinned Qwen/CLIP weights directly from upstream. Base weights are not part of the tracked runtime assets. LoRA adapters and fitted heads are experiment outputs derived using the declared base models; their hashes provide provenance, not replacement upstream terms. No affiliation with or endorsement by upstream providers is claimed.

## Software and methods

ShopLens uses PyTorch, Hugging Face Transformers/PEFT/Accelerate/Hub, Sentence Transformers, NumPy, SciPy, scikit-learn, Pillow, pandas, Streamlit, Plotly, and their dependencies. Runtime/development versions are declared in the dependency files; those packages retain their upstream terms.

The recommendation work includes BPR and content/history-based ranking; the BPR reference is [Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618). Exploratory natural-language request retrieval is kept separate from the formal history-based benchmark. References and model size are not measured evidence of a performance improvement.

