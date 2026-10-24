# Attribution and rights status

This file credits upstream models, data, and software. **No project-wide software license is declared in this distribution.** This attribution file does not supply a new license, an ownership claim, or third-party redistribution permission. Check the source owner's terms and each upstream provider's applicable terms before using or redistributing code, weights, adapters, descriptions, or photographs.

## Product data

Prepared catalogs use the Baby, Sports & Outdoors, and Clothing, Shoes & Jewelry domains of **Amazon Reviews 2014**, from the UCSD/McAuley dataset collection:

- [Amazon dataset source](https://mcauleylab.ucsd.edu/public_datasets/data/amazon/datasets.html)

These are three domains of one source. Titles, descriptions, brands, categories, historical prices, and photos remain third-party catalog content. A local photograph or metadata record is not newly authored ShopLens content. Attribution, a downloaded copy, or a checksum manifest does not state that redistribution permission has been obtained.

The repository tracks the selected runtime catalog/photo payload as bounded data chunks under `assets/runtime/`, with selected heads and adapters under `models/`. Its manifest records every restored path, checksum, scope and transformation. The local installer reconstructs catalogs, photos, aligned caches and selected exports without downloading a Release package. It excludes raw reviews, chronological events/user histories, user identifiers, private course documents, credentials, and base weights. Distribution of that runtime payload remains subject to the applicable source terms.

## Models

| Component | Official source | Identity used by the demo |
| --- | --- | --- |
| Qwen3-Embedding-0.6B | [Qwen model repository](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3) | Qwen/Qwen3-Embedding-0.6B, revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 |
| CLIP ViT-B/32 | [OpenAI model repository](https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268), [original implementation](https://github.com/openai/CLIP), [paper](https://proceedings.mlr.press/v139/radford21a.html) | openai/clip-vit-base-patch32, revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 |
| MiniLM retained baseline | [Sentence Transformers repository](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | sentence-transformers/all-MiniLM-L6-v2 |

The explicit preparation helper obtains pinned Qwen/CLIP weights directly from upstream. Base weights are not part of the tracked runtime assets. LoRA adapters and fitted heads are experiment outputs derived using the declared base models; their hashes provide provenance, not replacement upstream terms. No affiliation with or endorsement by upstream providers is claimed.

## Software and methods

ShopLens uses PyTorch, Hugging Face Transformers/PEFT/Accelerate/Hub, Sentence Transformers, NumPy, SciPy, scikit-learn, Pillow, pandas, Streamlit, Plotly, and their dependencies. Runtime/development versions are declared in the dependency files; those packages retain their upstream terms.

The recommendation work includes BPR and content/history-based ranking; the BPR reference is [Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618). Exploratory natural-language request retrieval is kept separate from the formal history-based benchmark. References and model size are not measured evidence of a performance improvement.

