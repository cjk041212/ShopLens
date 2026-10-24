# ShopLens Shopping Studio

Shopping Studio is a local, task-oriented interface: describe a need, compare three model checkpoints immediately, optionally refine real candidates with images and feedback, and save a decision. No liked-product history is required for a text request. The existing six-page experiment interface remains available under **Workspace → Research lab**.

## Prepare and run

Use the existing project environment and prepared catalogs. Explicitly download the public pinned encoders once:

```bash
python scripts/prepare_studio_models.py --root . --include-clip
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

If the weights are already cached, use `--offline` to fingerprint them without network access. Studio uses `data/studio_hf_cache` and `data/studio_models/manifest.json`; it does not replace formal model provenance or experiment exports. The code archive excludes model weights, raw data and feature caches.

The model selector also requires the existing completed `artifacts/qwen/<domain>/<mode>/seed42/` exports for all three objectives. Studio checks the selected head, serving/candidate caches, completion record and adapter provenance before using them. Preparing the frozen query encoders does not download or recreate these trained exports. A missing or inconsistent model is shown as unavailable; another model is never silently substituted.

## Try a complete task

1. Select **Sports & Outdoors** and **Shopping Studio**.
2. Enter: “Find a yoga mat for practicing at home, under $40. Avoid pink.” Click **Interpret brief**.
3. Review the editable search request, soft avoid preferences and historical price ceiling. Optionally narrow **Source catalog category**, for example `Sports & Outdoors / Exercise & Fitness / Yoga / Mats`. This matches original metadata, which can be inaccurate; it is not an automatically verified product type. Interpretation uses transparent English keyword rules. Qwen supplies semantic retrieval; it does not generate a chat response or guarantee that every phrase was parsed.
4. Optionally open **Add a visual reference**. Upload a JPG, PNG or WebP photo, or choose an explicitly labeled catalog example. Adjust the text/image weight, then click **Find products**.
5. Switch between **Frozen Qwen**, **Semantic alignment LoRA** and **User preference LoRA** directly. No **More like this** selection is required. Open **Compare models** to see the three result lists, overlap and rank changes relative to Frozen Qwen. The request, photo, constraints and feedback stay fixed; each model supplies its own request encoding and product-text space. The resulting semantic candidate pools can differ across models.
6. Optionally use **More like this**, **Less like this** and **Hide this**. Positive feedback adds the selected checkpoint's trained history recommender; negative feedback and exact hiding refine the candidates. Feedback reranks with existing query vectors. Reference products can be managed, undone and shortlisted in **Your feedback**. Switching models reuses all three encoded requests.
7. Save two or three candidates to **Your shortlist**. Compare the actual photos, brand, historical price, category and original description.
8. Choose a **Final choice**, confirm it, then download the shortlist as JSON or CSV, or save a standalone HTML result card with embedded catalog photos. The model identity and ranking provenance accompany the exported decision.

Changing the request, constraints, image or text/image weight hides stale recommendations until **Find products** runs again. Changing only the ranking model updates the selected results immediately without encoding the text or image again. Shortlists, feedback and the selected model are separated by collection and remain in the current session. Download them to retain them after the session ends. Returning to Research lab does not overwrite the original positive-preference history.

## What the system computes

- A pinned Qwen3-Embedding-0.6B base with the selected domain-specific adapters encodes the search request three ways: adapters disabled for Frozen Qwen, the selected semantic adapter for Semantic alignment, and the selected preference adapter for User preference. Each request vector matches that same checkpoint's exported 1024-dimensional product-text vectors. Soft-avoid text uses the corresponding encoder too. All three use the same shopping retrieval instruction, left padding, last-valid-token pooling, 256-token truncation and L2 normalization. Frozen requests are not compared directly with adapted product vectors.
- The pinned local CLIP model encodes reference image bytes in 512 dimensions using EXIF correction and the catalog preprocessing. Uploaded images are processed in memory, never persisted by Studio or sent to an external service. The UI accepts single-image JPG/PNG/WebP references up to 8 MB and 16 megapixels.
- Each text model first retrieves its own top 200 text-similar candidates; image-only queries use the same CLIP image similarities, retaining boundary ties. Image and feedback signals rerank only the respective pool. Budgets and exclusions can leave fewer results, rather than promoting products from outside the pool. Different model pools are part of this request-retrieval comparison, not a controlled common-pool history-head ablation. This boundary does not guarantee semantic relevance.
- Separate full-catalog similarity percentiles are blended across available text and image signals. Positive examples contribute content similarity and the selected model's learned user-network score. These two feedback percentiles have equal weight; the resulting feedback contributes 30% when a query is available. Negative examples and soft avoid preferences apply disclosed penalties. Scores are Studio ranking values, not probabilities, causal importance, or the scores from formal Qwen tests. Raw scores should not be compared across different ranking models as evidence of model quality.
- Positive, negative and hidden reference products are excluded from subsequent suggestions. **Hide** removes an exact product; **Less like this** additionally penalizes similar candidates. Shortlisting by itself supplies no training feedback.
- Budget filtering uses recorded historical metadata prices and excludes unknown prices. These are old catalog values, not current offers. Colors, compatibility, dimensions, stock and suitability are not guaranteed merely because a phrase appears in a description.

## Scientific scope

The three model choices use the actual selected **seed-42** checkpoints for the current collection. The direct request path loads the pinned local base and the selected adapters, then compares each encoded request with that checkpoint's cached product-text vectors. Switching adapters is serialized with encoding so concurrent sessions cannot mix model states. Missing or inconsistent adapters produce an error, without a fallback pretending to be the requested model. No model is retrained during an interaction.

The formal experiment is asymmetric: histories use base Qwen text plus CLIP, while the candidate tower may use LoRA-adapted text. Studio preserves this geometry when scoring **More like this** products, using at most the latest 20 positive preferences in the same order as the existing inference backend. Negative preferences remain a disclosed Studio penalty and are not passed off as a new trained negative-history model.

Free-text shopping requests were not the formal history ranker's trained input. Direct request retrieval is therefore an explicitly **exploratory extension**: it evaluates the adapted language encoders as request-to-product retrievers, with a common shopping instruction. It does not insert a 1024-dimensional request into the trained 128-dimensional user network. Optional real product feedback still uses the original trained history geometry.

The formal semantic objective matched frozen review anchors to adapted product documents, while the preference objective trained candidates through behavioral ranking. Neither objective guarantees that adapting the request encoder improves free-text retrieval. Three models may return some or all of the same products. Displayed rank changes are actual observations, not a new relevance benchmark, a causal explanation, or proof of improved quality. Image-only searches with no positive preferences share CLIP and can be identical across all three models.

Studio does not change any formal test result. Product-text excerpts and computed similarities provide inspectable evidence rather than generated product claims. The product atlas, model comparison, preference-removal diagnostics and seed-level evaluation charts remain in Research lab.
