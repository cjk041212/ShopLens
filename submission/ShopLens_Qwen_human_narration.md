# ShopLens: English human narration

**Source paths (reference only, not spoken).** Source path convention: Resolve src/..., submission/..., data/..., and artifacts/... from the repository root. Resolve assets/... relative to submission/. Runtime data and artifacts require a separate runtime asset installation.

Main presentation: 18 slides, approximately 848 spoken words.
At 115–125 words per minute, narration takes approximately 6.8–7.4 minutes. Reserve the remaining time within a 10-minute total for transitions, about 60 seconds of actual UI operation and 30 seconds of code explanation. Rehearse to 9:30.
Appendix slides 19–26 support questions and are outside the timed narration. Do not read the source links or discussion notes aloud.
A real team member must record the presentation. Complete genuine names and contributions before submission.

## 1. ShopLens

ShopLens combines a controlled recommendation study with an interactive shopping interface. We compare frozen Qwen with semantic alignment and user preference LoRA. I will show the actual offline results, then a complete local workflow from a written request to a saved decision.

**Sources**

- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
- https://arxiv.org/abs/2106.09685
- https://proceedings.mlr.press/v139/radford21a.html

## 2. Semantic alignment and user preference

The central question separates semantic matching from preference learning. Semantic LoRA matches training reviews with product descriptions. Preference LoRA learns directly from ranking loss. Every arm trains its recommendation head, including Frozen. The image encoder stays fixed, so the controlled comparison concerns the language objective.

**Discussion notes**

Semantic InfoNCE uses frozen base-Qwen training-review anchors and live candidate descriptions, with multiple positives for the same item. Its BPR text input is detached. Preference training never uses the target review as its recommendation query.

**Sources**

- src/shoplens/qwen_pipeline.py
- src/shoplens/qwen_models.py
- https://arxiv.org/abs/1205.2618

## 3. Three Amazon domains and one ranking protocol

The data contains real product text, reviews and photographs across Baby, Sports and Clothing. We split chronologically and use only strictly earlier positive history. Ranking covers the full prepared catalogue, excluding previous positives. Metrics first average targets within users. These are three domains from one Amazon source, which the instructor still needs to accept as the dataset requirement.

**Discussion notes**

Global timestamp quantiles retain ties in the earlier split. Evaluation is prequential: earlier held-out observations may enter later histories while weights remain fixed. Repeated targets already in history do not enter evaluation. A stable item-ID rule breaks score ties. Positive interactions have ratings of four or five stars.

**Sources**

- https://mcauleylab.ucsd.edu/public_datasets/data/amazon/datasets.html
- src/shoplens/qwen_evaluate.py
- data/processed/baby/audit.json
- data/processed/sports/audit.json
- data/processed/clothing/audit.json

## 4. History and candidate representations

The formal model uses an asymmetric architecture. Histories always enter through frozen base text and fixed images. Candidate text changes across objectives. Separate towers produce compact vectors. A user network pools history and scores candidates by a dot product. Adapted candidate vectors must never replace the frozen history inputs.

**Sources**

- src/shoplens/qwen_models.py
- src/shoplens/qwen_inference.py

## 5. Shared initialization and adaptation budget

Each domain and seed first selects a common warmup head. The three arms then share thirty thousand matched training cases, their order and sampled negatives for two adaptation epochs. Validation selects epoch one or two. Three complete pipeline seeds give twenty-seven formal test records. Test scores never choose the checkpoint.

**Discussion notes**

Warmup uses all eligible cases, at most 20 epochs and validation early stopping. Adaptation batch size is 32. Head learning rate is 1e-4 and LoRA learning rate is 2e-5. LoRA rank 8, alpha 16, dropout 0.05, q_proj/v_proj. Epoch 0 cannot replace a poor adaptation checkpoint.

**Sources**

- src/shoplens/qwen_data.py
- src/shoplens/qwen_pipeline.py
- submission/results_qwen_summary.json

## 6. Formal test results

These bars come from all twenty-seven audited test records after validation-only selection. Frozen has the highest observed mean in each domain. Semantic alignment loses to Frozen in seven of nine matched comparisons. Preference LoRA loses in six of nine. The experiment does not support a general improvement from adaptation under this budget.

**Discussion notes**

Means and sample standard deviations appear in Appendix 1. A matched win is a positive difference for the same domain and seed. These measurements describe the declared offline protocol, not purchasing conversion.

**Sources**

- submission/results_qwen_summary.json
- artifacts/qwen/baby/results.json
- artifacts/qwen/sports/results.json
- artifacts/qwen/clothing/results.json

## 7. Seed variation in the Research lab

The Research lab keeps the matched seed differences visible. We inspect paired changes against Frozen rather than selecting the best run. Three seeds provide limited evidence about variability. A positive bar is an observed difference, not statistical significance. The negative results stay in the report and interface.

**Sources**

- assets/research_paired.png
- src/shoplens/visual_charts.py
- src/shoplens/ui.py
- submission/results_qwen_summary.json

## 8. A shopping request without a liked-product history

Studio starts with a written request, here a compact stroller for everyday outings. Interpretation uses transparent rules and leaves editable fields. The user reviews omitted nuance before searching. Qwen retrieves products after Find products. Positive feedback is optional.

**Sources**

- assets/studio_brief.png
- src/shoplens/studio_ui.py
- src/shoplens/studio.py

## 9. The same request across three trained models

The same stroller request runs through all three selected checkpoints. Each arm encodes the query in the same space as its candidate text vectors. The comparison shows actual products, overlap and observed ranks. The respective candidate pools can differ. This is exploratory request retrieval, separate from the controlled history-based test.

**Discussion notes**

The example uses no positive feedback. Current Studio retrieves each model's own top 200 text-similar candidates, including boundary ties. Missing Top-K ranks stay unknown. Raw Studio scores are not compared across models.

**Sources**

- assets/studio_models.jpg
- src/shoplens/studio_encoders.py
- src/shoplens/studio_models.py
- src/shoplens/studio_ui.py

## 10. Visual reference and explicit constraints

This later state adds a catalogue photo, soft avoidance of bulky products and a historical price ceiling of one hundred and fifty dollars. Frozen CLIP supplies the image signal. Uploads stay in local memory. Filters can leave fewer results without expanding the retrieved pool.

**Discussion notes**

This screenshot shows a later state than the earlier zero-image, unrestricted comparison. Actual selected reference is Capri Stroller - Red Tech and text weight is 0.65. The image source is Use catalog example, not an uploaded file. Current UI limits uploads to single JPG/PNG/WebP, 8 MB and 16 megapixels. Budget filtering excludes unknown prices. Source metadata categories are not verified product types. An avoid penalty does not guarantee perfect semantic exclusion.

**Sources**

- assets/studio_visual_filters.png
- src/shoplens/studio_encoders.py
- src/shoplens/studio.py

## 11. Feedback, exact hiding and undo

More like this adds positive evidence and activates the selected model’s trained history recommender. Less like this penalizes similar products. Hide removes an exact product. All reference products leave subsequent suggestions. The feedback manager exposes the current state and Undo. These actions rerank products using cached request vectors rather than retraining a model.

**Discussion notes**

At most the latest 20 distinct positive products enter the history network. Shortlisting alone gives no training feedback. Feedback and query percentiles follow a disclosed application policy rather than the formal evaluation score.

**Sources**

- assets/studio_feedback.png
- src/shoplens/studio_ui.py
- src/shoplens/studio.py
- src/shoplens/studio_models.py

## 12. A shortlist with original product evidence

The shortlist compares a few real candidates through photographs, historical prices, categories and original descriptions. Source details can reveal information that embeddings miss. The user makes the final choice. Saving a shortlist item supplies no preference feedback.

**Sources**

- assets/studio_shortlist.png
- src/shoplens/studio_ui.py

## 13. A final choice with portable exports

After confirming a final choice, the user downloads the decision. JSON retains context and model provenance. CSV provides a simple shortlist. HTML includes catalogue photos in a standalone card. Exporting preserves the record after the session ends. The app places no order.

**Sources**

- assets/studio_export.png
- src/shoplens/studio.py
- src/shoplens/studio_ui.py

## 14. Product atlas with a real product inspector

Product atlas places catalogue products, selected preferences and recommendations on one fixed PCA map. Clicking a point opens its photo, title and product ID and can add a preference. Switching models changes highlights while keeping coordinates fixed. The two-dimensional view loses information.

**Discussion notes**

This earlier actual Baby capture shows the complete map and inspector together. The displayed capture does not identify all history items, so it is not treated as the same session as the fresh Research figures. The fixed PCA uses the Frozen candidate tower. The plotted sample is not the ranking candidate set. Ranking still uses the full eligible catalogue. The 2D variance fraction is shown, and proximity is a lossy projection rather than a causal explanation.

**Sources**

- assets/research_atlas_complete.jpg
- src/shoplens/visual_data.py
- src/shoplens/visual_ui.py

## 15. Preference relationships in the actual tower

Preference lab shows how selected history products relate to recommended candidates in the actual one-hundred-and-twenty-eight-dimensional towers. The network retains positive and negative cosine links. A heatmap provides the same numerical relationship in a different form. These links are neither attention weights nor causal contributions to the final ranking.

**Sources**

- assets/research_network.png
- src/shoplens/visual_ui.py
- src/shoplens/visual_charts.py

## 16. Observed rank flow across objectives

Rank flow follows products across the three model lists for a fixed history. The adjacent view reports Top-K Jaccard overlap. Lines break when products leave the displayed list. We never invent their missing ranks. These views describe disagreement, rather than which list is correct.

**Sources**

- assets/research_rankflow.png
- src/shoplens/visual_charts.py
- src/shoplens/ui.py

## 17. Removal sensitivity for the current history

The removal experiment drops one active history item at a time and reruns the recommender. It measures score changes and retention of the original Top-K. It never backfills older history. Original selected products stay excluded. This is session sensitivity, rather than causal attribution.

**Discussion notes**

This earlier actual Baby capture names Peg Perego Aria Twin, Capri Stroller Red Tech and Prince Lionheart Stroller Connectors. The model selector lies outside the capture. No numeric comparison with another capture or model is inferred. The result is conditional on its current history and selected seed42 model. It is neither a formal held-out metric nor a causal attribution. Empty and single-item histories do not invent a removal experiment.

**Sources**

- assets/research_removal_complete.jpg
- src/shoplens/visual_data.py
- src/shoplens/visual_ui.py

## 18. Findings and limits

Our main finding is negative: neither adaptation objective improves average NDCG at ten in any domain under this budget. The interface makes the trained models inspectable and supports a complete decision workflow, but attractive examples cannot replace relevance evaluation. Three seeds, one Amazon source and static metadata limit the evidence. The references are the original Amazon archive, official Qwen documentation, LoRA, CLIP and BPR.

**Discussion notes**

Earlier MiniLM baselines remain separate because their architecture and training budget differ. Clothing live CUDA LoRA training used MATH SDPA after a reproduced gradient failure, while earlier completed runs retained automatic dispatch. No specific fused-kernel cause or bitwise trajectory identity is established.

**Sources**

- submission/results_qwen_summary.json
- https://mcauleylab.ucsd.edu/public_datasets/data/amazon/datasets.html
- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
- https://arxiv.org/abs/2106.09685
- https://proceedings.mlr.press/v139/radford21a.html
- https://arxiv.org/abs/1205.2618

## 19. Formal NDCG@10 detail (optional appendix)

Optional discussion: every cell uses the three actual complete pipeline seeds. Variation is a sample standard deviation, not a population confidence interval. Matched win counts pair the same domain and seed.

**Discussion notes**

All 27 raw records passed the declared artifact and protocol audit. No diagnostic or synthetic run supplies these values.

**Sources**

- submission/results_qwen_summary.json

## 20. Sparse items and numerical scope (optional appendix)

Optional discussion: sparse-item results remain close to zero, including zero in Baby. The cohort combines unseen items with low-interaction items, so it does not establish strict cold start. Clothing LoRA training used the MATH attention backend after same-state nonfinite-gradient diagnostics. That recovery does not identify a particular fused-kernel cause or establish bitwise-equivalent training.

**Discussion notes**

Candidate cache numerical checks cover only the first min(4,N) live-Qwen candidates and one validation history. CPU serving replay checks a different layer of correctness.

**Sources**

- submission/results_qwen_summary.json

## 21. Request retrieval and history evaluation (optional appendix)

Optional discussion: the application and the formal experiment answer different questions. Studio uses a matching query encoder for each candidate space and retrieves a model-specific pool before applying constraints and feedback. The formal test instead begins with actual earlier history and ranks the complete eligible catalogue. Studio scores and examples cannot be substituted for those metrics.

**Discussion notes**

All query arms share the shopping instruction, left padding, last-valid-token pooling, maximum 256 tokens and L2 normalization. Image-only requests can produce identical base retrieval lists. The trained head activates with positive history, which always uses frozen base history inputs.

**Sources**

- src/shoplens/studio_encoders.py
- src/shoplens/studio_models.py
- src/shoplens/studio.py
- src/shoplens/qwen_evaluate.py

## 22. Studio source evidence (optional appendix)

Optional discussion: a result card exposes its original title, source category, historical price and computed text and image signals. The disclosed retrieval pool and penalty policy help the user inspect the current result. These signals do not establish suitability or explain the score causally.

**Discussion notes**

This is the later Baby state with a reference photo, a 150 USD historical budget, soft avoid bulky and optional positive/negative/hidden feedback. The screenshot concerns Baby Trend Snap N Go Stroller, not the final chosen Aprica. Its observed text-query cosine is 0.7240 and image-query cosine is 0.9293 in this state.

**Sources**

- assets/studio_evidence.png
- src/shoplens/studio_ui.py
- src/shoplens/studio.py

## 23. Research recommendation evidence (optional appendix)

Optional discussion: Research recommendation evidence combines the product record with the selected history and model. It exposes cross-tower similarities and structured inference evidence. This page supports inspection of an individual recommendation, while the experiment dashboard provides the held-out quality measurements.

**Discussion notes**

The actual Baby example selects Frozen and Fisher-Price Booster Seat, Blue/Green/Gray (B0000DEW8N), with the three displayed preset history items. This slide retains the original candidate description. The full inference evidence view exposes the actual scoring and provenance fields. Similarities remain diagnostics rather than causal score contributions.

**Sources**

- assets/research_evidence_product.png
- src/shoplens/ui.py

## 24. Similarity heatmap (optional appendix)

Optional discussion: the heatmap gives a numerical alternative to the preference network. Rows are actual active history products and columns are current candidates. Entries use the complete one-hundred-and-twenty-eight-dimensional cross-tower cosine, with negative values retained. These values do not decompose the final user-network score.

**Discussion notes**

The image shows a real session diagnostic, not a new benchmark or attention map.

**Sources**

- assets/research_heatmap.png
- src/shoplens/visual_ui.py
- src/shoplens/visual_charts.py

## 25. Training traces and selected checkpoints (optional appendix)

Optional discussion: the experiment dashboard exposes saved training traces and selection evidence. The public demonstration uses the selected seed-forty-two exports, while aggregate results retain all three formal seeds. The plotted trace describes the recorded run. Its appearance is not a new performance claim.

**Discussion notes**

The displayed Baby semantic seed42 trace selects adaptation epoch 2 using 30,000 examples. The line is adaptation_curve/validation_ndcg@10, not test performance. Across the study, all three groups select adaptation epoch 1 or 2 by full-catalogue validation NDCG@10. Existing aggregate results and original baseline results remain separate.

**Sources**

- assets/research_training.png
- src/shoplens/ui.py
- artifacts/qwen/baby/qwen_semantic/seed42/training.json

## 26. Sources and reproducibility (optional appendix)

Optional discussion: the speaker notes contain primary source links and project-relative evidence references. The team should record a real human presentation, complete genuine contribution statements and review the AI disclosure before submission.

**Discussion notes**

Public release preparation is separate from a completed code submission. No private account, machine path, operational log or user-level review data is required to explain the work.

**Sources**

- https://mcauleylab.ucsd.edu/public_datasets/data/amazon/datasets.html
- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
- https://arxiv.org/abs/2106.09685
- https://proceedings.mlr.press/v139/radford21a.html
- https://arxiv.org/abs/1205.2618
- submission/results_qwen_summary.json
- submission/ShopLens_Qwen_AI_disclosure_draft.md
