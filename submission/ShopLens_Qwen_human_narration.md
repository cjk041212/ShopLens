# ShopLens Qwen human narration script

A real group member must record the narration. No artificial voice accompanies this deck.
The entire recording must stay below 10 minutes, including slides, code, the UI demonstration, discussion, references and transitions.
The formal narration is about 800–850 words, approximately 7–7.5 minutes at 110–120 words per minute. Results, limitations and the brief reference acknowledgement are already part of that narration.
Reserve 90–120 seconds total for real operation and code within the same 10-minute budget: about 60–80 seconds for the UI and 30–40 seconds for code. These are not extra minutes after the slides. Aim for 9:30 overall and verify the actual duration by rehearsal.
Use the source notes for discussion. Do not read the technical notes and URLs aloud.
Prepared narration: approximately 850 words. Presenter names and genuine contributions remain for the team to complete.

## Slide 1: ShopLens

ShopLens studies whether adapting language representations improves multimodal recommendation. Product photographs provide an additional input, while the main experiment focuses on text. We compare frozen Qwen embeddings with two LoRA objectives: matching reviews to descriptions, and directly learning preference rankings. We then export compact vectors so that the demonstration can run on a local machine.

**Sources**

- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- https://arxiv.org/abs/2106.09685
- https://proceedings.mlr.press/v139/radford21a.html

**Discussion notes outside timed narration**

ShopLens studies how language adaptation changes recommendation. We keep product photographs as an additional modality and focus on the text representation. The central experiment compares a frozen Qwen embedding model with two LoRA objectives. One aligns reviews with descriptions. The other trains directly for preference ranking. The trained system exports compact item vectors so that an interactive demonstration can run locally.

## Slide 2: Research question

A model can recognise that a review describes a product without learning what a particular user prefers next. The semantic arm matches review language to product descriptions. The preference arm learns to rank a liked product above an unobserved product, given earlier history. The experiment asks whether these objectives produce the same benefit on the recommendation metric. The model family and image features stay fixed.

**Sources**

- https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md
- https://arxiv.org/abs/1205.2618
- src/shoplens/qwen_pipeline.py

**Discussion notes outside timed narration**

A model can understand that a review refers to a product without learning which product a particular user prefers next. We therefore separate semantic alignment from preference ranking. The semantic arm pairs a positive training review with its matching description. The preference arm uses the user’s earlier liked products to train a positive item above an unobserved item. All arms use the same Qwen model family and fixed image features. The question is empirical: under this controlled budget, does semantic improvement transfer to the recommendation metric, or does the ranking objective produce a different result?

## Slide 3: Datasets

We use three product domains from Amazon Reviews 2014. These are real reviews, catalogue descriptions and downloaded photographs. The table shows the cleaned data counts. Four- and five-star interactions count as positive feedback. Only training-period reviews supply semantic supervision. Product text contains the title and description. These are separate benchmarks from one data source, so the course instructor still needs to confirm that they satisfy the requested dataset count.

**Sources**

- https://mcauleylab.ucsd.edu/public_datasets/data/amazon/datasets.html
- data/processed/baby/audit.json
- data/processed/sports/audit.json
- data/processed/clothing/audit.json

**Discussion notes outside timed narration**

The project uses three domains from the Amazon Reviews 2014 collection. We retain products with usable text and real, decodable product photographs. The table reads directly from the processed data audits. Four- and five-star interactions provide positive behavioural feedback. Training-period review text supplies semantic supervision. Catalogue text itself contains only the title and description. These are separate domain benchmarks from a single source. We must ask the course instructor whether that satisfies the required number of datasets, rather than calling them three independently sourced collections.

## Slide 4: Evaluation protocol

Each domain uses chronological training, validation and test splits. A target receives only strictly earlier positive history, with at most twenty recent distinct products. The model ranks the full prepared catalogue and excludes products already liked. Earlier held-out observations may enter later histories, but weights stay fixed. We average targets within each user, then average users. Validation selects the checkpoint. Test labels never select the model.

**Sources**

- src/shoplens/models.py
- src/shoplens/qwen_evaluate.py

**Discussion notes outside timed narration**

Each domain uses global timestamp quantiles for training, validation and test. Timestamp ties stay in the earlier split, so record counts need not equal the nominal percentages exactly. A target uses only strictly earlier positive history, with at most twenty recent distinct products. Repeated targets that already occur in that history do not enter evaluation. We rank the complete prepared catalogue and exclude earlier positives. Scores use a stable item-ID tie rule. The evaluation is prequential: earlier observations within a held-out split may enter a later history, while model weights stay fixed. We average target scores within each user before calculating the user macro mean.

## Slide 5: Asymmetric architecture

The history path always receives frozen base-Qwen text and frozen CLIP images. The candidate path receives either base text or selected LoRA text, with the same images. Small projection and fusion networks produce separate one-hundred-and-twenty-eight-dimensional vectors. A user network pools history and scores candidates through a dot product. Frozen refers to the Qwen encoder. Its recommendation head still trains. Exported history and candidate caches preserve this distinction during local inference.

**Sources**

- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- https://proceedings.mlr.press/v139/radford21a.html
- src/shoplens/qwen_models.py
- src/shoplens/qwen_inference.py

**Discussion notes outside timed narration**

All three arms receive the same frozen base-Qwen history embeddings and frozen CLIP image embeddings. Only the candidate text representation changes across objectives. Separate small projection and fusion networks form history and candidate vectors. The user encoder mean-pools recent history vectors and applies an MLP. A dot product scores every candidate. The small ranking head can update in every arm, including frozen. Frozen therefore means a frozen Qwen encoder, not an untrained recommender. At export, we save history vectors and candidate vectors separately. The local interface must use the frozen-history path instead of accidentally using adapted candidate vectors as history.

## Slide 6: Objectives and gradients

The frozen arm trains only the recommendation head with BPR. The semantic arm uses detached text vectors for BPR, while a separate contrastive loss updates LoRA by matching descriptions to frozen review anchors. Reviews of the same product count as multiple positives. The preference arm lets BPR update both the head and LoRA. It never uses the target review as a preference query. All arms keep the image encoder fixed.

**Sources**

- https://arxiv.org/abs/2106.09685
- https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md
- src/shoplens/qwen_text.py
- src/shoplens/qwen_pipeline.py

**Discussion notes outside timed narration**

The frozen arm updates the recommendation head using BPR while keeping Qwen fixed. The semantic arm runs live candidate embeddings, but detaches them for the ranking-head loss. Its separate symmetric contrastive loss updates LoRA by matching product descriptions to frozen review anchors from the base Qwen model. Reviews for the same product count as multiple positives, preventing them from becoming obvious false negatives. The preference arm lets the BPR gradient update both the ranking head and LoRA through positive and negative candidate text. We configure LoRA with rank eight, alpha sixteen and dropout zero point zero five on the query and value projections. The target review never enters the preference query.

## Slide 7: Training budget and checkpoint selection

Every domain and seed first trains a common head on all eligible cases. The three arms share that selected initialization. Adaptation uses the same thirty thousand review-matched cases, order and sampled negatives for two epochs. Full-catalogue validation NDCG selects epoch one or two in every arm. Warmup epoch zero cannot replace a poor adaptation result. Three complete pipeline seeds across three domains and three arms produce twenty-seven test records.

**Sources**

- src/shoplens/qwen_data.py
- src/shoplens/qwen_pipeline.py
- deliverables/results_qwen_summary.json

**Discussion notes outside timed narration**

For each domain and seed, we first train a common ranking head using every eligible training case. Warmup runs for at most twenty epochs, with patience three after at least five epochs, selecting through full-catalogue validation NDCG. Every objective starts from that same head. Adaptation then uses thirty thousand eligible training cases with matching positive review text. The case order and negative arrays match across arms. We run two adaptation epochs with batch size thirty-two. The head learning rate is one times ten to the minus four, and the LoRA learning rate is two times ten to the minus five. Every arm selects epoch one or two using the same downstream validation metric. Warmup epoch zero cannot replace a failed adaptation checkpoint. Test evaluation runs only after selection. Three seeds repeat the full pipeline, rather than only the final head.

The study uses a full-input warmup, sampled adaptation, and full evaluation. It does not claim that every eligible case enters the LoRA adaptation stage. Formal unit hashes record the common warmup, cases, order and negatives. The existing five-epoch MiniLM comparisons remain a separate earlier experiment with a different budget.

## Slide 8: Test ranking results

These are actual test scores after validation-only selection. The chart shows mean user-macro NDCG at ten across three complete pipeline seeds. Semantic alignment exceeds frozen NDCG@10 in 2 of 9 matched domain-and-seed comparisons. Preference ranking exceeds frozen NDCG@10 in 3 of 9 matched domain-and-seed comparisons. The following slide checks variation rather than relying on a single best seed. These measurements describe offline ranking under this budget.

**Sources**

- deliverables/results_qwen_summary.json
- artifacts/qwen/baby/results.json
- artifacts/qwen/sports/results.json
- artifacts/qwen/clothing/results.json

**Discussion notes outside timed narration**

The chart reports actual test NDCG at ten after validation-only checkpoint selection. Baby: Frozen has the highest observed mean, 0.0128. Sports: Frozen has the highest observed mean, 0.0071. Clothing: Frozen has the highest observed mean, 0.0052. Semantic alignment exceeds frozen NDCG@10 in 2 of 9 matched domain-and-seed comparisons. Preference ranking exceeds frozen NDCG@10 in 3 of 9 matched domain-and-seed comparisons. These are observations under the declared budget, rather than claims about online purchasing behaviour. The next slide reports seed variation and paired changes.

## Slide 9: Seed variation

The table shows means and sample standard deviations. Each paired change compares the same domain and seed, sharing initialization and adaptation cases. A win counts a positive metric difference. It does not mean statistical significance. Three seeds provide a limited view of variation. We retain settings where LoRA hurts, rather than silently replacing them with the warmup checkpoint.

**Sources**

- deliverables/results_qwen_summary.json

**Discussion notes outside timed narration**

Semantic alignment exceeds frozen NDCG@10 in 2 of 9 matched domain-and-seed comparisons. Preference ranking exceeds frozen NDCG@10 in 3 of 9 matched domain-and-seed comparisons. The table reports sample standard deviation across the three independent pipeline seeds. We pair changes by domain and seed because the three arms share warmup and adaptation examples for that seed. A win means a positive metric difference, not a statistical significance result. With three seeds, these differences remain descriptive. We retain results where LoRA hurts rather than replacing them with the warmup checkpoint.

## Slide 10: Sparse item results

This analysis groups targets with at most five distinct training-positive users, including zero. It asks where language representations may help when behavioural evidence is sparse. Every method still ranks the same full catalogue. The cohort mixes unseen products with products that have some interactions, so it does not establish strict item cold-start performance. The saved results retain cohort user and target counts.

**Sources**

- deliverables/results_qwen_summary.json
- src/shoplens/qwen_evaluate.py

**Discussion notes outside timed narration**

We also report recommendation quality for products with at most five distinct positive users in the training period. This includes some zero-interaction products, but it combines them with products that do have training feedback. It therefore does not isolate strict item cold start. Baby: Frozen 0.0000 ± 0.0000, Semantic alignment 0.0000 ± 0.0000, Preference ranking 0.0000 ± 0.0000. Sports: Frozen 0.0002 ± 0.0001, Semantic alignment 0.0001 ± 0.0001, Preference ranking 0.0002 ± 0.0001. Clothing: Frozen 0.0003 ± 0.0000, Semantic alignment 0.0003 ± 0.0001, Preference ranking 0.0002 ± 0.0001. The cohort uses the same full candidate catalogue as the primary evaluation. Its user and target counts appear in the raw saved results. These scores address where language information might matter, while retaining the same limitations of an offline observational dataset.

## Slide 11: Local demonstration

The local interface displays real product photographs and accepts a short history of liked products. We hold that history fixed while comparing the three objectives. The screen shows ranked products, rank changes and actual saved metrics. It uses compact exported vectors, so the Qwen encoder stays outside the local recommendation path. Similarities and matching text provide diagnostic evidence, rather than causal explanations. I will now briefly demonstrate the same preference history across the three columns.

**Sources**

- src/shoplens/ui.py
- src/shoplens/qwen_inference.py
- assets/qwen_compare_baby.jpg

**Discussion notes outside timed narration**

The interface shows real product photographs and lets the audience choose a short preference history. The comparison keeps that history fixed across the frozen, semantic and preference arms. It displays ranked products, rank changes and saved evaluation metrics. Original baseline methods remain available as separate references. The local application loads the exported history and candidate vectors with the small ranking head, so it does not need to load the Qwen encoder to recommend. Feature similarities and matching text provide diagnostic evidence, rather than a causal explanation of the ranking. During a live demonstration, choose a few products and compare the three columns without changing the history.

## Slide 12: Findings and limitations

Semantic alignment exceeds frozen NDCG@10 in 2 of 9 matched domain-and-seed comparisons. Preference ranking exceeds frozen NDCG@10 in 3 of 9 matched domain-and-seed comparisons. The earlier baselines remain available, but their different architecture and budget prevent attributing a score change to Qwen alone. The domains share one Amazon source, and static catalogue metadata limits historical realism. These offline observations do not establish online conversion or broad cold-start gains. Clothing used MATH SDPA after a reproduced gradient failure. Earlier completed runs kept automatic dispatch. We claim no bitwise trajectory identity. The main contribution is a controlled comparison of semantic and preference objectives, with a reproducible local demonstration. Our references are Amazon Reviews 2014, official Qwen documentation, and the original LoRA, CLIP and BPR papers. Thank you.

**Sources**

- https://mcauleylab.ucsd.edu/public_datasets/data/amazon/datasets.html
- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md
- https://arxiv.org/abs/2106.09685
- https://proceedings.mlr.press/v139/radford21a.html
- https://arxiv.org/abs/1205.2618
- deliverables/results_qwen_summary.json
- experiments/qwen/recovery_note.json

**Discussion notes outside timed narration**

The central finding is the measured relationship between a semantic objective and a preference objective. Semantic alignment exceeds frozen NDCG@10 in 2 of 9 matched domain-and-seed comparisons. Preference ranking exceeds frozen NDCG@10 in 3 of 9 matched domain-and-seed comparisons. Baby: Frozen has the highest observed mean, 0.0128. Sports: Frozen has the highest observed mean, 0.0071. Clothing: Frozen has the highest observed mean, 0.0052. We preserve the earlier six-method comparisons and independent MF budget check. Their architecture and training budget differ from this upgrade, so a score difference cannot establish that changing the text backbone alone caused an improvement. The three domains share a single Amazon source. Catalogue metadata comes from a static crawl whose historical availability is unknown. Our measurements predict recorded positive interactions and do not establish online purchase conversion. The human team must complete its genuine contribution statements and record the presentation in a real member’s voice. Thank you.

Baby preserved TF-IDF mean NDCG@10: 0.007049 (3 records). Sports preserved TF-IDF mean NDCG@10: 0.013770 (3 records). Clothing preserved TF-IDF mean NDCG@10: 0.014282 (3 records).

The recovery policy applies MATH SDPA only to live CUDA candidate training forward for all six Clothing LoRA units. Earlier completed Baby/Sports LoRA units keep automatic dispatch. Evaluation and cache encoding keep automatic dispatch. The objective, samples, learning rates and budget remain unchanged. The same clean state, batch and RNG reproduced nonfinite LoRA gradients under automatic dispatch, while MATH replay and a fresh prefix stayed finite. These observations do not establish a particular fused-kernel root cause, bitwise trajectory equality or a performance improvement. Cross-domain differences remain conditional on the numerical configurations.

Primary references: Amazon Reviews 2014 data archive, Qwen3-Embedding official model card and training documentation, Hu et al. (LoRA), Radford et al. (CLIP), Rendle et al. (BPR). Full URLs appear above. Statistical significance and online evaluation remain outside this study.
