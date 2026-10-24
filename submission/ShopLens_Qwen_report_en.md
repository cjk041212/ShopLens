# ShopLens Qwen objective comparison

COMPLETE — all 27 records passed the declared evidence audit.

27 / 27 records; 0 audit errors.

The three benchmarks are domains of the same Amazon2014 source; they are not independently sourced datasets. Course acceptance of this three-domain interpretation remains to be confirmed.

Model: `Qwen/Qwen3-Embedding-0.6B`, pinned revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`. The declared encoding uses at most256 tokens, left padding, last-valid-token pooling and L2 normalization, with identical training/export truncation. Candidate descriptions receive no query instruction. Trainable candidates use LoRA r8, alpha16, dropout0.05 on q_proj/v_proj; the recommendation head has128 dimensions.

The primary contrast is an asymmetric two-tower experiment. Histories always use frozen base-Qwen text plus the original fixed CLIP features. Only the candidate text tower changes: frozen Qwen, semantic LoRA, or preference LoRA. Candidate features must not replace history features.

Each complete pipeline seed is repeated independently (42/43/44). Within a domain/seed, all objectives share the selected full-input warmup head, the same sampled behavioral cases and order, and the same negative arrays. Adaptation uses up to30,000 eligible TRAIN cases with exactly matched positive reviews, over two epochs. Select epoch1 or2 by full-catalog downstream validation NDCG@10; epoch0 and test scores cannot select a checkpoint.

Semantic LoRA aligns frozen base-Qwen review anchors with live candidate descriptions using multi-positive InfoNCE; its BPR text input is detached so BPR updates the head. Preference LoRA receives BPR gradients directly. Target reviews never enter recommendation-history queries. All groups retain the same old temporal, exclusion, candidate and user-macro evaluation protocol.

## Metrics

Mean ± sample SD across pipeline seeds; incomplete rows are provisional.

| Domain | Objective | Seeds | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
|---|---|---:|---:|---:|---:|---:|
| baby | qwen_frozen | 3 | 0.027412 ± 0.002656 | 0.012796 ± 0.000959 | 0.042491 ± 0.001817 | 0.016589 ± 0.001039 |
| baby | qwen_preference | 3 | 0.026099 ± 0.002937 | 0.012475 ± 0.000849 | 0.041547 ± 0.002861 | 0.016357 ± 0.000882 |
| baby | qwen_semantic | 3 | 0.019995 ± 0.001775 | 0.009930 ± 0.000728 | 0.032398 ± 0.002454 | 0.013070 ± 0.000873 |
| clothing | qwen_frozen | 3 | 0.010730 ± 0.000846 | 0.005186 ± 0.000562 | 0.018709 ± 0.000847 | 0.007190 ± 0.000585 |
| clothing | qwen_preference | 3 | 0.010476 ± 0.000572 | 0.005124 ± 0.000463 | 0.018854 ± 0.000463 | 0.007227 ± 0.000375 |
| clothing | qwen_semantic | 3 | 0.009161 ± 0.000485 | 0.004460 ± 0.000384 | 0.014435 ± 0.001244 | 0.005788 ± 0.000536 |
| sports | qwen_frozen | 3 | 0.014565 ± 0.000848 | 0.007116 ± 0.000684 | 0.024839 ± 0.001840 | 0.009697 ± 0.000946 |
| sports | qwen_preference | 3 | 0.014429 ± 0.000688 | 0.007039 ± 0.000584 | 0.024675 ± 0.002447 | 0.009606 ± 0.001005 |
| sports | qwen_semantic | 3 | 0.013297 ± 0.001257 | 0.006872 ± 0.000613 | 0.020772 ± 0.001763 | 0.008745 ± 0.000547 |

## Paired changes

| Domain | After − before | Seeds | ΔNDCG@10 mean ± SD | Wins / ties / losses |
|---|---|---|---:|---:|
| baby | qwen_semantic − qwen_frozen | 42,43,44 | -0.002866 ± 0.000364 | 0 / 0 / 3 |
| baby | qwen_preference − qwen_frozen | 42,43,44 | -0.000322 ± 0.000349 | 1 / 0 / 2 |
| baby | qwen_preference − qwen_semantic | 42,43,44 | +0.002545 ± 0.000558 | 3 / 0 / 0 |
| sports | qwen_semantic − qwen_frozen | 42,43,44 | -0.000244 ± 0.000817 | 2 / 0 / 1 |
| sports | qwen_preference − qwen_frozen | 42,43,44 | -0.000076 ± 0.000190 | 1 / 0 / 2 |
| sports | qwen_preference − qwen_semantic | 42,43,44 | +0.000168 ± 0.000865 | 2 / 0 / 1 |
| clothing | qwen_semantic − qwen_frozen | 42,43,44 | -0.000726 ± 0.000405 | 0 / 0 / 3 |
| clothing | qwen_preference − qwen_frozen | 42,43,44 | -0.000062 ± 0.000153 | 1 / 0 / 2 |
| clothing | qwen_preference − qwen_semantic | 42,43,44 | +0.000664 ± 0.000251 | 3 / 0 / 0 |

- baby, qwen_semantic versus qwen_frozen: mean NDCG@10 is lower; per-seed wins/ties/losses are 0/0/3. This describes the declared runs only.
- baby, qwen_preference versus qwen_frozen: mean NDCG@10 is lower; per-seed wins/ties/losses are 1/0/2. This describes the declared runs only.
- baby, qwen_preference versus qwen_semantic: mean NDCG@10 is higher; per-seed wins/ties/losses are 3/0/0. This describes the declared runs only.
- sports, qwen_semantic versus qwen_frozen: mean NDCG@10 is lower; per-seed wins/ties/losses are 2/0/1. This describes the declared runs only.
- sports, qwen_preference versus qwen_frozen: mean NDCG@10 is lower; per-seed wins/ties/losses are 1/0/2. This describes the declared runs only.
- sports, qwen_preference versus qwen_semantic: mean NDCG@10 is higher; per-seed wins/ties/losses are 2/0/1. This describes the declared runs only.
- clothing, qwen_semantic versus qwen_frozen: mean NDCG@10 is lower; per-seed wins/ties/losses are 0/0/3. This describes the declared runs only.
- clothing, qwen_preference versus qwen_frozen: mean NDCG@10 is lower; per-seed wins/ties/losses are 1/0/2. This describes the declared runs only.
- clothing, qwen_preference versus qwen_semantic: mean NDCG@10 is higher; per-seed wins/ties/losses are 3/0/0. This describes the declared runs only.

## Low-interaction targets

Targets with <=5 distinct positive TRAIN users, including zero. This is not a separately constructed strict cold-start test.

| Domain | Objective | Seeds | Users / targets | Recall@10 | NDCG@10 |
|---|---|---:|---:|---:|---:|
| baby | qwen_frozen | 3 | 1605 / 2694 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| baby | qwen_preference | 3 | 1605 / 2694 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| baby | qwen_semantic | 3 | 1605 / 2694 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| clothing | qwen_frozen | 3 | 5294 / 7430 | 0.000728 ± 0.000098 | 0.000263 ± 0.000046 |
| clothing | qwen_preference | 3 | 5294 / 7430 | 0.000602 ± 0.000397 | 0.000226 ± 0.000134 |
| clothing | qwen_semantic | 3 | 5294 / 7430 | 0.000750 ± 0.000319 | 0.000255 ± 0.000116 |
| sports | qwen_frozen | 3 | 4554 / 6834 | 0.000403 ± 0.000168 | 0.000161 ± 0.000085 |
| sports | qwen_preference | 3 | 4554 / 6834 | 0.000366 ± 0.000127 | 0.000153 ± 0.000082 |
| sports | qwen_semantic | 3 | 4554 / 6834 | 0.000186 ± 0.000163 | 0.000062 ± 0.000055 |

## Interpretation and limits

Three training seeds do not establish statistical significance or a population confidence interval. A mean increase can coexist with losses in individual seeds. Report decreases and near-zero low-interaction scores without promising online gains.

Legacy MiniLM baselines used a fixed five-epoch main ranker budget and one shared adaptation seed. Qwen uses a separately declared warmup of up to20 epochs plus sampled adaptation and full-pipeline seeds. Differences against legacy tables mix model, budget and sampling changes; within-Qwen contrasts are the controlled objective comparison.

All source baseline reports and artifacts remain separate. Hash declarations are audited here; baseline byte preservation is claimed only when the separate verification evidence is supplied. Complete findings require all27 real full-evaluation records and all training evidence, rather than a visually complete table.

## Training evidence

Gradient routing is checked through the recorded informative probes (at most10), not a measurement of every step.

Export numerical checks cover only the first min(4,N) live-Qwen candidates and one validation history. They do not establish full-catalog live-Qwen re-encoding equality. A separate CPU cached-feature-to-head versus serving check, when supplied, is a different test and does not rerun Qwen.

| Unit | Selected epoch | Sample / eligible | Head grad max | Total LoRA grad max | BPR→LoRA max | Candidate cache error |
|---|---:|---:|---:|---:|---:|---:|
| baby/qwen_frozen/seed42 | 1 | 30000 / 61880 | 0.846 | 0 | 0 | 1.19e-07 |
| baby/qwen_frozen/seed43 | 2 | 30000 / 61880 | 0.794 | 0 | 0 | 1.79e-07 |
| baby/qwen_frozen/seed44 | 1 | 30000 / 61880 | 0.818 | 0 | 0 | 2.09e-07 |
| baby/qwen_preference/seed42 | 2 | 30000 / 61880 | 0.846 | 0.697 | 0.697 | 0.0038 |
| baby/qwen_preference/seed43 | 1 | 30000 / 61880 | 0.794 | 0.919 | 0.919 | 0.00473 |
| baby/qwen_preference/seed44 | 1 | 30000 / 61880 | 0.818 | 0.897 | 0.897 | 0.00433 |
| baby/qwen_semantic/seed42 | 2 | 30000 / 61880 | 0.846 | 1.21 | 0 | 0.00303 |
| baby/qwen_semantic/seed43 | 1 | 30000 / 61880 | 0.794 | 1.29 | 0 | 0.0056 |
| baby/qwen_semantic/seed44 | 2 | 30000 / 61880 | 0.818 | 1.16 | 0 | 0.00383 |
| clothing/qwen_frozen/seed42 | 2 | 30000 / 114449 | 0.431 | 0 | 0 | 2.68e-07 |
| clothing/qwen_frozen/seed43 | 1 | 30000 / 114449 | 0.512 | 0 | 0 | 1.79e-07 |
| clothing/qwen_frozen/seed44 | 1 | 30000 / 114449 | 0.525 | 0 | 0 | 2.38e-07 |
| clothing/qwen_preference/seed42 | 2 | 30000 / 114449 | 0.431 | 0.647 | 0.647 | 0.00402 |
| clothing/qwen_preference/seed43 | 1 | 30000 / 114449 | 0.512 | 0.623 | 0.623 | 0.00432 |
| clothing/qwen_preference/seed44 | 2 | 30000 / 114449 | 0.523 | 0.536 | 0.536 | 0.00256 |
| clothing/qwen_semantic/seed42 | 2 | 30000 / 114449 | 0.431 | 1.69 | 0 | 0.0071 |
| clothing/qwen_semantic/seed43 | 1 | 30000 / 114449 | 0.512 | 1.42 | 0 | 0.00188 |
| clothing/qwen_semantic/seed44 | 2 | 30000 / 114449 | 0.523 | 1.13 | 0 | 0.00328 |
| sports/qwen_frozen/seed42 | 2 | 30000 / 139446 | 0.714 | 0 | 0 | 1.79e-07 |
| sports/qwen_frozen/seed43 | 1 | 30000 / 139446 | 0.768 | 0 | 0 | 2.38e-07 |
| sports/qwen_frozen/seed44 | 1 | 30000 / 139446 | 0.698 | 0 | 0 | 2.09e-07 |
| sports/qwen_preference/seed42 | 1 | 30000 / 139446 | 0.714 | 0.758 | 0.758 | 0.00337 |
| sports/qwen_preference/seed43 | 1 | 30000 / 139446 | 0.768 | 1.05 | 1.05 | 0.00404 |
| sports/qwen_preference/seed44 | 1 | 30000 / 139446 | 0.698 | 1.12 | 1.12 | 0.00423 |
| sports/qwen_semantic/seed42 | 2 | 30000 / 139446 | 0.714 | 1.1 | 0 | 0.00344 |
| sports/qwen_semantic/seed43 | 1 | 30000 / 139446 | 0.768 | 1.21 | 0 | 0.0043 |
| sports/qwen_semantic/seed44 | 1 | 30000 / 139446 | 0.698 | 0.981 | 0 | 0.00402 |

## Actual execution recovery

[scheduler-job] → [scheduler-job]; baby/qwen_frozen/seed42.

The initial test computation finished, but saving its result failed because torch.device was not JSON serializable. After recording the device with str(device), the unsaved evaluation was recomputed from the same selected training checkpoint. This is one persisted formal record for that selected checkpoint, not a claim that test computation never repeated. Existing persisted results are reused after subsequent interruptions.

For the first serialization recovery, the note records unchanged training code: True; reused selected training checkpoints: True; test scores used for selection or tuning: False. The recorded changed module is `qwen_evaluate.py`. This operational recovery does not relax any of the27-record or training-evidence audit requirements.

### Numerical recovery and backend limits

[scheduler-job] → diagnostic [scheduler-job] → [scheduler-job]; clothing/qwen_semantic/seed42.

The recorded automatic-SDPA diagnostic first failed at step 69, after 68 successful updates: forward values and head gradients were finite, but LoRA gradients were nonfinite before clipping. The same clean state, batch and RNG reproduced the failure; MATH replay and a fresh100-step MATH diagnostic prefix remained finite. This evidence does not establish a specific fused-kernel root cause.

The recovery changes live CUDA candidate training to MATH SDPA and rejects nonfinite head/LoRA gradient norms before optimizer.step; cached encoding remains automatic. The 19 completed formal units are preserved. The failed unit restarts from initialization, without a resume checkpoint. Unchanged objective, sampling, learning rates and budget: True; test scores used for selection or tuning: False.

All six Clothing LoRA units use MATH training SDPA; previously completed Baby/Sports LoRA units retain their automatic-SDPA execution and are not required to acquire new metadata retroactively. Clothing frozen42 is reused and has no LoRA training. These backend trajectories are not asserted bitwise identical; cross-domain differences are conditional on the numerical configurations.

Numerical recovery status: complete; production same-state regression: passed; production100-step prefix regression: passed. Final completeness additionally requires status=complete, both production gates=passed, and all six Clothing LoRA training.json records declaring MATH, the nonfinite gradient guard and automatic cached encoding.

Source: `experiments/qwen/recovery_note.json` (included verbatim in the machine-readable verification).

## Audit


## Sources

- [Pinned Qwen model](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3)
- [Amazon2014 original data](https://mcauleylab.ucsd.edu/public_datasets/data/amazon/datasets.html)
- [CLIP original paper](https://proceedings.mlr.press/v139/radford21a.html)
- Machine-readable run evidence: `experiments/qwen/verification.json` and `artifacts/qwen/<domain>/<mode>/seed<seed>/training.json`.
