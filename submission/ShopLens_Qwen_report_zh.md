# ShopLens Qwen 训练目标比较

完成：27 条结果通过本报告的声明与证据审计。

27 / 27 records; 0 audit errors.

三个基准来自同一 Amazon2014 数据源的三个领域，并非三个独立来源。课程是否接受三个领域的口径仍需确认。

模型：`Qwen/Qwen3-Embedding-0.6B`，固定 revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`。编码固定最多256 token、左侧 padding、最后有效 token 池化与 L2 归一化，训练和导出采用相同截断方式。候选商品描述不加 query instruction；可训练候选塔在 q_proj/v_proj 使用 LoRA r8、alpha16、dropout0.05，推荐头为128维。

这是非对称双塔实验：历史塔始终使用冻结的 base-Qwen 文本与原有固定 CLIP 图片特征；候选文本塔分别为冻结 Qwen、语义 LoRA、偏好 LoRA。不能把适配后的候选向量用于历史塔。

每个 seed42/43/44 独立重跑完整流水线；同域同 seed 的三种目标共享选中的全量 warmup 头、相同行为案例及顺序、相同负样本数组。适配从匹配正面评论的 TRAIN 行为中抽取最多30,000例，训练两个 epoch，以完整候选目录的下游验证 NDCG@10 在 epoch1/2 中选优，epoch0 和测试分数不参与选择。

语义 LoRA 用冻结 base-Qwen 评论锚点与实时商品描述做多正例 InfoNCE，BPR 输入文本向量 detach 后只更新推荐头；偏好 LoRA 直接接收 BPR 梯度。目标评论不进入用户历史查询。三组沿用原有时序、候选排除与用户宏平均评价协议。

## 实测指标

完整流水线种子的均值 ± 样本标准差；未完成状态下的表格仅供检查已有记录。

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

## 同 seed 配对变化

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

- baby，qwen_semantic 相比 qwen_frozen：NDCG@10 配对均值下降，逐 seed 改善/持平/下降为 0/0/3；仅描述本次声明的运行。
- baby，qwen_preference 相比 qwen_frozen：NDCG@10 配对均值下降，逐 seed 改善/持平/下降为 1/0/2；仅描述本次声明的运行。
- baby，qwen_preference 相比 qwen_semantic：NDCG@10 配对均值增加，逐 seed 改善/持平/下降为 3/0/0；仅描述本次声明的运行。
- sports，qwen_semantic 相比 qwen_frozen：NDCG@10 配对均值下降，逐 seed 改善/持平/下降为 2/0/1；仅描述本次声明的运行。
- sports，qwen_preference 相比 qwen_frozen：NDCG@10 配对均值下降，逐 seed 改善/持平/下降为 1/0/2；仅描述本次声明的运行。
- sports，qwen_preference 相比 qwen_semantic：NDCG@10 配对均值增加，逐 seed 改善/持平/下降为 2/0/1；仅描述本次声明的运行。
- clothing，qwen_semantic 相比 qwen_frozen：NDCG@10 配对均值下降，逐 seed 改善/持平/下降为 0/0/3；仅描述本次声明的运行。
- clothing，qwen_preference 相比 qwen_frozen：NDCG@10 配对均值下降，逐 seed 改善/持平/下降为 1/0/2；仅描述本次声明的运行。
- clothing，qwen_preference 相比 qwen_semantic：NDCG@10 配对均值增加，逐 seed 改善/持平/下降为 3/0/0；仅描述本次声明的运行。

## 低交互商品

训练期不同正反馈用户数 ≤5 的目标商品，包含零交互商品；这不是专门构造的严格冷启动测试。

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

## 解读边界

三个训练种子不能证明统计显著性，也不能代表用户总体置信区间。均值增加可能伴随部分 seed 下降；应如实报告下降和低交互组接近零的情况，不承诺线上收益。

旧 MiniLM 主表采用固定5个 epoch 的推荐器预算和一个共享适配 seed；Qwen 采用另行声明的最多20 epoch warmup、抽样适配和完整流水线种子。与旧主表的差异同时包含模型、预算和采样变化，不能全部归因为模型或 LoRA；Qwen 三目标之间才是本轮受控对照。

旧报告与工件保持独立。本报告核对声明的哈希与训练证据；只有提供独立 baseline verification 时才声称旧工件字节未变。完成结论需要27条真实完整评价及全部证据，不能由表格外观代替。

## 训练与导出证据

梯度路径依据记录的最多10个 informative probe 检查，不能描述为每个 step 都进行了测量。

导出数值抽查只覆盖前 min(4,N) 个实时 Qwen 候选与一条验证历史，不能解释为全目录逐项重编码等价。若另提供 CPU 上缓存原始特征经推荐头与 serving 的比较，那是独立检查，未重新运行 Qwen，须与这项抽查区分。

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

## 实际运行恢复记录

[scheduler-job] → [scheduler-job]; baby/qwen_frozen/seed42.

首次测试计算已经结束，但保存结果时因 torch.device 无法 JSON 序列化而失败，未留下正式结果。将记录字段修为 str(device) 后，从同一个已选训练 checkpoint 重新计算了这次未保存的评估。这里的口径是一条已持久化的正式记录，不代表测试计算从未重复；已有保存结果在之后的恢复中直接复用。

首次序列化恢复记录声明：训练代码未变=True，复用已选训练 checkpoint=True，使用测试分数选模或调参=False；记录的变更模块为 `qwen_evaluate.py`。这项运行恢复不放宽27条结果或任何训练证据的审计标准。

### 数值恢复与 backend 限制

[scheduler-job] → diagnostic [scheduler-job] → [scheduler-job]; clothing/qwen_semantic/seed42.

记录的 automatic SDPA 诊断在第 69 step 首次出现故障，此前完成 68 次更新：前向值和推荐头梯度有限，但裁剪前 LoRA 梯度出现非有限值。同一 clean state、batch 与 RNG 可复现；MATH 重放及全新100步 MATH 诊断前缀均有限。该证据不能确定具体 fused kernel 是根因。

恢复将实时 CUDA 候选训练改用 MATH SDPA，并在 optimizer.step 前拒绝非有限的推荐头/LoRA 梯度范数；缓存编码仍为 automatic。保留已经完成的 19 条正式记录；失败单元没有恢复 checkpoint，从初始化重启。训练目标、采样、学习率和预算未变=True，使用测试分数选模或调参=False。

Clothing 六个 LoRA 单元统一使用 MATH 训练 SDPA；Baby/Sports 已完成 LoRA 单元保留 automatic SDPA 原运行，不要求事后补写新元数据。Clothing frozen42 复用且没有 LoRA 训练。这些 backend 轨迹不声明 bitwise 等价，跨领域差异的解释受不同数值配置限制。

数值恢复状态：complete；生产函数同状态回归：passed；生产函数100步前缀回归：passed。最终完成还要求状态为 complete、两个生产门槛均为 passed，并核对 Clothing 六个 LoRA 的 training.json 声明 MATH、非有限梯度保护与 automatic 缓存编码。

Source: `experiments/qwen/recovery_note.json` (included verbatim in the machine-readable verification).

## 审计状态


## 资料

- [Pinned Qwen model](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3)
- [Amazon2014 original data](https://mcauleylab.ucsd.edu/public_datasets/data/amazon/datasets.html)
- [CLIP original paper](https://proceedings.mlr.press/v139/radford21a.html)
- Machine-readable run evidence: `experiments/qwen/verification.json` and `artifacts/qwen/<domain>/<mode>/seed<seed>/training.json`.
