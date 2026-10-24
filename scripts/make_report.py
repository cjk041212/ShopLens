#!/usr/bin/env python3
"""Create tables and a readable report only from measured artifacts."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,json,statistics,time,sys
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
LABELS={'popularity':'Popularity','mf':'BPR factors + history fold-in','tfidf':'TF-IDF history cosine','text':'MiniLM text ranker','multimodal':'CLIP + MiniLM ranker','adapted':'CLIP + adapted MiniLM ranker'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();root=a.root
 out=root/'deliverables';out.mkdir(exist_ok=True)
 lines=['# ShopLens 实验报告','',f'生成日期：{datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y-%m-%d %H:%M:%S")}（UTC+8）。','',
 '本报告只读取真实工件，不补全缺失分数。三个数据域来自同一Amazon2014数据源。是否满足课程“三个数据集”的口径仍需向授课老师确认。','',
 '## 问题与方法','',
 '研究图片信息在文本推荐之外是否有额外价值，以及使用训练期评论进行文本语义适配后，排序结果如何变化。CLIP ViT-B/32提供冻结图片特征，MiniLM-L6-v2提供文本特征。训练128维商品融合与历史偏好网络，使用BPR成对排序损失。','',
 '适配阶段微调MiniLM最后两个Transformer层，以正面评论和对应商品描述作对比学习。该目标学习评论与商品表达之间的语义对应，不能直接称为学习用户偏好。适配仅使用训练期评论，编码器种子42固定，重复种子42/43/44只覆盖下游推荐器训练。','',
 '图文模型增加图片投影支路及更宽的融合输入，参数量未与纯文本模型匹配。二者差值描述整套方法在该预算下的变化，不能单独归因为图片内容；图文模型与评论适配版本则使用相同推荐器结构。','',
 '## 数据与评测协议','',
 '以评分>=4为正反馈，按全局时间0.8/0.9分位数划分。相同时间戳的事件互不作为历史。权重在评测时固定，但已经发生的验证/测试反馈可进入之后的历史（prequential）。每个目标排除全部过去正反馈，使用最多20个最近不同商品构造用户表示。','',
 '对完整保留目录排序，逐目标计算单正样本Hit/Recall和NDCG，先在用户内平均再对用户平均。过滤无过去历史和重复已见目标。目录元数据是静态快照，不能保证还原商品描述在历史时间的可见性。','',
 '低交互商品按训练期不同正反馈用户数<=5定义，包含零交互商品。它不是专门构造的严格未见商品冷启动实验。BPR因子基线在推理时以history ridge fold-in生成用户向量，适用于新会话，不能等同标准用户ID矩阵分解的论文结果。','',
 '所有比较共用清洗后图文完整商品目录。商品编码时长文本按224 wordpieces分块并归一化均值聚合；文本适配训练每条输入上限256 wordpieces。','',
 '| 数据域 | 商品 | 用户 | 交互 | 训练 | 验证 | 测试 |','|---|---:|---:|---:|---:|---:|---:|']
 all_rows=[];summaries=[]
 for domain in ('baby','sports','clothing'):
  audit_path=root/'data/processed'/domain/'audit.json'
  if audit_path.exists():
   audit=json.loads(audit_path.read_text());c=audit['counts'];splits=c['splits']
   lines.append(f"| {domain} | {c['items']} | {c['users']} | {c['events']} | {splits.get('train',0)} | {splits.get('validation',0)} | {splits.get('test',0)} |")
  result_path=root/'artifacts'/domain/'results.json'
  if result_path.exists():
   for row in json.loads(result_path.read_text()).get('results',[]):
    if row.get('split')=='test':
     if row.get('data_audit',{}).get('synthetic') or row.get('features_metadata',{}).get('synthetic') or row.get('run_protocol',{}).get('evaluation_subset'):
      raise ValueError(f'Synthetic/diagnostic record cannot enter formal report: {domain}')
     all_rows.append({'domain':domain,**row})
 lines+=['','## 训练设置与语言适配','',
 '主比较固定5个推荐训练epoch，batch size 256，种子42/43/44；每轮按完整验证集NDCG@10选择checkpoint。这是固定预算比较，不代表各方法均已充分调优。Popularity和TF-IDF无迭代训练。',
 '', 'MiniLM适配固定seed42、2个epoch、batch size 32、学习率2e-5、最多30000个训练期评论—描述对，只更新最后两个Transformer层。下表保留各轮实测验证损失，不是未适配模型的对照；本次未记录适配前的验证损失。', '',
 '| 数据域 | 适配训练对 | 适配验证对 | 第1轮验证损失 | 第2轮验证损失 |','|---|---:|---:|---:|---:|']
 for domain in ('baby','sports','clothing'):
  adaptation=root/'artifacts'/domain/'text_adaptation/adaptation.json'
  if adaptation.exists():
   meta=json.loads(adaptation.read_text());history=meta.get('history',[])
   losses=[f"{r['validation_loss']:.5f}" for r in history[:2]]
   lines.append(f"| {domain} | {meta['training_pairs']} | {meta['validation_pairs']} | "+' | '.join(losses+['待完成']*(2-len(losses)))+' |')
 lines+=['', '验证损失衡量评论与商品描述的语义对齐目标。即使这一目标在训练轮次间改善，也不能推断个性化推荐必然改善；需要独立看下游排序结果。', '', '## 实测结果','', '均值 ± 样本标准差；一条记录时只报告实测值，不生成标准差。','',
 '| 数据域 | 方法 | 种子数 | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |','|---|---|---:|---:|---:|---:|---:|']
 groups=defaultdict(list)
 for row in all_rows:groups[(row['domain'],row['model'])].append(row)
 for (domain,mode),rows in sorted(groups.items()):
  reference=rows[0]
  for row in rows[1:]:
   for key in ('items_sha256','events_sha256','selected_features_sha256','n_users','n_targets'):
    if row.get(key)!=reference.get(key):raise ValueError(f'Incompatible experimental records: {domain}/{mode}/{key}')
   for key in ('candidate_set','history','averaging','n_catalog_items','positive_feedback','max_history'):
    if row.get('run_protocol',{}).get(key)!=reference.get('run_protocol',{}).get(key):raise ValueError(f'Incompatible ranking protocol: {key}')
  seeds=[r['seed'] for r in rows]
  if len(seeds)!=len(set(seeds)):raise ValueError('Duplicate seed records in source results')
  entry={'domain':domain,'model':mode,'n_seeds':len(rows)};cells=[]
  for metric in ('recall@10','ndcg@10','recall@20','ndcg@20'):
   values=[r[metric] for r in rows];mean=statistics.mean(values);std=statistics.stdev(values) if len(values)>1 else None
   entry[metric]=mean;entry[metric+'_std']=std;cells.append(f'{mean:.5f} ± {std:.5f}' if std is not None else f'{mean:.5f}')
  lines.append(f"| {domain} | {LABELS.get(mode,mode)} | {len(rows)} | "+' | '.join(cells)+' |');summaries.append(entry)
 if not all_rows:lines+=['','当前没有正式测试集结果。不得将UI联调或随机特征测试当作推荐性能。']
 for domain in ('baby','sports','clothing'):
  comparable=[r for r in all_rows if r['domain']==domain]
  if comparable:
   reference=comparable[0]
   for row in comparable[1:]:
    for key in ('items_sha256','events_sha256','n_users','n_targets'):
     if row.get(key)!=reference.get(key):raise ValueError(f'Cross-model dataset mismatch: {domain}/{key}')
    for key in ('candidate_set','history','averaging','n_catalog_items','positive_feedback','max_history'):
     if row.get('run_protocol',{}).get(key)!=reference.get('run_protocol',{}).get(key):raise ValueError(f'Cross-model protocol mismatch: {domain}/{key}')
 lines+=['','## 结果解释','']
 for domain in ('baby','sports','clothing'):
  lookup={r['model']:r for r in summaries if r['domain']==domain}
  if 'text' in lookup and 'multimodal' in lookup:
   delta=lookup['multimodal']['ndcg@10']-lookup['text']['ndcg@10']
   lines.append(f"- {domain}：加入图像后的NDCG@10绝对变化为 {delta:+.5f}。")
  if 'multimodal' in lookup and 'adapted' in lookup:
   delta=lookup['adapted']['ndcg@10']-lookup['multimodal']['ndcg@10']
   lines.append(f"- {domain}：文本语义适配后的NDCG@10绝对变化为 {delta:+.5f}。")
 lines+=['','上述变化只适用于本次切分、模型和训练预算，没有显著性检验时不声称统计显著，不由此承诺线上效果或解释因果。','',
 '## 本地演示与限制','',
 '本地使用缓存向量和小型排序网络。新会话选择商品后直接使用与训练一致的历史编码器。推荐依据展示过去选品、原始文本和向量相似度，属于可检查的相关证据，不是因果解释或模型内部推理过程。','',
 '三域、六方法、每方法三个推荐种子齐全后才认为实验矩阵完成。真人配音视频、真实组员贡献与签名、AI声明确认由成员完成，系统不会代签或合成配音。','',
 '## 参考资料','',
 '- Amazon 2014: https://mcauleylab.ucsd.edu/public_datasets/data/amazon/datasets.html',
 '- CLIP: https://proceedings.mlr.press/v139/radford21a.html',
 '- CLIP weights: https://huggingface.co/openai/clip-vit-base-patch32',
 '- MiniLM: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2',
 '- BPR: https://arxiv.org/abs/1205.2618',
 '- Streamlit: https://docs.streamlit.io/','']
 sys.path.insert(0,str(root/'src'))
 from shoplens.report_analysis import build_analysis
 lines+=['',build_analysis(root)]
 benchmark=root/'artifacts/local_benchmark.json'
 if benchmark.exists():
  local=json.loads(benchmark.read_text())
  lines+=['','## 本机推理实测','',local['definition'],'',f"平台：{local['platform']}；CPU线程：{local['torch_threads']}；测量进程峰值RSS：{local['process_peak_rss_mib']:.1f} MiB。该内存不是整个浏览器与系统内存。",'', '| 数据域 | 方法 | 加载并首次推荐(ms) | 缓存推理中位数(ms) | P95(ms) |','|---|---|---:|---:|---:|']
  for r in local['records']:
   lines.append(f"| {r['domain']} | {LABELS.get(r['model'],r['model'])} | {r['cold_model_load_and_recommend_ms']:.2f} | {r['warm_median_ms']:.3f} | {r['warm_p95_ms']:.3f} |")
 (out/'实验报告.md').write_text('\n'.join(lines),encoding='utf-8')
 (out/'results_summary.json').write_text(json.dumps({'results':summaries,'raw_record_count':len(all_rows)},indent=2))
 if summaries:
  with (out/'results_summary.csv').open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
 print(json.dumps({'test_records':len(all_rows),'groups':len(summaries),'report':str(out/'实验报告.md')}))
if __name__=='__main__':main()
