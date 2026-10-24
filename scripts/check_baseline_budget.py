#!/usr/bin/env python3
"""Separate, predeclared MF budget check motivated by validation curves only."""
from pathlib import Path
import argparse,json,sys,shutil,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int,default=20)
 p.add_argument('--domains',nargs='+',default=['baby','sports','clothing']);p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);a=p.parse_args()
 import torch
 torch.set_num_threads(4)
 from shoplens.train import train_model
 from shoplens.evaluate import evaluate_model
 collected=[]
 for domain in a.domains:
  for seed in a.seeds:
   directory=ROOT/'artifacts/budget_check'/domain/f'seed{seed}'
   complete=directory/'budget_check.json'
   if complete.exists():
    previous=json.loads(complete.read_text())
    if previous['epochs']!=a.epochs:raise ValueError('Budget check config changed')
    collected.append(previous);continue
   started=time.time()
   metadata=train_model(ROOT/'data/processed'/domain,directory,model='mf',seed=seed,epochs=a.epochs,device=a.device)
   result=evaluate_model(ROOT/'data/processed'/domain,directory,model='mf',split='test',seed=seed,device=a.device)
   record={'domain':domain,'seed':seed,'epochs':a.epochs,'selected_epoch':metadata['selected_epoch'],'validation_curve':metadata['validation_ndcg@10'],'result':result,'reason':'MF validation NDCG was still rising at the fixed five-epoch primary budget; this is a separate sensitivity check, not a replacement of the primary table.','elapsed_seconds':time.time()-started}
   complete.write_text(json.dumps(record,indent=2));collected.append(record)
   print('BUDGET_COMPLETE',domain,seed,flush=True)
 (ROOT/'experiments/baseline_budget.json').write_text(json.dumps({'checks':collected},indent=2))
if __name__=='__main__':main()
