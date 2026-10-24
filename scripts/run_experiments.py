#!/usr/bin/env python3
"""Run the actual three-domain experiments sequentially on one allocated GPU."""
from pathlib import Path
import argparse,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('HF_HOME',str(ROOT/'data/hf-cache'))
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--domains',nargs='+',default=['baby','sports','clothing'])
 p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
 p.add_argument('--epochs',type=int,default=5);p.add_argument('--adapt-epochs',type=int,default=2)
 p.add_argument('--adapt-pairs',type=int,default=30000);p.add_argument('--device',default='cuda')
 p.add_argument('--batch-size',type=int,default=256);p.add_argument('--resume',action='store_true')
 a=p.parse_args()
 import torch
 from shoplens.features import encode,sha256
 from shoplens.adapt import adapt
 from shoplens.train import train_model
 from shoplens.evaluate import evaluate_model
 torch.set_num_threads(min(int(os.environ.get('PBS_NCPUS','4')),8))
 environment={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'device':a.device,
   'gpu':torch.cuda.get_device_name(0) if a.device=='cuda' else None,'config':vars(a),'started_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 (ROOT/'experiments').mkdir(exist_ok=True)
 environment['code_sha256']={name:sha256(ROOT/'src/shoplens'/name) for name in ['models.py','train.py','evaluate.py','inference.py','adapt.py','features.py']}
 (ROOT/'experiments/environment.json').write_text(json.dumps(environment,indent=2))
 job_dir=ROOT/'experiments/jobs'/os.environ.get('PBS_JOBID','local')
 job_dir.mkdir(parents=True,exist_ok=True)
 (job_dir/'environment.json').write_text(json.dumps(environment,indent=2))
 for domain in a.domains:
  processed=ROOT/'data/processed'/domain;artifacts=ROOT/'artifacts'/domain
  print('DOMAIN_START',domain,flush=True)
  encode(processed,artifacts,ROOT,a.device,64)
  adapted=artifacts/'text_adaptation/model'
  adaptation_meta=artifacts/'text_adaptation/adaptation.json'
  if a.resume and (adapted/'modules.json').exists() and adaptation_meta.exists():
   previous=json.loads(adaptation_meta.read_text())
   if previous['source_reviews_sha256']!=sha256(processed/'reviews_train.jsonl') or previous['epochs']!=a.adapt_epochs or previous['max_pairs']!=a.adapt_pairs:
    raise ValueError('Adaptation cache does not match current data/config; use a new artifact directory')
  else:
   adapt(processed,artifacts/'text_adaptation',a.device,a.adapt_epochs,32,a.adapt_pairs,42)
  encode(processed,artifacts,ROOT,a.device,64,adapted)
  for seed in a.seeds:
   run_dir=ROOT/'experiments/runs'/domain/f'seed{seed}'
   completed=run_dir/'complete.json'
   fingerprint={'epochs':a.epochs,'batch_size':a.batch_size,'adapt_epochs':a.adapt_epochs,'adapt_pairs':a.adapt_pairs,'seed':seed,'items_sha256':sha256(processed/'items.jsonl'),'events_sha256':sha256(processed/'events.jsonl'),'features_sha256':sha256(artifacts/'features.npz')}
   if a.resume and completed.exists():
    if json.loads(completed.read_text()).get('fingerprint')!=fingerprint:
     raise ValueError(f'Resume fingerprint mismatch: {completed}')
    print('SKIP_COMPLETE',domain,seed,flush=True);continue
   started=time.time();run_dir.mkdir(parents=True,exist_ok=True)
   for mode in ['popularity','mf','tfidf','text','multimodal','adapted']:
    train_model(processed,artifacts,model=mode,seed=seed,epochs=a.epochs,batch_size=a.batch_size,device=a.device)
    evaluate_model(processed,artifacts,model=mode,split='test',seed=seed,device=a.device)
   shutil.copytree(artifacts/'models',run_dir/'models',dirs_exist_ok=True)
   all_results=json.loads((artifacts/'results.json').read_text())
   (run_dir/'results.json').write_text(json.dumps({'results':[r for r in all_results['results'] if r['seed']==seed]},indent=2,ensure_ascii=False))
   completed.write_text(json.dumps({'domain':domain,'seed':seed,'seconds':time.time()-started,'config':vars(a),'fingerprint':fingerprint},indent=2))
   print('SEED_COMPLETE',domain,seed,'seconds',round(time.time()-started,1),flush=True)
  # The UI uses a predeclared seed, never a seed chosen by the test result.
  default=ROOT/'experiments/runs'/domain/f'seed{a.seeds[0]}'/'models'
  if default.exists():shutil.copytree(default,artifacts/'models',dirs_exist_ok=True)
  print('DOMAIN_COMPLETE',domain,flush=True)
 (ROOT/'experiments/complete.json').write_text(json.dumps({'completed_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'config':vars(a)},indent=2))
if __name__=='__main__':main()
