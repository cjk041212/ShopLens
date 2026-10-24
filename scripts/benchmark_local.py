#!/usr/bin/env python3
"""Measure actual local cached recommendation latency, not browser end-to-end time."""
from pathlib import Path
import argparse,gc,json,platform,resource,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
 import numpy as np,torch
 from shoplens.inference import available_models,recommend,_cached_scorer
 from shoplens.models import load_items
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--domains',nargs='+',default=['baby','sports','clothing']);p.add_argument('--repeats',type=int,default=20);a=p.parse_args()
 torch.set_num_threads(4);records=[]
 for domain in a.domains:
  processed=ROOT/'data/processed'/domain;artifact=ROOT/'artifacts'/domain
  items=load_items(processed);liked=[r['item_id'] for r in items[:3]]
  for mode in available_models(artifact):
   _cached_scorer.cache_clear();gc.collect()
   start=time.perf_counter();first=recommend(processed,artifact,mode,liked,10);cold=time.perf_counter()-start
   durations=[]
   for _ in range(a.repeats):
    start=time.perf_counter();result=recommend(processed,artifact,mode,liked,10);durations.append(time.perf_counter()-start)
    assert len(result)==10 and not set(liked)&{r['item_id'] for r in result}
   records.append({'domain':domain,'model':mode,'device':'cpu','n_catalog_items':len(items),'n_liked_items':3,'cold_model_load_and_recommend_ms':cold*1000,'warm_median_ms':float(np.median(durations)*1000),'warm_p95_ms':float(np.quantile(durations,.95)*1000),'repeats':a.repeats})
   print(json.dumps(records[-1]),flush=True)
 peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 peak_mib=peak/(1024**2) if sys.platform=='darwin' else peak/1024
 output={'platform':platform.platform(),'architecture':platform.machine(),'torch':torch.__version__,'torch_threads':4,'process_peak_rss_mib':peak_mib,'definition':'CPU inference API with 3 selected catalog items and top10 output. Warm measurements reuse loaded scorer; excludes browser rendering and encoder execution. Cold includes model load and feature checksum verification, after Python/Torch imports.','records':records}
 path=ROOT/'artifacts/local_benchmark.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(output,indent=2))
if __name__=='__main__':main()
