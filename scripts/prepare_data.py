#!/usr/bin/env python3
"""Reproduce source downloads, image validation and portable catalog preparation."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
 from shoplens.data import DOMAINS,download_domain,prepare,audit_images,sha256_file
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
 p.add_argument('--domains',nargs='+',choices=list(DOMAINS),default=list(DOMAINS));p.add_argument('--download',action='store_true')
 p.add_argument('--workers',type=int,default=12);p.add_argument('--seed',type=int,default=42);a=p.parse_args();root=a.root.resolve()
 for domain in a.domains:
  raw=root/'data/raw';processed=root/'data/processed'/domain;images=root/'data/images'/domain
  if a.download:download_domain(domain,raw)
  stem=DOMAINS[domain];reviews=raw/f'reviews_{stem}_5.json.gz';metadata=raw/f'meta_{stem}.json.gz'
  prepare(domain,reviews,metadata,processed,require_text=True,require_image=True,images_dir=images,seed=a.seed)
  media=audit_images(processed,images_dir=images,limit=None,download=a.download,workers=a.workers)
  audit=prepare(domain,reviews,metadata,processed,require_text=True,require_images=True,images_dir=images,seed=a.seed)
  catalog=processed/'items.jsonl';rows=[json.loads(x) for x in catalog.read_text().splitlines() if x.strip()]
  for row in rows:row['image_path']=str(Path(row['image_path']).relative_to(root))
  catalog.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
  audit['catalog_sha256']=sha256_file(catalog);audit['image_audit_counts']=media['counts']
  (processed/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
  print(json.dumps({'domain':domain,'counts':audit['counts'],'image_audit_counts':media['counts']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
