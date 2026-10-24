"""Encode real catalog data with frozen CLIP and MiniLM; no synthetic fallback."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

CLIP_ID = "openai/clip-vit-base-patch32"
TEXT_ID = "sentence-transformers/all-MiniLM-L6-v2"

def read_items(processed_dir):
    with (Path(processed_dir) / "items.jsonl").open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError("Empty catalog")
    if len({r['item_id'] for r in rows}) != len(rows):
        raise ValueError("Duplicate item IDs")
    return rows

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def choose_device(name):
    import torch
    if name != 'auto':
        if name == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
        if name == 'mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS requested but unavailable')
        return name
    return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def encode_texts(model, texts, batch_size=64):
    # Explicitly split long descriptions instead of silently discarding the end.
    tokenizer=model.tokenizer
    chunks=[]; owners=[]
    for i,text in enumerate(texts):
        ids=tokenizer.encode(str(text),add_special_tokens=False)
        for start in range(0,max(len(ids),1),224):
            chunks.append(tokenizer.decode(ids[start:start+224],skip_special_tokens=True) or '[empty]')
            owners.append(i)
    vectors=model.encode(chunks,batch_size=batch_size,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True)
    out=np.zeros((len(texts),vectors.shape[1]),dtype=np.float32)
    np.add.at(out,np.asarray(owners),vectors)
    out/=np.maximum(np.linalg.norm(out,axis=1,keepdims=True),1e-12)
    return out

def encode(processed_dir,artifact_dir,root,device='auto',batch_size=32,adapted_model=None):
    import torch
    from PIL import Image,ImageOps
    from sentence_transformers import SentenceTransformer
    from transformers import CLIPModel,CLIPProcessor
    processed_dir,artifact_dir,root=map(Path,(processed_dir,artifact_dir,root))
    artifact_dir.mkdir(parents=True,exist_ok=True)
    rows=read_items(processed_dir); ids=np.asarray([r['item_id'] for r in rows])
    device=choose_device(device); start=time.time()
    destination=artifact_dir/'features.npz'; meta_path=artifact_dir/'features.json'
    catalog_hash=sha256(processed_dir/'items.jsonl')
    existing={}
    previous_meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if destination.exists() and previous_meta:
        if previous_meta.get('features_sha256') != sha256(destination):
            raise ValueError('Cached feature checksum mismatch')
        if previous_meta.get('clip_model') != CLIP_ID or previous_meta.get('text_model') != TEXT_ID:
            raise ValueError('Cached encoder model IDs mismatch')
    if destination.exists() and not previous_meta:
        raise ValueError('Existing features lack provenance metadata; use a new artifact directory')
    if destination.exists():
        with np.load(destination,allow_pickle=False) as old:
            if np.array_equal(old['item_ids'],ids): existing={k:old[k].copy() for k in old.files}
        if not existing: raise ValueError('Existing features have different item IDs; use a new artifact directory')
    metadata=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if metadata and metadata.get('catalog_sha256')!=catalog_hash:
        raise ValueError('Catalog content changed: use a new artifact directory to avoid stale features')
    texts=[r.get('text') or (r.get('title','')+' '+r.get('description','')) for r in rows]
    if 'text' not in existing:
        text_model=SentenceTransformer(TEXT_ID,device=device)
        existing['text']=encode_texts(text_model,texts,batch_size)
        del text_model
    if 'image' not in existing:
        processor=CLIPProcessor.from_pretrained(CLIP_ID,use_fast=False)
        clip=CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
        blocks=[]; mask=[]
        for start_i in range(0,len(rows),batch_size):
            pictures=[]
            for row in rows[start_i:start_i+batch_size]:
                path=Path(row.get('image_path') or '__missing__')
                path=path if path.is_absolute() else root/path
                try:
                    with Image.open(path) as im: pictures.append(ImageOps.exif_transpose(im).convert('RGB').copy())
                    mask.append(True)
                except (OSError,ValueError) as e:
                    # Missing media is recorded; fail rather than encode blank placeholder images.
                    raise RuntimeError(f"Missing/invalid real image for {row['item_id']}: {path}") from e
            inputs=processor(images=pictures,return_tensors='pt').to(device)
            with torch.inference_mode():
                vec=clip.get_image_features(**inputs)
                if not isinstance(vec,torch.Tensor): vec=vec.pooler_output
                vec=torch.nn.functional.normalize(vec.float(),dim=-1)
            blocks.append(vec.cpu().numpy())
            if start_i%max(batch_size*20,1)==0: print(f'image encoding {start_i}/{len(rows)}',flush=True)
        existing['image']=np.concatenate(blocks).astype(np.float32)
        existing['image_present']=np.asarray(mask)
        del clip
    if adapted_model:
        adapted=SentenceTransformer(str(adapted_model),device=device)
        existing['adapted_text']=encode_texts(adapted,texts,batch_size)
        metadata['adapted_model']=str(adapted_model)
        adaptation_path=Path(adapted_model).parent/'adaptation.json'
        if adaptation_path.exists():
            metadata['adaptation']=json.loads(adaptation_path.read_text())
            metadata['adaptation_sha256']=sha256(adaptation_path)
        del adapted
    for key in ('image','text','adapted_text'):
        if key in existing and (len(existing[key])!=len(rows) or not np.isfinite(existing[key]).all()):
            raise ValueError(f'Invalid feature matrix: {key}')
    tmp=destination.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,item_ids=ids,**{k:v for k,v in existing.items() if k!='item_ids'})
    tmp.replace(destination)
    metadata.update({'schema_version':1,'real_data':True,'synthetic':False,'clip_model':CLIP_ID,'text_model':TEXT_ID,
        'catalog_sha256':catalog_hash,'n_items':len(rows),'text_chunk_wordpieces':224,'text_chunk_aggregation':'normalized mean',
        'image_preprocessing':'CLIPProcessor + EXIF transpose','device':device,'elapsed_seconds':time.time()-start,
        'features_sha256':sha256(destination)})
    meta_path.write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
    print(json.dumps(metadata,ensure_ascii=False),flush=True)
    return destination

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--processed-dir',required=True); p.add_argument('--artifact-dir',required=True)
    p.add_argument('--root',default='.'); p.add_argument('--device',default='auto',choices=['auto','cpu','mps','cuda'])
    p.add_argument('--batch-size',type=int,default=32); p.add_argument('--adapted-model')
    a=p.parse_args(); encode(a.processed_dir,a.artifact_dir,a.root,a.device,a.batch_size,a.adapted_model)
if __name__=='__main__': main()
