"""Adapt the final two MiniLM layers on TRAIN-period review/product text pairs."""
from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader,Dataset
from .features import TEXT_ID,read_items,choose_device,sha256

class Pairs(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.rows[i]

def _encode(model,texts,device):
    encoded={k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in model.tokenize(texts).items()}
    return torch.nn.functional.normalize(model(encoded)['sentence_embedding'],dim=-1)

def adapt(processed_dir,output_dir,device='auto',epochs=1,batch_size=32,max_pairs=30000,seed=42,lr=2e-5):
    if epochs < 1 or batch_size < 2 or max_pairs < 4:
        raise ValueError('epochs>=1, batch_size>=2 and max_pairs>=4 are required')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device=choose_device(device); processed_dir=Path(processed_dir); output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    items={r['item_id']:r for r in read_items(processed_dir)}
    audit=json.loads((processed_dir/'audit.json').read_text())
    cutoff=audit.get('cutoffs',{}).get('train') or audit.get('train_cutoff') or audit.get('cutoffs',{}).get('train_end')
    pairs=[]
    with (processed_dir/'reviews_train.jsonl').open() as review_stream:
        review_lines=list(review_stream)
    for line in review_lines:
        r=json.loads(line); text=(r.get('text') or '').strip(); item=items.get(r['item_id'])
        if cutoff is not None and r['timestamp']>cutoff:raise ValueError('Training review exceeds training cutoff')
        if item and len(text.split())>=5 and float(r.get('rating',0))>=4:
            pairs.append((r['item_id'],text,item.get('text') or item['title']))
    if len(pairs)<4:raise ValueError('Too few real training pairs for text adaptation')
    random.shuffle(pairs); pairs=pairs[:max_pairs]
    # Item-disjoint adaptation validation; downstream validation/test reviews are never read.
    item_ids=sorted({p[0] for p in pairs}); random.shuffle(item_ids)
    if len(item_ids)<4:raise ValueError('Need at least four distinct items for item-disjoint adaptation validation')
    held=set(item_ids[:max(2,len(item_ids)//10)])
    train=[p for p in pairs if p[0] not in held]; valid=[p for p in pairs if p[0] in held]
    model=SentenceTransformer(TEXT_ID,device=device); model.max_seq_length=256
    auto=model[0].auto_model
    for p in model.parameters():p.requires_grad=False
    layers=auto.encoder.layer
    for layer in layers[-2:]:
        for p in layer.parameters():p.requires_grad=True
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=.01)
    # A batch contains at most one pair per item to avoid false negatives for the same ASIN.
    def batch_loss(batch,training):
        seen=set(); unique=[]
        for p in batch:
            if p[0] not in seen:seen.add(p[0]);unique.append(p)
        if len(unique)<2:return None
        q=_encode(model,[p[1] for p in unique],device)
        d=_encode(model,[p[2] for p in unique],device)
        logits=q@d.T/.07; labels=torch.arange(len(unique),device=device)
        return (torch.nn.functional.cross_entropy(logits,labels)+torch.nn.functional.cross_entropy(logits.T,labels))/2
    best=float('inf'); history=[]; started=time.time()
    for epoch in range(epochs):
        model.train(); random.shuffle(train); losses=[]
        for start in range(0,len(train),batch_size):
            loss=batch_loss(train[start:start+batch_size],True)
            if loss is None:continue
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();losses.append(float(loss.detach().cpu()))
            if start%(batch_size*50)==0:print(f'adapt epoch={epoch+1} pairs={start}/{len(train)} loss={losses[-1]:.4f}',flush=True)
        model.eval();vl=[]
        with torch.inference_mode():
            for start in range(0,min(len(valid),4096),batch_size):
                loss=batch_loss(valid[start:start+batch_size],False)
                if loss is not None:vl.append(float(loss.cpu()))
        if not losses or not vl:raise ValueError('No valid contrastive training/validation batches')
        metric=float(np.mean(vl));history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation_loss':metric})
        if metric<best:
            best=metric; model.save(str(output_dir/'model'))
        print(json.dumps(history[-1]),flush=True)
    record={'base_model':TEXT_ID,'trainable_layers':'last two Transformer blocks','objective':'symmetric in-batch contrastive review-description loss','seed':seed,'epochs':epochs,'batch_size':batch_size,'max_pairs':max_pairs,'training_pairs':len(train),'validation_pairs':len(valid),'validation_protocol':'item-disjoint partition of training-period reviews','history':history,'source_reviews_sha256':sha256(processed_dir/'reviews_train.jsonl'),'real_data':True,'elapsed_seconds':time.time()-started}
    (output_dir/'adaptation.json').write_text(json.dumps(record,indent=2))
    return output_dir/'model'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--processed-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--device',default='auto');p.add_argument('--epochs',type=int,default=1);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--max-pairs',type=int,default=30000);p.add_argument('--seed',type=int,default=42);p.add_argument('--lr',type=float,default=2e-5)
    a=p.parse_args();adapt(a.processed_dir,a.output_dir,a.device,a.epochs,a.batch_size,a.max_pairs,a.seed,a.lr)
if __name__=='__main__':main()
