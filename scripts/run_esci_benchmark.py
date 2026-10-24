#!/usr/bin/env python3
"""Prepare and run a predeclared, real 500-query ESCI transfer benchmark."""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shoplens.search_benchmark import (MODELS, SOURCE_COMMIT, SOURCE_FILES, SOURCE_REPOSITORY,
    TextEncoder, canonical_hash, declared_protocol, evaluate_scores, file_sha,
    make_queries, select_query_ids, tfidf_scores, validate_model_exports)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def write_jsonl(path, rows):
    # Real product titles contain U+2028/U+0085. Escape them so generic JSONL
    # consumers using splitlines() cannot split the middle of a JSON string.
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=True, allow_nan=False) + '\n' for row in rows))


def media_url(name):
    return f'https://media.githubusercontent.com/media/amazon-science/esci-data/{SOURCE_COMMIT}/shopping_queries_dataset/{name}'


class RangeFile(io.RawIOBase):
    """Seekable bounded HTTP range reader with content-addressed resume chunks."""
    def __init__(self, url, size, cache, budget=350 * 1024**2):
        import requests
        self.url, self.size, self.cache, self.budget = url, size, Path(cache), budget
        self.cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.position = 0
        self.network_bytes = 0
        self.records = {}

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        value = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if value < 0: raise ValueError('Negative seek')
        self.position = value
        return value

    def read(self, size=-1):
        import hashlib
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size <= 0: return b''
        start, end = self.position, self.position + size - 1
        cache = self.cache / f'{start}-{end}.bin'
        receipt = cache.with_suffix('.json')
        data = None
        if cache.exists() and receipt.exists():
            record = json.loads(receipt.read_text())
            value = cache.read_bytes()
            if len(value) == size and hashlib.sha256(value).hexdigest() == record['sha256']:
                data = value
        if data is None:
            if self.network_bytes + size > self.budget:
                raise RuntimeError('Product column range-download budget exceeded; refusing full-file download')
            error = None
            for attempt in range(3):
                try:
                    with self.session.get(self.url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'}, timeout=120, stream=True) as r:
                        if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                            raise RuntimeError(f'Server did not honor exact range: HTTP {r.status_code}, {r.headers.get("Content-Range")}')
                        data = r.raw.read(size + 1)
                        if len(data) != size: raise RuntimeError('Truncated product range')
                    break
                except Exception as exc:
                    error = exc
                    if attempt == 2: raise error
                    time.sleep(1 + attempt)
            self.network_bytes += len(data)
            cache.write_bytes(data)
            write_json(receipt, {'start': start, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        self.records[f'{start}-{end}'] = {'start': start, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        self.position += size
        return data


def download_examples(data_dir):
    import requests
    info = SOURCE_FILES['examples']
    path = data_dir / info['name']
    if path.exists() and path.stat().st_size == info['bytes'] and file_sha(path) == info['sha256']:
        return path, 0
    temporary = path.with_suffix('.part')
    # Only a SHA-verified complete file is reused. A partial download resumes with Range.
    start = temporary.stat().st_size if temporary.exists() else 0
    if start > info['bytes']:
        raise ValueError('Unexpected partial examples file size')
    with requests.get(media_url(info['name']), headers={'Range': f'bytes={start}-{info["bytes"]-1}', 'Accept-Encoding': 'identity'}, stream=True, timeout=120) as r:
        if r.status_code not in (200, 206): r.raise_for_status()
        if start and (r.status_code != 206 or not r.headers.get('Content-Range', '').startswith(f'bytes {start}-')):
            raise RuntimeError('Server refused examples resume; partial file preserved')
        with temporary.open('ab' if start else 'wb') as f:
            for block in r.iter_content(1024 * 1024):
                f.write(block)
                if f.tell() > info['bytes']: raise RuntimeError('Examples exceed declared source bytes')
    if temporary.stat().st_size != info['bytes'] or file_sha(temporary) != info['sha256']:
        raise ValueError('Official examples full SHA256 mismatch')
    temporary.replace(path)
    return path, info['bytes'] - start


def prepare(root, output, data_dir):
    import pyarrow.parquet as pq
    protocol = declared_protocol()
    previous_source = json.loads((output/'source_provenance.json').read_text()) if (output/'source_provenance.json').exists() else {}
    exports = validate_model_exports(root)
    declaration = {'protocol': protocol, 'protocol_sha256': canonical_hash(protocol), 'model_exports': exports,
                   'declaration': 'Written before loading ESCI test relevance labels. Parameters and Clothing transfer adapters are fixed.'}
    prereg = output / 'protocol.json'
    if prereg.exists() and json.loads(prereg.read_text()) != declaration:
        raise ValueError('Existing predeclaration differs; refuse silent protocol replacement')
    write_json(prereg, declaration)
    data_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(data_dir).free < 600 * 1024**2:
        raise RuntimeError('Need 600MiB free for bounded source/cache preparation')
    examples, downloaded = download_examples(data_dir)
    filters = [('small_version', '=', 1), ('product_locale', '=', 'us')]
    structural = pq.read_table(examples, columns=['query_id', 'product_id', 'split'], filters=filters).to_pylist()
    test_ids = select_query_ids(structural, 'test')
    train_ids = select_query_ids(structural, 'train')
    from collections import defaultdict
    counts = defaultdict(set)
    for r in structural:
        counts[(r['split'], str(r['query_id']))].add(str(r['product_id']))
    selection_counts = {split: {'official_queries':sum(s==split for s,q in counts),
        'eligible_queries_2_to_40':sum(s==split and 2<=len(ids)<=40 for (s,q),ids in counts.items()),
        'excluded_query_count_by_candidate_count':sum(s==split and not 2<=len(ids)<=40 for (s,q),ids in counts.items())}
        for split in ('train','test')}
    if set(test_ids) & set(train_ids): raise ValueError('Train/test query ID overlap')
    write_json(output / 'selected_query_ids.json', {'test': test_ids, 'train': train_ids,
               'selection_uses_labels': False, 'protocol_sha256': declaration['protocol_sha256']})
    numeric_ids = [int(x) for x in test_ids + train_ids]
    rows = pq.read_table(examples, columns=['query_id','query','product_id','split','esci_label'],
                         filters=filters + [('query_id', 'in', numeric_ids)]).to_pylist()
    wanted = {str(r['product_id']) for r in rows}
    product_info = SOURCE_FILES['products']
    remote = RangeFile(media_url(product_info['name']), product_info['bytes'], data_dir / 'product-ranges')
    parquet = pq.ParquetFile(remote, pre_buffer=False)
    products = {}
    for rg in range(parquet.num_row_groups):
        keys = parquet.read_row_group(rg, columns=['product_id','product_locale']).to_pydict()
        indices = [i for i, (pid, locale) in enumerate(zip(keys['product_id'], keys['product_locale'])) if locale == 'us' and pid in wanted]
        if indices:
            titles = parquet.read_row_group(rg, columns=['product_title']).column('product_title').to_pylist()
            for i in indices:
                pid = keys['product_id'][i]
                if pid in products: raise ValueError('Duplicate locale/product source key')
                products[pid] = titles[i]
        print(f'Product row group {rg+1}/{parquet.num_row_groups}: matched {len(products)}/{len(wanted)} titles, network {remote.network_bytes} bytes', flush=True)
    if set(products) != wanted: raise ValueError('Official product join missing selected product IDs')
    test_rows = [r for r in rows if r['split'] == 'test']
    train_rows = [{k:v for k,v in r.items() if k != 'esci_label'} for r in rows if r['split'] == 'train']
    queries = make_queries(test_rows, test_ids, products)
    training = make_queries(train_rows, train_ids, products)
    write_jsonl(output / 'test_inputs.jsonl', queries)
    write_jsonl(output / 'tfidf_training_inputs.jsonl', training)
    source = {'repository': SOURCE_REPOSITORY, 'commit': SOURCE_COMMIT,
              'examples': {**SOURCE_FILES['examples'], 'full_file_sha256_verified': True, 'bytes_downloaded_this_prepare': downloaded},
              'products': {**product_info, 'full_file_sha256_verified': False,
                  'verification_note': 'Immutable pinned upstream LFS SHA256 recorded, but only required Parquet byte ranges read. Actual extracted titles and every cached range have their own SHA256.',
                  'columns_read': ['product_id','product_locale','product_title'], 'range_network_bytes_this_prepare': remote.network_bytes,
                  'unique_cached_ranges': len(remote.records), 'unique_cached_range_bytes': sum(r['bytes'] for r in remote.records.values()),
                  'extracted_product_count': len(products), 'extracted_products_sha256': canonical_hash(products)},
              'test_inputs_sha256': file_sha(output / 'test_inputs.jsonl'),
              'tfidf_training_inputs_sha256': file_sha(output / 'tfidf_training_inputs.jsonl'),
              'query_ids_sha256': canonical_hash(test_ids),
              'selection_counts': selection_counts,
              'mapping_sources': {}}
    old_examples=previous_source.get('examples',{})
    old_products=previous_source.get('products',{})
    source['examples']['bytes_downloaded_cumulative']=downloaded+old_examples.get('bytes_downloaded_cumulative',old_examples.get('bytes_downloaded_this_prepare',0))
    source['products']['range_network_bytes_cumulative']=remote.network_bytes+old_products.get('range_network_bytes_cumulative',old_products.get('range_network_bytes_this_prepare',0))
    import requests
    for name in ['train.py', 'prepare_trec_eval_files.py']:
        url = f'https://raw.githubusercontent.com/amazon-science/esci-data/{SOURCE_COMMIT}/ranking/{name}'
        r = requests.get(url, timeout=60); r.raise_for_status()
        import hashlib
        source['mapping_sources'][name] = {'url': url, 'sha256': hashlib.sha256(r.content).hexdigest()}
    write_json(output / 'source_provenance.json', source)
    write_json(data_dir / 'product-range-receipts.json', list(remote.records.values()))
    print(json.dumps({'prepared_test_queries':len(queries),'test_candidates':sum(len(q['candidates']) for q in queries),
                      'unique_products_train_and_test':len(products),'network_bytes':downloaded+remote.network_bytes}), flush=True)


def run(root, output, data_dir, manifest, device, batch_size):
    import platform
    from importlib.metadata import version
    import torch
    started = time.perf_counter()
    declaration = json.loads((output / 'protocol.json').read_text())
    protocol = declared_protocol()
    if declaration['protocol'] != protocol or declaration['model_exports'] != validate_model_exports(root):
        raise ValueError('Protocol/model exports changed after predeclaration')
    source = json.loads((output / 'source_provenance.json').read_text())
    base_manifest = json.loads(Path(manifest).read_text())
    base_hashes = base_manifest['model_files']['qwen']
    if base_hashes.get('model.safetensors', {}).get('sha256') != '0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd':
        raise ValueError('Prepared weight manifest does not match the recorded pinned Qwen base')
    for name, key in [('test_inputs.jsonl','test_inputs_sha256'), ('tfidf_training_inputs.jsonl','tfidf_training_inputs_sha256')]:
        if file_sha(output/name) != source[key]: raise ValueError('Prepared selected source input hash changed')
    def read(name):
        with (output/name).open(encoding='utf-8') as stream:
            return [json.loads(line) for line in stream if line.strip()]
    queries, training = read('test_inputs.jsonl'), read('tfidf_training_inputs.jsonl')
    if len(queries) != 500 or len(training) != 500: raise ValueError('Formal run requires exactly 500 test and train queries')
    scores = {'tfidf': tfidf_scores(training, queries)}
    products = {c['product_id']:c['product_title'] for q in queries for c in q['candidates']}
    product_ids = sorted(products)
    product_index = {pid:i for i,pid in enumerate(product_ids)}
    data_dir.mkdir(parents=True, exist_ok=True)
    encoder = None
    timings = {}
    for mode in MODELS[1:]:
        begin = time.perf_counter()
        cache = data_dir / f'{mode}_features.npz'
        cache_meta = cache.with_suffix('.json')
        identity = {'protocol_sha256': declaration['protocol_sha256'], 'export': declaration['model_exports'][mode],
                    'input_sha256': source['test_inputs_sha256'], 'base_model_hashes':base_hashes,
                    'device': device, 'dtype': 'float32', 'batch_size': batch_size}
        used_cache = False
        if cache.exists() and cache_meta.exists():
            record = json.loads(cache_meta.read_text())
            if record.get('identity') == identity and record.get('sha256') == file_sha(cache):
                with np.load(cache, allow_pickle=False) as a:
                    query_vectors, titles = a['query'], a['title']
                used_cache = True
        if not used_cache:
            if encoder is None:
                print(f'Loading one actual local Qwen base on {device}, 4 CPU threads', flush=True)
                encoder = TextEncoder(root, manifest, device, batch_size)
            query_vectors = encoder.encode(mode, [q['query'] for q in queries], query=True)
            titles = encoder.encode(mode, [products[pid] for pid in product_ids])
            np.savez_compressed(cache, query=query_vectors, title=titles)
            write_json(cache_meta, {'identity': identity, 'sha256': file_sha(cache)})
        for vectors, count in [(query_vectors,len(queries)),(titles,len(product_ids))]:
            if vectors.shape != (count,1024) or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-4):
                raise ValueError('Invalid computed/cached real embedding matrix')
        scores[mode] = [titles[[product_index[c['product_id']] for c in q['candidates']]] @ query_vectors[i] for i,q in enumerate(queries)]
        timings[mode] = {'seconds': time.perf_counter()-begin,'feature_cache_reused':used_cache,
                         'feature_cache_sha256':file_sha(cache)}
        print(f'{mode} complete: {timings[mode]["seconds"]:.2f}s', flush=True)
    rows, model_results = evaluate_scores(queries, scores)
    write_jsonl(output / 'queries.jsonl', rows)
    source_catalog = set(json.loads((root/'models/qwen/clothing/qwen_semantic/seed42/head.json').read_text())['item_ids'])
    overlap = set(products) & source_catalog
    summary = {'schema_version':1,'benchmark_id':protocol['benchmark_id'],'complete':True,'diagnostic':False,
               'protocol':protocol,'protocol_sha256':declaration['protocol_sha256'],'source_provenance':source,
               'models':model_results,'model_exports':declaration['model_exports'],'query_count':len(queries),
               'base_model_hashes':base_hashes,
               'candidate_count':sum(len(q['candidates']) for q in queries),'unique_test_products':len(products),
               'tfidf_training_unique_products':len({c['product_id'] for q in training for c in q['candidates']}),
               'source_catalog_overlap':{'clothing_training_catalog_products':len(source_catalog),
                   'unique_test_products_also_in_source_catalog':len(overlap),
                   'test_candidate_occurrences_also_in_source_catalog':sum(c['product_id'] in overlap for q in queries for c in q['candidates']),
                   'note':'Product-ID overlap with the original adapter catalogue is reported, not filtered. This is not a guarantee of unseen-product transfer.'},
               'query_ids_sha256':source['query_ids_sha256'],'queries_sha256':file_sha(output/'queries.jsonl'),
               'zero_ideal_gain_queries':sum(not any(c['gain'] for c in q['candidates']) for q in queries),
               'no_exact_queries':sum(not any(c['esci_label']=='E' for c in q['candidates']) for q in queries),
               'runtime':{'device':device,'dtype':'float32','cpu_threads':4,'batch_size':batch_size,'python':platform.python_version(),
                          'torch':torch.__version__,'versions':{name:version(name) for name in ['numpy','transformers','peft','scikit-learn','pyarrow']},
                          'seconds':time.perf_counter()-started,'encoding':timings},
               'limitations':['A label-blind deterministic subset of 500 official English Task1 test queries, not the complete public test set.',
                 'Each model ranks the same small official per-query candidate pool. This does not measure full-catalogue retrieval.',
                 'Clothing seed42 transfer adapters are fixed in advance, trained on a different historical recommendation task, and not fine-tuned on ESCI.',
                 'Title-only text evaluation. No product images, preference histories, user heads or online outcomes.',
                 'NDCG uses the declared training-code gain mapping. Reversed S/C qrel-script gains are reported separately as a sensitivity.',
                 'The existing 27 history recommendation evaluations remain separate. These results do not replace their protocol or scores.']}
    write_json(output/'summary.json',summary)
    print(json.dumps({'complete':True,'query_count':500,'models':model_results,'seconds':summary['runtime']['seconds']},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--phase',choices=['prepare','run','all'],default='all')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--data-dir',type=Path,required=True,help='Local bounded source/range/feature cache, excluded from public result JSON.')
    parser.add_argument('--model-cache-manifest',type=Path)
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    root=args.root.expanduser().resolve(); output=(args.output or root/'results/esci').resolve()
    output.mkdir(parents=True,exist_ok=True)
    if args.phase in ('prepare','all'): prepare(root,output,args.data_dir)
    if args.phase in ('run','all'):
        if args.model_cache_manifest is None: parser.error('--model-cache-manifest is required for real encoding')
        run(root,output,args.data_dir,args.model_cache_manifest,args.device,args.batch_size)


if __name__=='__main__': main()
