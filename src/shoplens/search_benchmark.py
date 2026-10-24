"""Labelled ESCI candidate-list ranking, separate from history recommendation.

The primary gain mapping follows Amazon's ranking/train.py. The official qrel
export script reverses S/C positions, so that mapping is a named sensitivity
result, never silently substituted for the declared primary metric.
"""
from __future__ import annotations

import hashlib
import copy
import json
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

MODELS = ('tfidf', 'qwen_frozen', 'qwen_semantic', 'qwen_preference')
LABELS = {'tfidf': 'TF-IDF', 'qwen_frozen': 'Frozen Qwen',
          'qwen_semantic': 'Clothing semantic LoRA', 'qwen_preference': 'Clothing preference LoRA'}
GAINS = {'E': 1.0, 'S': .1, 'C': .01, 'I': 0.0}
QRELS_SCRIPT_GAINS = {'E': 1.0, 'S': .01, 'C': .1, 'I': 0.0}
SOURCE_COMMIT = '7916cdf6ab75a462e77f20ab40428a10923998d5'
SOURCE_REPOSITORY = 'https://github.com/amazon-science/esci-data'
MODEL_ID = 'Qwen/Qwen3-Embedding-0.6B'
REVISION = '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'
INSTRUCTION = "Given a shopping request, retrieve product descriptions that match the user's needs."
SOURCE_FILES = {
    'examples': {'name': 'shopping_queries_dataset_examples.parquet', 'bytes': 51286808,
                 'sha256': '4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a'},
    'products': {'name': 'shopping_queries_dataset_products.parquet', 'bytes': 1108857465,
                 'sha256': '25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265'},
}


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def file_sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tree_sha(directory) -> str:
    h = hashlib.sha256()
    directory = Path(directory)
    for path in sorted(directory.rglob('*')):
        if path.is_file():
            h.update(str(path.relative_to(directory)).encode())
            h.update(file_sha(path).encode())
    return h.hexdigest()


def declared_protocol() -> dict:
    return {
        'schema_version': 1, 'benchmark_id': 'esci_title_reranking_v1',
        'dataset': 'Amazon Shopping Queries / ESCI Task 1, reduced version',
        'source_repository': SOURCE_REPOSITORY, 'source_commit': SOURCE_COMMIT,
        'locale': 'us', 'small_version': 1, 'test_queries': 500, 'tfidf_train_queries': 500,
        'selection': 'Sort eligible query IDs by SHA256(shoplens-esci-v1|42|split|query_id), take first 500, independent of labels and text.',
        'seed': 42, 'candidate_min': 2, 'candidate_max': 40,
        'candidate_policy': 'Complete official query candidate list, without pruning, reretrieval, or adding unjudged items.',
        'product_fields': ['product_title'], 'images_used': False,
        'models': list(MODELS), 'adapter_domain': 'clothing', 'adapter_seed': 42,
        'adapter_selection': 'Clothing seed42 selected exports fixed before ESCI test labels are read; no best-domain selection.',
        'dense_score': 'Cosine of matching base/LoRA query and title embeddings, without recommendation heads or image signals.',
        'query_instruction': INSTRUCTION, 'document_instruction': None,
        'encoding': {'max_length': 256, 'padding_side': 'left', 'pooling': 'last valid token', 'normalization': 'L2'},
        'tfidf': {'ngram_range': [1, 2], 'lowercase': True, 'sublinear_tf': True,
                  'max_features': 100000, 'norm': 'l2', 'fit': 'Distinct candidate titles of the preselected 500 official training queries only.'},
        'configuration_selection': 'Fixed settings, no ESCI tuning or ESCI fine-tuning. No development subset required.',
        'primary_metric': 'ndcg@10', 'secondary_metrics': ['ndcg@all', 'mrr_exact@10', 'recall_exact@10'],
        'gains': GAINS, 'dcg': 'sum(gain / log2(rank+1)), using linear gains, not 2**gain-1',
        'gain_mapping_source': f'{SOURCE_REPOSITORY}/blob/{SOURCE_COMMIT}/ranking/train.py#L45-L50',
        'gain_mapping_discrepancy': 'prepare_trec_eval_files.py maps S to qrel 2 and C to 3, then declares qrel gains 0.01 and 0.1 respectively. Report that reversed S/C mapping as a separate sensitivity.',
        'sensitivity_gains': QRELS_SCRIPT_GAINS,
        'ties': 'Descending score, ascending product_id', 'aggregation': 'Unweighted arithmetic mean over the 500 selected test queries',
        'zero_ideal_gain': 'NDCG=0, retained and counted', 'no_exact_items': 'Exact MRR and recall=0, retained and counted',
        'scope': 'A deterministic 500-query labelled candidate-list transfer benchmark, not full-catalogue retrieval or a replication of the published ESCI baseline.',
    }


def select_query_ids(rows, split: str, count=500, seed=42, minimum=2, maximum=40) -> list[str]:
    """Selection touches only query IDs, split and candidate counts, not labels."""
    groups = defaultdict(set)
    for row in rows:
        if row['split'] == split:
            groups[str(row['query_id'])].add(str(row['product_id']))
    eligible = [qid for qid, ids in groups.items() if minimum <= len(ids) <= maximum]
    if len(eligible) < count:
        raise ValueError(f'Need {count} eligible {split} queries, found {len(eligible)}')
    key = lambda qid: (hashlib.sha256(f'shoplens-esci-v1|{seed}|{split}|{qid}'.encode()).hexdigest(), qid)
    return sorted(eligible, key=key)[:count]


def make_queries(rows, query_ids, products: dict) -> list[dict]:
    selected = set(query_ids)
    groups = defaultdict(list)
    texts = {}
    for row in rows:
        qid = str(row['query_id'])
        if qid not in selected:
            continue
        query = str(row['query'])
        if qid in texts and query != texts[qid]:
            raise ValueError('A query ID has conflicting query text')
        texts[qid] = query
        pid = str(row['product_id'])
        title = products.get(pid)
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f'Missing title for selected product {pid}')
        candidate = {'product_id': pid, 'product_title': title}
        if 'esci_label' in row:
            label = row['esci_label']
            if label not in GAINS:
                raise ValueError('Unknown ESCI label')
            candidate.update(esci_label=label, gain=GAINS[label])
        groups[qid].append(candidate)
    result = []
    for qid in query_ids:
        candidates = sorted(groups[qid], key=lambda x: x['product_id'])
        if not 2 <= len(candidates) <= 40 or len({x['product_id'] for x in candidates}) != len(candidates):
            raise ValueError('Selected query must retain its complete distinct 2–40 product pool')
        result.append({'query_id': qid, 'query': texts[qid], 'locale': 'us',
                       'candidates': candidates, 'official_candidate_count': len(candidates)})
    return result


def ranked_indices(scores, product_ids) -> list[int]:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(product_ids),) or not np.isfinite(values).all() or len(set(product_ids)) != len(product_ids):
        raise ValueError('Scores must be finite and align with distinct candidate IDs')
    return sorted(range(len(values)), key=lambda i: (-float(values[i]), product_ids[i]))


def query_metrics(labels, scores, product_ids, k=10) -> dict:
    if len(labels) != len(product_ids) or not labels or any(label not in GAINS for label in labels):
        raise ValueError('Labels must match the candidate list')
    order = ranked_indices(scores, product_ids)
    def ndcg(gains, cutoff):
        actual = [gains[labels[i]] for i in order[:cutoff]]
        ideal = sorted((gains[x] for x in labels), reverse=True)[:cutoff]
        dcg = lambda values: sum(g / math.log2(i + 2) for i, g in enumerate(values))
        denominator = dcg(ideal)
        return dcg(actual) / denominator if denominator else 0.0
    exact = sum(x == 'E' for x in labels)
    first = next((rank for rank, i in enumerate(order[:k], 1) if labels[i] == 'E'), None)
    return {'ndcg@10': ndcg(GAINS, k), 'ndcg@all': ndcg(GAINS, len(labels)),
            'mrr_exact@10': 1.0 / first if first else 0.0,
            'recall_exact@10': sum(labels[i] == 'E' for i in order[:k]) / exact if exact else 0.0,
            'ndcg@10_qrels_script_variant': ndcg(QRELS_SCRIPT_GAINS, k),
            'ndcg@all_qrels_script_variant': ndcg(QRELS_SCRIPT_GAINS, len(labels))}


def tfidf_scores(train_queries, test_queries) -> list[np.ndarray]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    train_titles = {c['product_id']: c['product_title'] for q in train_queries for c in q['candidates']}
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True,
                                 max_features=100000, norm='l2', dtype=np.float64)
    vectorizer.fit([train_titles[pid] for pid in sorted(train_titles)])
    result = []
    for q in test_queries:
        query = vectorizer.transform([q['query']])
        titles = vectorizer.transform([c['product_title'] for c in q['candidates']])
        result.append(np.asarray((titles @ query.T).toarray()).ravel())
    return result


def evaluate_scores(queries, scores_by_model: dict) -> tuple[list[dict], list[dict]]:
    if set(scores_by_model) != set(MODELS):
        raise ValueError('All four declared models are required, without fallback')
    if any(len(scores_by_model[m]) != len(queries) for m in MODELS):
        raise ValueError('Every declared model must score every selected query')
    output = []
    aggregate = {m: defaultdict(list) for m in MODELS}
    for index, original in enumerate(queries):
        q = json.loads(json.dumps(original))
        q['split'] = 'test'
        q['metrics'] = {}
        ids = [c['product_id'] for c in q['candidates']]
        labels = [c['esci_label'] for c in q['candidates']]
        for m in MODELS:
            scores = scores_by_model[m][index]
            metrics = query_metrics(labels, scores, ids)
            q['metrics'][m] = metrics
            for key, value in metrics.items():
                aggregate[m][key].append(value)
            order = ranked_indices(scores, ids)
            ranks = {i: rank for rank, i in enumerate(order, 1)}
            for i, c in enumerate(q['candidates']):
                c.setdefault('scores', {})[m] = float(scores[i])
                c.setdefault('ranks', {})[m] = ranks[i]
        output.append(q)
    models = []
    frozen = np.asarray(aggregate['qwen_frozen']['ndcg@10'])
    for m in MODELS:
        delta = np.asarray(aggregate[m]['ndcg@10']) - frozen
        models.append({'model': m, 'label': LABELS[m],
                       'metrics': {k: float(np.mean(v)) for k, v in aggregate[m].items()},
                       'paired_ndcg10_vs_frozen': {'mean_delta': float(delta.mean()),
                           'wins': int((delta > 1e-12).sum()), 'losses': int((delta < -1e-12).sum()),
                           'ties': int((abs(delta) <= 1e-12).sum())}})
    return output, models


def validate_model_exports(root) -> dict:
    """Inspect only the declared transfer adapters and their existing metadata."""
    result = {}
    for mode in MODELS[1:]:
        unit = Path(root) / 'models/qwen/clothing' / mode / 'seed42'
        metadata = json.loads((unit / 'head.json').read_text())
        if (metadata.get('model'), metadata.get('seed'), metadata.get('model_id'), metadata.get('revision')) != (mode, 42, MODEL_ID, REVISION):
            raise ValueError('Transfer model identity does not match declared Clothing seed42 export')
        entry = {'model_id': MODEL_ID, 'revision': REVISION, 'domain': 'clothing', 'seed': 42,
                 'selected_epoch': metadata['selected_epoch'], 'head_metadata_sha256': file_sha(unit / 'head.json'),
                 'adapter_sha256': metadata.get('adapter_sha256'), 'head_used': False}
        if mode != 'qwen_frozen':
            if tree_sha(unit / 'adapter') != metadata.get('adapter_sha256'):
                raise ValueError('Selected transfer adapter SHA256 mismatch')
            config = json.loads((unit / 'adapter/adapter_config.json').read_text())
            if config.get('r') != 8 or sorted(config.get('target_modules', [])) != ['q_proj', 'v_proj']:
                raise ValueError('Unexpected transfer adapter architecture')
        elif metadata.get('adapter_sha256') is not None:
            raise ValueError('Frozen encoder cannot use an adapter')
        result[mode] = entry
    return result


class TextEncoder:
    """One explicit local base with two verified adapters, no downloads or fallback."""
    def __init__(self, root, manifest_path, device='cpu', batch_size=16, progress=print):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.exports = validate_model_exports(root)
        self.root, self.device, self.batch_size, self.progress = Path(root), device, batch_size, progress
        if batch_size < 1 or device not in ('cpu', 'mps'):
            raise ValueError('Positive batch size and cpu/mps device required')
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable; choose CPU explicitly')
        torch.set_num_threads(4)
        manifest = json.loads(Path(manifest_path).read_text())
        details = manifest['status']['qwen']
        if (details['model_id'], details['revision']) != (MODEL_ID, REVISION):
            raise ValueError('Prepared base model identity mismatch')
        model_path = Path(details['path'])
        self.base_hashes = manifest['model_files']['qwen']
        for name, record in self.base_hashes.items():
            f = model_path / name
            if f.stat().st_size != record['bytes'] or file_sha(f) != record['sha256']:
                raise ValueError('Prepared base model content mismatch: ' + name)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), padding_side='left', local_files_only=True,
                                                      token=False, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(str(model_path), torch_dtype=torch.float32,
                       attn_implementation='sdpa', local_files_only=True, token=False, trust_remote_code=False).to(device).eval()
        self.model.config.use_cache = False
        self.model.requires_grad_(False)
        if self.model.config.hidden_size != 1024:
            raise ValueError('Expected pinned 1024D Qwen encoder')
        self.peft = False

    def encode(self, mode, texts, *, query=False) -> np.ndarray:
        import torch
        from contextlib import nullcontext
        from peft import PeftModel
        from .qwen_text import embed_texts
        if mode not in MODELS[1:] or not texts:
            raise ValueError('Declared dense mode and nonempty texts required')
        if mode != 'qwen_frozen':
            adapter = str(self.root / 'models/qwen/clothing' / mode / 'seed42/adapter')
            if not self.peft:
                self.model = PeftModel.from_pretrained(self.model, adapter, adapter_name=mode,
                                                      is_trainable=False, local_files_only=True)
                self.peft = True
            elif mode not in self.model.peft_config:
                self.model.load_adapter(adapter, adapter_name=mode, is_trainable=False, local_files_only=True)
            self.model.set_adapter(mode)
            self.model.base_model.enable_adapter_layers()
        values = [f'Instruct: {INSTRUCTION}\nQuery:{text}' if query else text for text in texts]
        # Fixed length order reduces padding while restoring exactly the input order.
        order = sorted(range(len(values)), key=lambda i: (len(values[i]), i))
        output = np.empty((len(values), 1024), dtype=np.float32)
        context = self.model.disable_adapter() if mode == 'qwen_frozen' and self.peft else nullcontext()
        try:
            self.model.eval().requires_grad_(False)
            with context, torch.inference_mode():
                for start in range(0, len(order), self.batch_size):
                    ids = order[start:start + self.batch_size]
                    vectors = embed_texts(self.model, self.tokenizer, [values[i] for i in ids], self.device, 256).cpu().numpy()
                    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
                        raise ValueError('Nonfinite or non-normalized actual encoder output')
                    output[ids] = vectors
                    if start % (self.batch_size * 20) == 0 or start + self.batch_size >= len(order):
                        self.progress(f'{mode} {"queries" if query else "titles"} {min(start + self.batch_size, len(order))}/{len(order)}', flush=True)
        finally:
            if self.peft:
                self.model.base_model.disable_adapter_layers()
        return output


_RESULT_FILES = ('summary.json', 'queries.jsonl', 'test_inputs.jsonl',
                 'protocol.json', 'selected_query_ids.json', 'source_provenance.json', 'tfidf_training_inputs.jsonl')


def _same_numbers(actual, expected, context):
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError(f'ESCI {context} fields do not match recomputed evidence')
    for key, value in expected.items():
        observed = actual[key]
        if isinstance(value, dict):
            _same_numbers(observed, value, context + '/' + key)
        elif (isinstance(observed, bool) or not isinstance(observed, (int, float))
              or not math.isfinite(observed) or abs(observed-value) > 1e-12):
            raise ValueError(f'ESCI {context}/{key} differs from recomputed evidence')


@lru_cache(maxsize=2)
def _load_search_results_cached(directory: str, signature: tuple) -> dict:
    del signature
    directory = Path(directory)
    read_json = lambda name: json.loads((directory/name).read_text(encoding='utf-8'))
    def read_lines(name):
        with (directory/name).open(encoding='utf-8') as stream:
            return [json.loads(line) for line in stream if line.strip()]
    summary, declaration = read_json('summary.json'), read_json('protocol.json')
    protocol = declared_protocol()
    if summary.get('complete') is not True or summary.get('diagnostic') is not False:
        raise ValueError('ESCI benchmark is incomplete or diagnostic')
    if (summary.get('schema_version'), summary.get('benchmark_id')) != (1, protocol['benchmark_id']):
        raise ValueError('Unsupported ESCI result schema or benchmark identity')
    digest = canonical_hash(protocol)
    if (summary.get('protocol') != protocol or declaration.get('protocol') != protocol
            or summary.get('protocol_sha256') != digest or declaration.get('protocol_sha256') != digest):
        raise ValueError('ESCI result and predeclared protocol disagree')
    exports = summary.get('model_exports',{})
    if exports != declaration.get('model_exports') or set(exports) != set(MODELS[1:]):
        raise ValueError('ESCI model export provenance changed after predeclaration')
    for mode,metadata in exports.items():
        if (metadata.get('model_id'),metadata.get('revision'),metadata.get('domain'),metadata.get('seed'),metadata.get('head_used')) != (MODEL_ID,REVISION,'clothing',42,False):
            raise ValueError('ESCI transfer export identity mismatch')
        if metadata.get('selected_epoch') not in (1,2) or isinstance(metadata.get('selected_epoch'),bool):
            raise ValueError('ESCI transfer export selected epoch mismatch')
        adapter = metadata.get('adapter_sha256')
        if (mode=='qwen_frozen' and adapter is not None) or (mode!='qwen_frozen' and not isinstance(adapter,str)):
            raise ValueError('ESCI transfer adapter identity mismatch')
        if mode!='qwen_frozen' and not re.fullmatch('[0-9a-f]{64}',adapter):
            raise ValueError('ESCI transfer adapter checksum declaration invalid')
    source = read_json('source_provenance.json')
    if source != summary.get('source_provenance') or source.get('commit') != SOURCE_COMMIT:
        raise ValueError('ESCI source provenance does not match the summary')
    for key in ('examples','products'):
        if any(source.get(key,{}).get(field) != value for field,value in SOURCE_FILES[key].items()):
            raise ValueError('ESCI upstream source identity mismatch')
    if source['examples'].get('full_file_sha256_verified') is not True:
        raise ValueError('ESCI examples lack their recorded complete-file verification')
    if file_sha(directory/'queries.jsonl') != summary.get('queries_sha256'):
        raise ValueError('ESCI query result SHA256 mismatch')
    if file_sha(directory/'test_inputs.jsonl') != source.get('test_inputs_sha256'):
        raise ValueError('ESCI original test input SHA256 mismatch')
    if file_sha(directory/'tfidf_training_inputs.jsonl') != source.get('tfidf_training_inputs_sha256'):
        raise ValueError('ESCI TF-IDF training input SHA256 mismatch')
    rows, inputs = read_lines('queries.jsonl'), read_lines('test_inputs.jsonl')
    selected = read_json('selected_query_ids.json')
    ids = [str(q.get('query_id')) for q in inputs]
    training = read_lines('tfidf_training_inputs.jsonl')
    train_ids = [str(q.get('query_id')) for q in training]
    if (summary.get('query_count') != 500 or len(rows) != 500 or len(inputs) != 500
            or len(set(ids)) != 500 or ids != selected.get('test')
            or canonical_hash(ids) != summary.get('query_ids_sha256')
            or source.get('query_ids_sha256') != summary.get('query_ids_sha256')
            or selected.get('protocol_sha256') != digest or selected.get('selection_uses_labels') is not False):
        raise ValueError('ESCI selected 500 query identities disagree')
    if len(training)!=500 or len(set(train_ids))!=500 or train_ids!=selected.get('train') or set(train_ids)&set(ids):
        raise ValueError('ESCI TF-IDF training query identities overlap or disagree')
    model_rows = summary.get('models', [])
    if [m.get('model') for m in model_rows] != list(MODELS):
        raise ValueError('ESCI summary must contain exactly the four declared model IDs')
    scores = {m:[] for m in MODELS}
    candidate_count = 0
    products = set()
    for original, result in zip(inputs, rows):
        if result.get('split') != 'test' or result.get('locale') != 'us' or original.get('locale') != 'us':
            raise ValueError('ESCI result is not from the declared English test scope')
        for key in ('query_id','query','official_candidate_count'):
            if result.get(key) != original.get(key):
                raise ValueError('ESCI query or candidate pool changed after source preparation')
        source_candidates = original.get('candidates', [])
        candidates = result.get('candidates', [])
        n = len(source_candidates)
        candidate_ids = [str(c.get('product_id')) for c in source_candidates]
        if (not 2<=n<=40 or n != original.get('official_candidate_count') or len(candidates) != n
                or len(set(candidate_ids)) != n or candidate_ids != sorted(candidate_ids)):
            raise ValueError('ESCI official candidate pool is incomplete or duplicated')
        candidate_count += n
        products.update(candidate_ids)
        for source_candidate, candidate in zip(source_candidates, candidates):
            if any(candidate.get(k) != source_candidate.get(k) for k in ('product_id','product_title','esci_label','gain')):
                raise ValueError('ESCI candidate labels or product text changed from the original inputs')
            if source_candidate.get('esci_label') not in GAINS or source_candidate.get('gain') != GAINS[source_candidate['esci_label']]:
                raise ValueError('ESCI candidate gain does not match the primary label mapping')
            if set(candidate.get('scores',{})) != set(MODELS) or set(candidate.get('ranks',{})) != set(MODELS):
                raise ValueError('ESCI candidate lacks a score or rank from a declared model')
            if any(isinstance(s,bool) or not isinstance(s,(int,float)) or not math.isfinite(s)
                   for s in candidate['scores'].values()):
                raise ValueError('ESCI scores must be actual finite numeric values')
        for mode in MODELS:
            values = [c['scores'][mode] for c in candidates]
            scores[mode].append(values)
            expected_ranks = {i:rank for rank,i in enumerate(ranked_indices(values,candidate_ids),1)}
            for i,c in enumerate(candidates):
                rank = c['ranks'][mode]
                if isinstance(rank,bool) or not isinstance(rank,int) or rank != expected_ranks[i]:
                    raise ValueError('ESCI candidate ranks do not form the actual score-sorted full permutation')
    recomputed_rows, recomputed_models = evaluate_scores(inputs,scores)
    for actual, expected in zip(rows,recomputed_rows):
        if set(actual.get('metrics',{})) != set(MODELS):
            raise ValueError('ESCI query metrics omit a declared model')
        for mode in MODELS:
            _same_numbers(actual['metrics'][mode],expected['metrics'][mode],'query metrics/'+mode)
    for actual,expected in zip(model_rows,recomputed_models):
        if actual.get('label') != expected['label']:
            raise ValueError('ESCI model labels differ from the declared identities')
        _same_numbers(actual.get('metrics'),expected['metrics'],'summary metrics/'+expected['model'])
        _same_numbers(actual.get('paired_ndcg10_vs_frozen'),expected['paired_ndcg10_vs_frozen'],'paired counts/'+expected['model'])
    if summary.get('candidate_count') != candidate_count or summary.get('unique_test_products') != len(products):
        raise ValueError('ESCI summary candidate counts do not match complete query pools')
    return {'summary':summary,'queries':rows}


def load_search_results(root) -> dict:
    """Validate portable real results without importing torch or loading models.

    Content checks cache only while all seven evidence files keep their full stat
    signature. Return a defensive copy so a UI session cannot mutate that cache.
    """
    directory = Path(root).expanduser().resolve()/'results/esci'
    try:
        signature = tuple((name,(directory/name).stat().st_size,(directory/name).stat().st_mtime_ns,
                           (directory/name).stat().st_ctime_ns) for name in _RESULT_FILES)
    except OSError as exc:
        raise FileNotFoundError('ESCI results are missing; run the declared ESCI benchmark first') from exc
    value = copy.deepcopy(_load_search_results_cached(str(directory),signature))
    value['signature'] = canonical_hash(signature)
    return value
