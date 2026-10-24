#!/usr/bin/env python3
"""GPU regression of the production embedding path using the frozen diagnostic."""
from pathlib import Path
import json
import os
import sys

import torch
import diagnose_qwen_numerics as diagnostic

sys.path.insert(0, str(diagnostic.ROOT / 'src'))
from shoplens.models import digest
from shoplens.qwen_text import embed_texts, last_token_pool


def production_embeddings(encoder, tokenizer, texts, device, max_length):
    captured = {}
    def observe(_module, _args, kwargs, output):
        with torch.no_grad():
            pooled = last_token_pool(output.last_hidden_state, kwargs['attention_mask']).float()
            norms = pooled.double().norm(dim=-1)
            captured.update(diagnostic.tensor_stats(pooled))
            captured.update(min_row_norm=diagnostic.scalar(norms.min()), max_row_norm=diagnostic.scalar(norms.max()),
                            token_length_min=int(kwargs['attention_mask'].sum(1).min()),
                            token_length_max=int(kwargs['attention_mask'].sum(1).max()))
    handle = encoder.register_forward_hook(observe, with_kwargs=True)
    try:
        values = embed_texts(encoder, tokenizer, texts, device, max_length)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError('Production forward hook did not observe pooled hidden states')
    return values, captured


def main():
    arguments = sys.argv[1:]
    if '--backend' in arguments and arguments[arguments.index('--backend')+1] != 'default':
        raise ValueError('Production regression must use default outer context so the source fix is exercised')
    diagnostic.live_embeddings = production_embeddings
    code = diagnostic.main()
    root = Path(arguments[arguments.index('--root')+1]).resolve() if '--root' in arguments else diagnostic.ROOT
    replay = '--replay-state' in arguments
    tag = ('replay' if replay else 'prefix') + '_default'
    directory = root / 'experiments/qwen/diagnostics' / os.environ['PBS_JOBID']
    source = directory / f'{tag}_summary.json'
    summary = json.loads(source.read_text())
    expected = 'diagnostic_replay_finite' if replay else 'diagnostic_prefix_finite'
    passed = code == 0 and summary['status'] == expected
    result = {'diagnostic': True, 'formal_results': False, 'status': 'passed' if passed else 'failed',
              'production_function': 'shoplens.qwen_text.embed_texts', 'outer_sdpa_backend': 'default',
              'summary': str(source.relative_to(root)), 'summary_sha256': digest(source),
              'wrapper_sha256': digest(Path(__file__)),
              'production_source_sha256': digest(root/'src/shoplens/qwen_text.py'),
              'replay_source_sha256': summary.get('replay_source_sha256'),
              'successful_optimizer_steps': summary['successful_optimizer_steps']}
    (directory/f'production_verification_{tag}.json').write_text(json.dumps(result, indent=2)+'\n')
    print('QWEN_PRODUCTION_VERIFICATION', json.dumps(result), flush=True)
    if not passed:
        raise RuntimeError('Production numerical regression failed; formal training must not start')


if __name__ == '__main__':
    main()
