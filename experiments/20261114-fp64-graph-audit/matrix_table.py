#!/usr/bin/env python3
"""One Markdown row per audit case: ops and gradients checked, worst forward / gradient scale_rel_err per precision
(fp32-only checks vs checks with an fp16/bf16 operand), flag count. Usage: matrix_table.py matrix/*.jsonl"""
import json, sys, collections
from analyze_audit import THRESH, precision

rows = [json.loads(l) for p in sys.argv[1:] for l in open(p) if l.strip()]
cases = collections.OrderedDict()
for r in rows:
    cases.setdefault(r.get('case'), []).append(r)
print('| case | dtype | ops | gradients | worst fwd fp32 | worst fwd 16-bit | worst grad fp32 | worst grad 16-bit | flags |')
print('|---|---|---:|---:|---:|---:|---:|---:|---:|')
fmt = lambda v: '' if v is None else f'{v:.1e}'
for case, R in cases.items():
    ops = [r for r in R if r.get('event') == 'op' and 'forward' in r]
    grads = [r for r in R if r.get('event') == 'gradient' and 'gradient' in r]
    def worst(lst, key, p32):
        v = [r[key]['scale_rel_err'] for r in lst if (precision(r) == 'float32') == p32]
        return max(v) if v else None
    flags = sum(1 for lst, key in ((ops, 'forward'), (grads, 'gradient')) for r in lst
                if r[key].get('non_finite', 0) or r[key]['scale_rel_err'] > THRESH[precision(r)])
    dtype = R[0].get('compute_data_type')
    n_ops = sum(1 for r in R if r.get('event') == 'op')
    n_g = sum(1 for r in R if r.get('event') == 'gradient')
    print(f'| {case} | {dtype} | {n_ops} | {n_g} | {fmt(worst(ops,"forward",True))} | {fmt(worst(ops,"forward",False))} '
          f'| {fmt(worst(grads,"gradient",True))} | {fmt(worst(grads,"gradient",False))} | {flags} |')
