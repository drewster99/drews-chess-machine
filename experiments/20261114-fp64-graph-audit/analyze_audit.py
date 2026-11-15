#!/usr/bin/env python3
"""Summarize --graph-audit JSONL output: per case, the worst forward and gradient checks and every check above
its precision's threshold. The judged error is scale_rel_err = ||mps - fp64|| / ||magnitude of the summed terms||,
so a gradient that cancels to ~0 is judged on the size of what was added up.

Thresholds: output and float operands all fp32 -> 1e-4; any fp16 -> 1e-2; any bf16 -> 3e-2.
The fp32 conv weight-gradient defect the audit must find measured 0.09-0.75."""
import json, sys, collections

THRESH = {'float32': 1e-4, 'float16': 1e-2, 'bfloat16': 3e-2}
ORDER = ['float32', 'float16', 'bfloat16']


def precision(rec):
    types = [rec.get('dtype')] + list(rec.get('operand_dtypes', []))
    return max((t for t in types if t in THRESH), key=ORDER.index, default='float32')


def main(paths, top=5):
    rows = [json.loads(l) for p in paths for l in open(p) if l.strip()]
    cases = collections.OrderedDict()
    for r in rows:
        cases.setdefault(r.get('case'), []).append(r)
    total = 0
    for case, R in cases.items():
        errs = [r for r in R if r.get('event') == 'case_error']
        ops = [r for r in R if r.get('event') == 'op']
        grads = [r for r in R if r.get('event') == 'gradient']
        print(f"== {case}: {len(ops)} ops, {len(grads)} gradients" + (f"  CASE ERROR: {errs[0]['error']}" if errs else ''))
        flags = []
        for kind, lst, key in (('fwd', ops, 'forward'), ('grad', grads, 'gradient')):
            checked = [r for r in lst if key in r]
            for r in checked:
                e = r[key].get('scale_rel_err', 0)
                if r[key].get('non_finite', 0) or e > THRESH[precision(r)]:
                    flags.append((kind, r, key, e))
            for r in sorted(checked, key=lambda r: -r[key].get('scale_rel_err', 0))[:top]:
                c = r[key]
                print(f"   {kind:4} {r.get('kind', r.get('producer_kind')):19} {precision(r):8} "
                      f"{(r.get('label') or r.get('tensor_name') or '')[:44]:44} scale={c.get('scale_rel_err', 0):.2e} "
                      f"rel={c.get('rel_err') or 0:.2e}")
        for r in ops:
            if 'forward_error' in r or 'backward_error' in r:
                print('   REF-ERROR', r['kind'], r.get('label'), r.get('forward_error') or r.get('backward_error'))
        skipped = collections.Counter(r['skipped'] for r in grads if 'skipped' in r)
        if skipped:
            print('   skipped:', dict(skipped))
        for kind, r, key, e in flags:
            print(f"   FLAG {kind} {r.get('kind', r.get('producer_kind'))} {precision(r)} "
                  f"{r.get('label') or r.get('tensor_name')} scale={e:.3e} non_finite={r[key].get('non_finite', 0)}")
        total += len(flags)
    print(f"total flags: {total}")


if __name__ == '__main__':
    main(sys.argv[1:])
