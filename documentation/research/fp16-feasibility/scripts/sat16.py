"""ReZero tanh and SE-gate saturation under fp16 vs bf16: a stored output of
exactly 1.0 (or 0.0) makes a derivative computed from the output
(1 - t^2, s(1 - s)) exactly zero."""
import sys, os, json, pickle, math, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from fwd16 import *
A = os.path.expanduser('~/Library/Application Support/DrewsChessMachine/')
P = pickle.load(open(POSSET, 'rb'))
X = np.stack([p['x'] for p in P]).astype(np.float64)
out = {}
for rel in sys.argv[1:]:
    md, T = load(A + rel); arch = norm_arch_md(md); g = arch['block_groups'][0]
    r = dict(model_id=md['model_id'], training_step=md.get('training_step'))
    if g['use_rezero']:
        C = g['rezero_alpha_cap']; rz = {}
        for i in range(g['count']):
            a = float(T[f'blocks.{i}.rezero_alpha'].reshape(-1)[0]); t = math.tanh(a / C)
            rz[f'b{i}'] = dict(alpha_raw=a, tanh=t, deriv_fp64=1 - t * t, tanh_bf16=float(bf16(t)), deriv_from_bf16=float(1 - bf16(t) ** 2),
                              tanh_fp16=float(f16(t)), deriv_from_fp16=float(1 - f16(t) ** 2))
        r['rezero'] = rz
    sat = {}
    for s in range(0, len(X), 128):
        a = forward(T, arch, X[s:s + 128], capture=True, md=md)['acts']
        for k, v in a.items():
            if not k.endswith('se.sig'): continue
            d = sat.setdefault(k, [0, 0, 0, 0, 0])
            d[0] += int((bf16(v) == 1).sum()); d[1] += int((f16(v) == 1).sum()); d[2] += int((bf16(v) == 0).sum()); d[3] += int((f16_ftz(v) == 0).sum()); d[4] += v.size
    r['se_gate'] = {k: dict(frac_bf16_eq1=v[0] / v[4], frac_fp16_eq1=v[1] / v[4], frac_bf16_eq0=v[2] / v[4], frac_fp16ftz_eq0=v[3] / v[4]) for k, v in sat.items()}
    out[md['model_id'] + '@' + str(md.get('training_step'))] = r
    print(json.dumps(r, indent=0)[:3000], flush=True)
json.dump(out, open("sat16.json", 'w'), indent=1)
