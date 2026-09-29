"""Single-point scan: fused head-output rounding + bf16 rounding at ONE internal
point; count exact matches with the bot's recorded outputs."""
import sys, os, pickle, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from fwd4 import load, bf16, forward_batched, forward, softmax
from archnorm import norm_arch
MP = os.path.expanduser('~/Library/Application Support/DrewsChessMachine/Models/20260702-Qeu8-resume3-replay-step681000.safetensors')
md, T = load(MP); arch = norm_arch(md['architecture'])
P = pickle.load(open(os.path.join(HERE, 'posset_ejp0.pkl'), 'rb')); L = [p for p in P if p['ours']]
X = np.stack([p['x'] for p in L]).astype(np.float64)
names = list(forward(T, arch, X[:2], capture=True)['acts'].keys())
REC = np.array([[p['obs']['win'], p['obs']['draw'], p['obs']['loss']] for p in L], dtype=np.float32)
def score(R):
    o = forward_batched(T, arch, X, R)
    pl = bf16(o['pl_raw']); vp = bf16(softmax(bf16(o['vl_raw']))).astype(np.float32)
    w = int(np.sum(np.all(vp == REC, 1))); pe = 0
    for k, p in enumerate(L):
        lg = pl[k][p['legal']].astype(np.float32); e = np.exp(lg-lg.max()); pr = e/e.sum()
        emu = {p['legal_uci'][j]: float(pr[j]) for j in range(len(pr))}
        pe += max(abs(emu[t['uci']]-t['probability']) for t in p['obs']['topMoves']) < 1e-6
    return w, pe
out = {}
base = score(frozenset()); print('base', base, flush=True); out['base'] = base
for n in names:
    out[n] = score(frozenset({n})); print(f'{n:16s} wdl {out[n][0]:4d} top5 {out[n][1]:4d}', flush=True)
json.dump(out, open(os.path.join(HERE, 'calib_scan.json'), 'w'), indent=1)
