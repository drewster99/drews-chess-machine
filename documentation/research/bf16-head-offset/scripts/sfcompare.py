"""Does bf16 rounding of the value head move the engine's W/D/L away from a
reference evaluation? Stockfish depth-14 expected score (side to move, best
move, sf WDL model) from the earlier bug check, vs the fp64 numpy value, the
bot's RECORDED (real bf16 engine) value, and the recentered-bf16 value."""
import json, os, sys, pickle, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from fwd4 import load, bf16, forward_batched, softmax
from archnorm import norm_arch_md
SF = json.load(open(os.path.join(os.path.dirname(HERE), 'results', 'ejp0-681k', 'ejp0_sf_full.json')))
sf = {(p['game'], p['ply']): p['exp_best'] for p in SF if p.get('exp_best') is not None}
P = pickle.load(open(os.path.join(HERE, 'posset_ejp0.pkl'), 'rb'))
L = [p for p in P if p['ours'] and (p['game'], p['ply']) in sf]
md, T = load(os.path.expanduser('~/Library/Application Support/DrewsChessMachine/Models/20260702-Qeu8-resume3-replay-step681000.safetensors'))
arch = norm_arch_md(md); X = np.stack([p['x'] for p in L]).astype(np.float64)
o = forward_batched(T, arch, X, md=md); vl = o['vl_raw']; W = T['value.wdl_fc2.weight']; b = T['value.wdl_fc2.bias']
p64 = softmax(vl); prc = bf16(softmax(bf16(o['f1'] @ bf16(W-W.mean(0)).T + bf16(b-b.mean()))))
rec = np.array([[p['obs']['win'], p['obs']['draw'], p['obs']['loss']] for p in L])
y = np.array([sf[(p['game'], p['ply'])] for p in L])
E = lambda q: q[:, 0]+0.5*q[:, 1]
out = {}
for k, q in (('fp64', p64), ('recorded (real bf16 engine)', rec), ('recentered fc2, bf16', prc)):
    e = E(q); out[k] = dict(mae_vs_sf=float(np.abs(e-y).mean()), corr_vs_sf=float(np.corrcoef(e, y)[0, 1]), n=len(L))
    print(k, out[k])
json.dump(out, open(os.path.join(HERE, 'sfcompare.json'), 'w'), indent=1)
