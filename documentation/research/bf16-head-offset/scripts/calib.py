"""Which bf16 emulation reproduces the real MPSGraph engine? Compare several
rounding sets against the bot's recorded decisions (W/D/L read back as bf16
probabilities; top-5 policy probabilities = fp32 softmax over the fp32-widened
bf16 legal logits) for all Ejp0 step-681000 moves."""
import sys, os, pickle, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from fwd4 import load, bf16, forward_batched, softmax
from archnorm import norm_arch
MP = os.path.expanduser('~/Library/Application Support/DrewsChessMachine/Models/20260702-Qeu8-resume3-replay-step681000.safetensors')
md, T = load(MP); arch = norm_arch(md['architecture'])
assert md['model_id'] == '20260727-1-Ejp0' and md['training_step'] == '681000'
P = pickle.load(open(os.path.join(HERE, 'posset_ejp0.pkl'), 'rb')); L = [p for p in P if p['ours']]
X = np.stack([p['x'] for p in L]).astype(np.float64)
# all point names
from fwd4 import forward
names = list(forward(T, arch, X[:2], capture=True)['acts'].keys())
ALLN = set(names)
variants = {
  'f64 internals, heads fused-rounded': (frozenset(), False),
  'f64 internals, heads mm+bias rounded': (frozenset({'p.conv', 'v.fc2mm'}), True),
  'input only + heads': (frozenset({'input', 'p.conv', 'v.fc2mm'}), True),
  'per-op ALL (incl. norm .sub)': ('ALL', True),
  'per-op, no norm .sub': (frozenset(n for n in ALLN if not n.endswith('.sub')), True),
  'kernel outputs only (conv/mm/norm/add, no SE-internal/alpha/.sub/ln stats)':
      (frozenset(n for n in ALLN if not (n.endswith('.sub') or n.endswith('.mean') or n.endswith('.var') or n.endswith('alpha')
       or any(t in n for t in ('se.sig', 'se.scaled', 'se.fc1mm', 'se.fc2mm', 'v.fc1mm')))), True),
}
res = {}
for name, (R, mm) in variants.items():
    t = time.time()
    o = forward_batched(T, arch, X, R)
    pl = bf16(o['pl_raw']); vl = bf16(o['vl_raw'])
    vp = bf16(softmax(vl)).astype(np.float32)
    wdl_exact = 0; wdl_err = []; pol_exact = 0; pol_err = []; top1 = 0; tie_rep = 0; tie_obs = 0
    for k, p in enumerate(L):
        ob = p['obs']; rec = np.array([ob['win'], ob['draw'], ob['loss']], dtype=np.float32)
        wdl_exact += np.array_equal(vp[k], rec); wdl_err.append(float(np.abs(vp[k]-rec).max()))
        lg = pl[k][p['legal']].astype(np.float32); e = np.exp(lg-lg.max()); pr = e/e.sum()
        emu = {p['legal_uci'][j]: float(pr[j]) for j in range(len(pr))}
        tm = ob['topMoves']; err = max(abs(emu[t['uci']]-t['probability']) for t in tm)
        pol_err.append(err); pol_exact += err < 1e-6
        top1 += emu[tm[0]['uci']] >= pr.max()
        if len(tm) > 1 and tm[0]['probability'] == tm[1]['probability']:
            tie_obs += 1; tie_rep += emu[tm[0]['uci']] == emu[tm[1]['uci']]
    res[name] = dict(wdl_exact=int(wdl_exact), wdl_err_med=float(np.median(wdl_err)), wdl_err_p90=float(np.percentile(wdl_err, 90)),
                     pol_exact=int(pol_exact), pol_err_med=float(np.median(pol_err)), pol_err_p90=float(np.percentile(pol_err, 90)),
                     top1=int(top1), ties_obs=int(tie_obs), ties_rep=int(tie_rep), n=len(L), secs=time.time()-t)
    print(f"{name:75s} WDL exact {wdl_exact}/{len(L)} (err med {np.median(wdl_err):.1e} p90 {np.percentile(wdl_err,90):.1e}) | "
          f"top5 exact {pol_exact}/{len(L)} (err med {np.median(pol_err):.1e} p90 {np.percentile(pol_err,90):.1e}) | top1 {top1} | recorded top-2 ties reproduced {tie_rep}/{tie_obs}", flush=True)
json.dump(res, open(os.path.join(HERE, 'calib.json'), 'w'), indent=1)
