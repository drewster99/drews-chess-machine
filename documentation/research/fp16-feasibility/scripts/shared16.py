"""Per-position gradient along the softmax-invariant (shared-shift) direction,
sum_k dL/dz_k, for the policy loss (signed-advantage CE + complement CE) and the
value CE, with the CE targets built in fp64 / bf16 / fp16 exactly as
ChessTrainer.buildTrainingOps orders the ops. In exact math this is
a+ (1 - sum y) + a- (1 - sum yc) for the policy and (1 - sum yv) for the value;
nonzero values drive the shared offset. Batch = the 900 corpus positions of
posset.pkl resampled to 4096 (seed 0); advantage from the model's fp64 value."""
import sys, os, json, pickle, math, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from fwd16 import *
A = os.path.expanduser('~/Library/Application Support/DrewsChessMachine/')
md, T = load(A + sys.argv[1]); arch = norm_arch_md(md); T = {k: v for k, v in T.items() if not k.startswith('opt.')}
P = [p for p in pickle.load(open(POSSET, 'rb')) if p['src'] == 'corpus']
B = 4096; idx = np.random.default_rng(0).integers(0, len(P), B); ui, cnt = np.unique(idx, return_counts=True)
o = forward_batched(T, arch, np.stack([P[i]['x'] for i in ui]).astype(np.float64), md=md)
pv = softmax(o['vl_raw']); v = pv[:, 0] - pv[:, 2]
lab = np.array([P[i]['label'] for i in ui]); z = np.array([{0: 1., 1: 0., 2: -1.}[l] for l in lab])
Adv = z - v; rms = math.sqrt(max(np.sum(cnt * Adv ** 2) / B, 0.04) + 1e-6); an = Adv / rms; ap = np.maximum(an, 0); am = np.maximum(-an, 0)
nleg = np.array([len(P[i]['legal']) for i in ui])
res = {}
for nm, q in (('fp64', ident), ('bf16', bf16), ('fp16', f16)):
    e = q(np.float64(0.1)); ome = q(1 - e); sp = []; sc = []
    for j, i in enumerate(ui):
        n = nleg[j]; u = q(1.0 / n); uc = q(u * e)
        ys = np.zeros(n); ys[:] = uc; t = list(P[i]['legal']).index(P[i]['target']); ys[t] = q(ome + uc)
        oth = q(1.0 / max(n - 1, 1)) if n > 1 else 0.0
        yc = np.full(n, q(q(oth * ome) + uc)); yc[t] = uc
        sp.append(1 - ys.sum()); sc.append(1 - yc.sum())
    sp = np.array(sp); sc = np.array(sc)
    per = ap * sp + am * sc
    ev = q(np.float64(0.013)); yv = q(q(ome * 0 + q(1 - ev)) + q(q(1 / 3) * ev)) + 2 * q(q(1 / 3) * ev)
    res[nm] = dict(policy_shared_grad_batch_mean=float(np.sum(cnt * per) / B), policy_from_positive_branch=float(np.sum(cnt * ap * sp) / B),
                   policy_from_complement_branch=float(np.sum(cnt * am * sc) / B),
                   complement_from_single_legal_positions=float(np.sum((cnt * am * sc)[nleg == 1]) / B),
                   value_shared_grad_per_position=float(1 - yv))
res['frac_single_legal'] = float(np.sum(cnt[nleg == 1]) / B); res['mean_a_minus'] = float(np.sum(cnt * am) / B); res['mean_a_plus'] = float(np.sum(cnt * ap) / B)
res['model'] = dict(model_id=md['model_id'], training_step=md.get('training_step'))
print(json.dumps(res, indent=1))
json.dump(res, open("shared16.json", 'w'), indent=1)
