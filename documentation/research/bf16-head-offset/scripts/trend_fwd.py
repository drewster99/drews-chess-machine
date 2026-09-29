"""Forward-based head metrics at sampled checkpoints along the Ejp0 lineage
(files chosen from trend_struct.csv, i.e. by metadata model_id + step).
Positions: the prior survey's 900 corpus positions (c900) + every 3rd of our
own Lichess-bot positions. Emulation 'real' = calibrated (policy pre-BN
output + fused head outputs rounded); 'heads' = prior survey's emulation
(float64 internals, only head outputs rounded)."""
import sys, os, csv, json, pickle, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from fwd4 import load, bf16, forward_batched, softmax
from archnorm import norm_arch
D = os.path.expanduser('~/Library/Application Support/DrewsChessMachine/Models/')
rows = list(csv.DictReader(open(os.path.join(HERE, 'trend_struct.csv'))))
WANT = {('20260702-7-Qeu8', 0), ('20260702-9-GLu5', 41000), ('20260703-1-Lnji', 67508), ('20260706-1-PVZp', 67000)}
WANT |= {('20260727-1-Ejp0', s) for s in (1000, 10000, 20000, 30000, 50000, 75000, 100000, 150000, 200000, 300000, 400000, 500000,
                                           600000, 681000, 800000, 1000000, 1200000, 1397000)}
sel = [r for r in rows if (r['model_id'], int(r['step'])) in WANT]
P = pickle.load(open(os.path.join(HERE, 'posset_ejp0.pkl'), 'rb'))
oi = [i for i, p in enumerate(P) if p['ours']][::3]
idx = [i for i, p in enumerate(P) if p['c900']] + oi
Q = [P[i] for i in idx]; X = np.stack([p['x'] for p in Q]).astype(np.float64); N = len(Q)
corpus = np.array([p['src'] == 'corpus' for p in Q]); labels = np.array([p['label'] for p in Q])
legal = [p['legal'] for p in Q]; targets = [p['target'] for p in Q]
def lsm(z): z = z-z.max(); return z-np.log(np.exp(z).sum())
out = []
for r in sel:
    md, T = load(D+r['file']); assert md['model_id'] == r['model_id']; arch = norm_arch(md['architecture'])
    t = time.time()
    o64 = forward_batched(T, arch, X); oreal = forward_batched(T, arch, X, frozenset({'p.pre_bn'}))
    vl = o64['vl_raw']; p64 = softmax(vl); ce64 = -np.log(p64[np.arange(N), labels])
    d = dict(model_id=r['model_id'], step=int(r['step']), file=r['file'])
    d['v_shared_med'] = float(np.median(vl.mean(1))); d['v_shared_p5'] = float(np.percentile(vl.mean(1), 5)); d['v_shared_p95'] = float(np.percentile(vl.mean(1), 95))
    d['v_spread_med'] = float(np.median(vl.max(1)-vl.min(1))); d['v_ce64_c900'] = float(ce64[corpus].mean())
    for tag, o in (('real', oreal), ('heads', o64)):
        vq = bf16(o['vl_raw']); pq = bf16(softmax(vq)); ce = -np.log(np.maximum(pq[np.arange(N), labels], 1e-30))
        d[f'v_tie_{tag}'] = float(np.mean([len(set(x)) < 3 for x in vq]))
        d[f'v_dce_c900_{tag}'] = float((ce-ce64)[corpus].mean())
        d[f'v_dv_abs_mean_{tag}'] = float(np.abs((pq[:, 0]-pq[:, 2])-(p64[:, 0]-p64[:, 2])).mean())
        pl = bf16(o['pl_raw']); pl64 = o64['pl_raw']
        kl = []; tie2 = []; dce = []; top1 = []
        for i in range(N):
            li = legal[i]; la = lsm(pl64[i][li]); b = pl[i][li]; lb = lsm(b); pa = np.exp(la)
            kl.append(float(np.sum(pa*(la-lb)))); ob = np.sort(b)[::-1]; tie2.append(len(li) > 1 and ob[0] == ob[1])
            top1.append(b[int(np.argmax(la))] < b.max())
            tt = li.tolist().index(targets[i]); dce.append(-lb[tt]+la[tt])
        d[f'p_kl_{tag}'] = float(np.mean(kl)); d[f'p_tie2_{tag}'] = float(np.mean(tie2)); d[f'p_top1_lost_{tag}'] = float(np.mean(top1))
        d[f'p_dce_c900_{tag}'] = float(np.array(dce)[corpus].mean())
    pl64 = o64['pl_raw']
    d['p_legal_mean_med'] = float(np.median([pl64[i][legal[i]].mean() for i in range(N)]))
    d['p_legal_std_med'] = float(np.median([pl64[i][legal[i]].std() for i in range(N)]))
    d['p_allmove_mean_med'] = float(np.median(pl64.mean(1)))
    d['v_meanrow_norm'] = float(r['v_meanrow_norm']); d['v_bias_mean'] = float(r['v_bias_mean']); d['p_bias_mean'] = float(r['p_bias_mean'])
    out.append(d); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if k != 'file'}, f'{time.time()-t:.0f}s', flush=True)
    json.dump(out, open(os.path.join(HERE, 'trend_fwd.json'), 'w'), indent=1)
