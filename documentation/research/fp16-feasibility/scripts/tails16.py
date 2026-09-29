"""Head-tail options under the calibrated emulation (float64 internals except the
policy pre-BN output, which the bot-record calibration shows is materialised in
the compute dtype) for the v5 architecture: compute-dtype head outputs vs fp32
head tails, bf16 vs fp16."""
import sys, os, json, pickle, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from fwd16 import *
A = os.path.expanduser('~/Library/Application Support/DrewsChessMachine/')
P = pickle.load(open(POSSET, 'rb'))
X = np.stack([p['x'] for p in P]).astype(np.float64); N = len(P)
corpus = np.array([p['src'] == 'corpus' for p in P]); labels = np.array([p['label'] for p in P])
legal = [np.asarray(p['legal']) for p in P]; tg = [list(legal[i]).index(P[i]['target']) for i in range(N)]
def lsm(z): z = z - z.max(); return z - np.log(np.exp(z).sum())
out = {}
for rel in sys.argv[1:]:
    md, T = load(A + rel); arch = norm_arch(md['architecture']); T = {k: v for k, v in T.items() if not k.startswith('opt.')}
    o64 = forward_batched(T, arch, X)
    p64 = softmax(o64['vl_raw']); ce64 = -np.log(p64[np.arange(N), labels]); la64 = [lsm(o64['pl_raw'][i][legal[i]]) for i in range(N)]
    r = {}
    for dt, q in (('bf16', bf16), ('fp16', f16)):
        o = forward_batched(quantise_weights(T, q), arch, X, frozenset({'p.pre_bn'}), q)
        for tail, qt in ((dt + ' head outputs', q), ('fp32 head tails', f32)):
            vl = qt(o['vl_raw']); p = qt(softmax(vl)); pn = p / p.sum(1, keepdims=True)
            vkl = np.sum(p64 * (np.log(p64) - np.log(np.maximum(pn, 1e-30))), 1)
            ce = -np.log(np.maximum(p[np.arange(N), labels], 1e-30))
            pl = qt(o['pl_raw']); kl = []; top1 = []; t2 = []; dce = []
            for i in range(N):
                b = pl[i][legal[i]]; lb = lsm(b); la = la64[i]; pa = np.exp(la)
                kl.append(np.sum(pa * (la - lb))); top1.append(b[int(np.argmax(la))] < b.max())
                ob = np.sort(b)[::-1]; t2.append(len(b) > 1 and ob[0] == ob[1]); dce.append(-lb[tg[i]] + la[tg[i]])
            kl = np.array(kl); dce = np.array(dce)
            r[f'{dt} calibrated, {tail}'] = dict(v_ties=float(np.mean([len(set(x)) < 3 for x in vl])), v_kl=float(vkl.mean()), v_argmax=float(np.mean(p.argmax(1) != p64.argmax(1))),
                v_dce=float((ce - ce64)[corpus].mean()), p_kl=float(kl.mean()), p_kl_max=float(kl.max()), p_top1_lost=float(np.mean(top1)), p_top2_ties=float(np.mean(t2)), p_dce=float(dce[corpus].mean()))
            print(md['model_id'], md.get('training_step'), f'{dt} calibrated, {tail}', {k: f'{v:.3g}' for k, v in r[f'{dt} calibrated, {tail}'].items()}, flush=True)
    fp64_top2 = np.mean([len(legal[i]) > 1 and np.sort(o64['pl_raw'][i][legal[i]])[-1] - np.sort(o64['pl_raw'][i][legal[i]])[-2] < 1e-2 for i in range(N)])
    r['fp64 top-2 gap < 0.01'] = float(fp64_top2)
    out[md['model_id'] + '@' + str(md.get('training_step'))] = r
json.dump(out, open("tails16.json", 'w'), indent=1)
