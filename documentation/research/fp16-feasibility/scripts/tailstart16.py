"""Where must the policy fp32 tail start? bf16 per-op body (pessimistic) with the
policy head's fp32 region starting (a) at the final conv (current plan), (b) at
the pre-BN normalize (BN + ReLU + final conv in fp32), (c) at the pre-conv."""
import sys, os, json, pickle, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from fwd16 import *
A = os.path.expanduser('~/Library/Application Support/DrewsChessMachine/')
P = pickle.load(open(POSSET, 'rb'))
X = np.stack([p['x'] for p in P]).astype(np.float64); N = len(P)
corpus = np.array([p['src'] == 'corpus' for p in P]); legal = [np.asarray(p['legal']) for p in P]
tg = [list(legal[i]).index(P[i]['target']) for i in range(N)]
def lsm(z): z = z - z.max(); return z - np.log(np.exp(z).sum())
out = {}
for rel in sys.argv[1:]:
    md, T = load(A + rel); arch = norm_arch_md(md); T = {k: v for k, v in T.items() if not k.startswith('opt.')}
    names = list(forward(T, arch, X[:2], capture=True, md=md)['acts'].keys())
    internal = frozenset(names) - {'p.convmm', 'v.fc2mm'}
    o64 = forward_batched(T, arch, X, md=md); la64 = [lsm(o64['pl_raw'][i][legal[i]]) for i in range(N)]
    r = {}
    for dt, q in (('bf16', bf16), ('fp16', f16)):
        Tq = quantise_weights(T, q)
        for lab, Rs in (('tail from final conv (plan)', internal),
                        ('tail from pre-BN normalize', internal - {'p.pre_bn', 'p.pre_bn.sub'}),
                        ('tail from pre-conv', internal - {'p.pre_bn', 'p.pre_bn.sub', 'p.pre_conv'})):
            # weights in the fp32 tail are the stored (compute-dtype-exact) values, as the plan specifies
            o = forward_batched(Tq, arch, X, Rs, q, md=md); pl = f32(o['pl_raw'])
            kl = []; top1 = []
            for i in range(N):
                b = pl[i][legal[i]]; lb = lsm(b); la = la64[i]; kl.append(np.sum(np.exp(la) * (la - lb))); top1.append(b[int(np.argmax(la))] < b.max())
            r[f'{dt} per-op body, {lab}'] = dict(p_kl=float(np.mean(kl)), p_kl_max=float(np.max(kl)), p_top1_lost=float(np.mean(top1)))
            print(md['model_id'], md.get('training_step'), f'{dt} per-op body, {lab}', {k: f'{v:.3g}' for k, v in r[f'{dt} per-op body, {lab}'].items()}, flush=True)
    out[md['model_id'] + '@' + str(md.get('training_step'))] = r
json.dump(out, open('tailstart16.json', 'w'), indent=1)
