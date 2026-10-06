"""Training-gradient magnitudes vs the fp16 range, for one checkpoint.

usage: grad16.py <model.safetensors> <out.json> [trainer.safetensors mu] [--eps-policy 0.1 --eps-value 0.013]

Exact numpy backward of the loss the trainer builds (ChessTrainer.buildTrainingOps),
restricted to the heads, on a batch of B = 4096 corpus positions (the 900 corpus
positions of posset.pkl resampled with replacement, seed 0):

  policy:  L_p = mean_i [ a+_i CE(y_i, softmax z_i) + a-_i CE(yc_i, softmax z_i) ]
           a± = max(0, ±A/rms(A)),  A = z_outcome − v_baseline (v from the fp64 forward)
  illegal: L_m = mean_i sum_{k illegal} softmax(z_i)_k
  value:   L_v = mean_i CE(yv_i, softmax(vl_i))
  (entropy coefficient taken as 0, the Ejp0 setting)

Reports |g| distributions for the logit gradients (the tensors the fp16
backward would carry first), the head weight gradients, and the activation
gradient arriving at the tower output (inference-mode BN used as the linear
map through the heads' BN — an approximation of the training-mode BN backward).
Also: sum(y) − 1 of the CE targets as the graph builds them in bf16 and fp16
(the per-position push along the softmax-invariant direction).

If a trainer.safetensors with `opt.*.velocity` tensors is given, it also reports
per-tensor velocity magnitudes and the implied per-step gradient range for
momentum mu:  |g| ≈ |v|·(1−mu) (persistent part) … |v|·sqrt(1−mu²) (noise part).
"""
import sys, os, json, pickle, math, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from fwd16 import *

MP = sys.argv[1]; OUT = sys.argv[2]
TR = sys.argv[3] if len(sys.argv) > 3 else None; MU = float(sys.argv[4]) if len(sys.argv) > 4 else None
B = 4096
md, T = load(MP); arch = norm_arch_md(md)
T = {k: v for k, v in T.items() if not k.startswith('opt.')}
P = [p for p in pickle.load(open(POSSET, 'rb')) if p['src'] == 'corpus']
rng = np.random.default_rng(0); idx = rng.integers(0, len(P), B)
X = np.stack([P[i]['x'] for i in idx]).astype(np.float64)
labels = np.array([P[i]['label'] for i in idx]); legal = [np.asarray(P[i]['legal']) for i in idx]; tgt = [P[i]['target'] for i in idx]
R = dict(model=dict(model_id=md['model_id'], training_step=md.get('training_step'), native=arch['compute_data_type'], batch=B))
ui, cnt = np.unique(idx, return_counts=True); U = len(ui)
o = forward_batched(T, arch, np.stack([P[i]['x'] for i in ui]).astype(np.float64), keep=('pl_raw', 'vl_raw', 'f1', 'feat', 'tower'), md=md)
pl = o['pl_raw']; vl = o['vl_raw']; f1 = o['f1']; feat = o['feat']; tower = o['tower']
labels = np.array([P[i]['label'] for i in ui]); legal = [np.asarray(P[i]['legal']) for i in ui]; tgt = [P[i]['target'] for i in ui]
cw = cnt.astype(np.float64)            # multiplicity of each unique position in the B-sample batch
R['model']['unique_positions'] = int(U)

def dist(g, name):
    a = np.abs(np.asarray(g)).ravel(); nz = a[a > 0]
    d = dict(n=int(a.size), absmax=float(a.max()), frac_exact_zero=float(np.mean(a == 0)),
             pct={str(q): float(np.percentile(nz, q)) for q in (0.1, 1, 10, 50, 90, 99, 99.9)},
             frac_nz_below_min_normal=float(np.mean(nz < F16_MIN_NORMAL)), frac_nz_below_half_min_sub=float(np.mean(nz < F16_MIN_SUB / 2)),
             rms=float(np.sqrt(np.mean(a ** 2))))
    # loss scale needed so that 99% (99.9%) of nonzero |g| are fp16-normal; headroom = 65504 / max
    d['scale_for_p1_normal'] = float(F16_MIN_NORMAL / d['pct']['1']); d['scale_for_p0.1_normal'] = float(F16_MIN_NORMAL / d['pct']['0.1'])
    d['scale_headroom_max'] = float(F16_MAX / d['absmax'])
    d['pct']['10'] = float(np.percentile(nz, 10))
    d['by_scale'] = {str(S): dict(frac_subnormal=float(np.mean(nz * S < F16_MIN_NORMAL)), frac_to_zero=float(np.mean(nz * S < F16_MIN_SUB / 2)),
                                  frac_overflow=float(np.mean(nz * S > F16_MAX)),
                                  energy_frac_subnormal=float(np.sum(nz[nz * S < F16_MIN_NORMAL] ** 2) / np.sum(nz ** 2)),
                                  energy_frac_to_zero=float(np.sum(nz[nz * S < F16_MIN_SUB / 2] ** 2) / np.sum(nz ** 2))) for S in (1, 2 ** 8, 2 ** 10, 2 ** 12, 2 ** 14, 2 ** 16)}
    print(f'{name:26s} max {d["absmax"]:.3g}  p50 {d["pct"]["50"]:.3g}  p1 {d["pct"]["1"]:.3g}  subnormal {d["frac_nz_below_min_normal"]:.3f}  ->0 {d["frac_nz_below_half_min_sub"]:.3f}  '
          f'S(p1 normal) {d["scale_for_p1_normal"]:.3g}  S max {d["scale_headroom_max"]:.3g}', flush=True)
    return d

# ---------------- targets (fp64) and their bf16 / fp16 construction ----------------
def policy_targets(i, q, eps):
    li = legal[i]; n = len(li)
    e = q(np.float64(eps)); ome = q(1 - e)
    y = np.zeros(4864); y[tgt[i]] = ome                                   # oneHot * (1 - eps)
    u = np.zeros(4864); u[li] = q(1.0 / n)                                 # legalMask / |legal|
    uc = q(u * e)                                                          # eps * uniform(legal)
    ys = q(q(y) + uc)
    oth = np.zeros(4864); oth[li] = 1.0; oth[tgt[i]] = 0.0
    yc = q(q(q(oth / max(n - 1, 1)) * ome) + uc)
    return ys, yc
def value_target(lab, q, eps):
    e = q(np.float64(eps)); oh = np.eye(3)[lab]
    return q(q(oh * q(1 - e)) + q(q(np.full(3, 1 / 3)) * e))
R['target_sum_minus_1'] = {}
for nm, q in (('fp64', ident), ('bf16', bf16), ('fp16', f16)):
    sp = []; sc = []
    for i in range(0, U, 4):
        ys, yc = policy_targets(i, q, 0.1); sp.append(ys.sum() - 1); sc.append(yc.sum() - 1)
    sv = np.array([value_target(l, q, 0.013).sum() - 1 for l in labels])
    R['target_sum_minus_1'][nm] = dict(policy_mean=float(np.mean(sp)), policy_absmean=float(np.mean(np.abs(sp))),
        complement_mean=float(np.mean(sc)), value_mean=float(sv.mean()), value_by_class=[float(value_target(l, q, 0.013).sum() - 1) for l in (0, 1, 2)])
print('target sums', R['target_sum_minus_1'], flush=True)

# ---------------- policy / illegal / value logit gradients (fp64) ----------------
pv = softmax(vl); v = pv[:, 0] - pv[:, 2]
zout = np.array([{0: 1.0, 1: 0.0, 2: -1.0}[l] for l in labels])
A = zout - v; rms = math.sqrt(max(np.sum(cw * A ** 2) / B, 0.04) + 1e-6); an = A / rms
ap = np.maximum(an, 0); am = np.maximum(-an, 0)
gp = np.zeros((U, 4864)); gill = np.zeros((U, 4864))
for i in range(U):
    p = softmax(pl[i]); ys, yc = policy_targets(i, ident, 0.1)
    gp[i] = (ap[i] * (p - ys) + am[i] * (p - yc)) / B
    ill = np.ones(4864, bool); ill[legal[i]] = False; M = p[ill].sum()
    gill[i] = p * (ill - M) / B
gv = (pv - np.array([value_target(l, ident, 0.013) for l in labels])) / B
glog = gp + gill
R['grad'] = {}
R['grad']['policy_logits (CE part)'] = dist(gp, 'policy dlogit CE')
R['grad']['policy_logits (illegal-mass part)'] = dist(gill, 'policy dlogit illegal')
R['grad']['policy_logits (total)'] = dist(glog, 'policy dlogit total')
lm = np.zeros((U, 4864), bool)
for i in range(U): lm[i, legal[i]] = True
R['grad']['policy_logits legal cells'] = dist(glog[lm], 'policy dlogit legal')
R['grad']['policy_logits illegal cells'] = dist(glog[~lm], 'policy dlogit illegal cells')
R['grad']['value_logits'] = dist(gv, 'value dlogit')

# ---------------- head weight gradients ----------------
def conv_wgrad(x, g, k):
    """dW[o,c,i,j] = sum_{n,y,x} g[n,o,y,x] * xpad[n,c,y+i,x+j]"""
    p = (k - 1) // 2; N, C = x.shape[:2]
    xp = np.zeros((N, C, 8 + 2 * p, 8 + 2 * p)); xp[:, :, p:p + 8, p:p + 8] = x
    O = g.shape[1]; dW = np.zeros((O, C, k, k))
    for i in range(k):
        for j in range(k):
            dW[:, :, i, j] = np.tensordot(g, xp[:, :, i:i + 8, j:j + 8], axes=([0, 2, 3], [0, 2, 3]))
    return dW
def conv_T(g, w):
    """input-gradient of conv(x, w) (same padding): dx = conv(g, flip(w) with in/out swapped)"""
    wt = w.transpose(1, 0, 2, 3)[:, :, ::-1, ::-1]; return conv(g, wt)
G = glog.reshape(U, 76, 8, 8); Gw = G * cw[:, None, None, None]; gvw = gv * cw[:, None]
Wp = T['policy.conv.weight']; kp = Wp.shape[2]
R['grad']['policy.conv.weight'] = dist(conv_wgrad(feat, Gw, kp), 'dW policy.conv')
R['grad']['policy.conv.bias'] = dist(Gw.sum((0, 2, 3)), 'db policy.conv')
W2 = T['value.wdl_fc2.weight']
R['grad']['value.wdl_fc2.weight'] = dist(gvw.T @ f1, 'dW value.fc2')
R['grad']['value.wdl_fc2.bias'] = dist(gvw.sum(0), 'db value.fc2')
gf1 = (gv @ W2) * (f1 > 0); gf1w = gf1 * cw[:, None]
# value.conv -> bn -> relu -> flatten -> fc1
vc = conv(tower, T['value.conv.weight']); vb = (vc - s4(T['value.bn.running_mean'])) / np.sqrt(s4(T['value.bn.running_var']) + 1e-5) * s4(T['value.bn.weight']) + s4(T['value.bn.bias'])
fv = np.maximum(vb, 0).reshape(U, -1)
R['grad']['value.fc1.weight'] = dist(gf1w.T @ fv, 'dW value.fc1')
gfv = (gf1 @ T['value.fc1.weight']) * (fv > 0)
gvb = gfv.reshape(vb.shape) * s4(T['value.bn.weight'] / np.sqrt(T['value.bn.running_var'] + 1e-5))
R['grad']['value.conv.weight'] = dist(conv_wgrad(tower, gvb * cw[:, None, None, None], T['value.conv.weight'].shape[2]), 'dW value.conv')
gtower_v = conv_T(gvb, T['value.conv.weight'])
gfeat = conv_T(G, Wp) * (feat > 0)
if arch['policy_head_style'] == 'intermediate_conv':
    gpre = gfeat * s4(T['policy.pre_bn.weight'] / np.sqrt(T['policy.pre_bn.running_var'] + 1e-5))
    R['grad']['policy.pre_conv.weight'] = dist(conv_wgrad(tower, gpre * cw[:, None, None, None], T['policy.pre_conv.weight'].shape[2]), 'dW policy.pre_conv')
    gtower_p = conv_T(gpre, T['policy.pre_conv.weight'])
else:
    gtower_p = gfeat
R['grad']['activation grad at tower output'] = dist(gtower_p + gtower_v, 'dAct tower output')

# ---------------- velocity-implied gradients for every tensor ----------------
if TR:
    mt, TT = load(TR)
    R['velocity'] = dict(trainer_model_id=mt.get('model_id'), trainer_step=mt.get('training_step'), mu=MU, tensors={})
    allv = []
    for k, vv in TT.items():
        if not (k.startswith('opt.') and k.endswith('.velocity')): continue
        a = np.abs(vv).ravel(); allv.append(a)
        R['velocity']['tensors'][k[4:-9]] = dict(n=int(a.size), median=float(np.median(a)), p1=float(np.percentile(a, 1)), max=float(a.max()),
            frac_implied_g_below_min_normal_persistent=float(np.mean(a * (1 - MU) < F16_MIN_NORMAL)),
            frac_implied_g_below_min_normal_noise=float(np.mean(a * math.sqrt(1 - MU * MU) < F16_MIN_NORMAL)))
    a = np.concatenate(allv); nz = a[a > 0]
    R['velocity']['all'] = dict(n=int(a.size), frac_zero=float(np.mean(a == 0)), pct={str(q): float(np.percentile(nz, q)) for q in (0.1, 1, 10, 50, 90, 99, 99.9)},
        max=float(a.max()), frac_implied_g_below_min_normal_persistent=float(np.mean(a * (1 - MU) < F16_MIN_NORMAL)),
        frac_implied_g_below_min_normal_noise=float(np.mean(a * math.sqrt(1 - MU * MU) < F16_MIN_NORMAL)))
    print('velocity', R['velocity']['all'], flush=True)
    worst = sorted(R['velocity']['tensors'].items(), key=lambda kv: -kv[1]['frac_implied_g_below_min_normal_persistent'])[:8]
    for k, d in worst: print('  ', k, d, flush=True)
json.dump(R, open(OUT, 'w'), indent=1)
print('done', flush=True)
