"""fp16 range + rounding audit of one checkpoint.

usage: range16.py <safetensors path> <out.json> [posset.pkl]

(a) weights: per tensor max|w|, fraction of nonzero |w| below the fp16 normal
    range (2^-14) — i.e. stored as an fp16 subnormal — and below 2^-25 (rounds
    to zero), fp16 vs bf16 relative representation error.
(b) activations (float64 forward, every named point): max|x|, min nonzero |x|,
    fraction of nonzero |x| < 2^-14, the L1 accumulation bound of every
    conv/matmul, and at every BN/LN input: max x^2 (what a variance op squares),
    pooled batch variance per channel (training-mode BN), LN per-square variance.
(c) head outputs and whole network under bf16 / fp16 / fp32 emulations vs fp64:
    value logit ties, KL, argmax change, CE delta, |dv|; policy KL, top-1 lost,
    top-2 / top-5 ties, CE delta.
"""
import sys, os, json, time, pickle, math, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from fwd16 import *
import dcm_arch

MP = sys.argv[1]; OUT = sys.argv[2]
PS = sys.argv[3] if len(sys.argv) > 3 else POSSET
md, T = load(MP); sites = dcm_arch.site_activations_md(md); arch = norm_arch_md(md)
T = {k: v for k, v in T.items() if not k.startswith('opt.')}
P = pickle.load(open(PS, 'rb'))
X = np.stack([p['x'] for p in P]).astype(np.float64); N = len(P)
corpus = np.array([p['src'] == 'corpus' for p in P]); labels = np.array([p['label'] for p in P])
legal = [np.asarray(p['legal']) for p in P]; targets = [p['target'] for p in P]
R = dict(model=dict(model_id=md['model_id'], training_step=md.get('training_step'), native=arch['compute_data_type'],
                    path=MP, n_positions=N, n_corpus=int(corpus.sum())))
print(R['model'], flush=True)

# ---------------- (a) weights ----------------
W = {}
allw = []
for k, v in T.items():
    a = np.abs(v).ravel(); nz = a[a > 0]
    e16 = np.abs(f16(v) - v).ravel(); ebf = np.abs(bf16(v) - v).ravel()
    W[k] = dict(n=int(a.size), absmax=float(a.max()), absmin_nz=float(nz.min()) if nz.size else None,
                frac_subnormal=float(np.mean((a > 0) & (a < F16_MIN_NORMAL))),
                frac_to_zero=float(np.mean((a > 0) & (a < F16_MIN_SUB / 2))),
                frac_over_max=float(np.mean(a > F16_MAX)),
                rel_err_rms_f16=float(np.sqrt(np.mean(e16 ** 2)) / max(np.sqrt(np.mean(a ** 2)), 1e-300)),
                rel_err_rms_bf16=float(np.sqrt(np.mean(ebf ** 2)) / max(np.sqrt(np.mean(a ** 2)), 1e-300)))
    if not (k.endswith('running_mean') or k.endswith('running_var')): allw.append(a)
aw = np.concatenate(allw)
R['weights_all_trainable'] = dict(n=int(aw.size), absmax=float(aw.max()), frac_zero=float(np.mean(aw == 0)),
    frac_subnormal=float(np.mean((aw > 0) & (aw < F16_MIN_NORMAL))), frac_to_zero=float(np.mean((aw > 0) & (aw < F16_MIN_SUB / 2))),
    frac_below_1e_minus_3=float(np.mean((aw > 0) & (aw < 1e-3))),
    pct={str(q): float(np.percentile(aw[aw > 0], q)) for q in (0.001, 0.01, 0.1, 1, 50, 99, 99.99)})
bnk = sorted({k.rsplit('.', 1)[0] for k in T if k.endswith('running_var')})
R['bn_params'] = {n: dict(running_var_min=float(T[n + '.running_var'].min()), running_var_max=float(T[n + '.running_var'].max()),
                          running_mean_absmax=float(np.abs(T[n + '.running_mean']).max()),
                          fold_scale_absmax=float(np.max(np.abs(T[n + '.weight']) / np.sqrt(T[n + '.running_var'] + 1e-5))),
                          gamma_absmin=float(np.abs(T[n + '.weight']).min())) for n in bnk}
R['weights'] = W
print('weights', R['weights_all_trainable'], flush=True)
print('worst subnormal tensors', sorted(((v['frac_subnormal'], k) for k, v in W.items()), reverse=True)[:6], flush=True)

# ---------------- (b) activations ----------------
t = time.time()
acc = {}; accb = {}; nin = {}; bstat = {}; lnvar = {}
for s in range(0, N, 64):
    o = forward(T, arch, X[s:s + 64], capture=True, sites=sites)
    for n, x in o['acts'].items():
        a = np.abs(np.asarray(x, dtype=np.float64)).ravel(); nz = a[a > 0]
        d = acc.setdefault(n, dict(absmax=0.0, absmin_nz=np.inf, n=0, nnz=0, nsub=0, nzero16=0, nover=0))
        d['absmax'] = max(d['absmax'], float(a.max())); d['n'] += a.size; d['nnz'] += nz.size
        if nz.size: d['absmin_nz'] = min(d['absmin_nz'], float(nz.min()))
        d['nsub'] += int(np.sum((a > 0) & (a < F16_MIN_NORMAL))); d['nzero16'] += int(np.sum((a > 0) & (a < F16_MIN_SUB / 2)))
        d['nover'] += int(np.sum(a > F16_MAX))
    for n, b in o['acc'].items(): accb[n] = max(accb.get(n, 0.0), b)
    for n, x in o['norm_in'].items():
        d = nin.setdefault(n, dict(absmax=0.0, sqmax=0.0))
        d['absmax'] = max(d['absmax'], float(np.abs(x).max())); d['sqmax'] = max(d['sqmax'], float((x ** 2).max()))
        if n.endswith('.ln'):
            v = x.var(1); lv = lnvar.setdefault(n, [np.inf, 0.0]); lv[0] = min(lv[0], float(v.min())); lv[1] = max(lv[1], float(v.max()))
        else:
            b = bstat.setdefault(n, [0, np.zeros(x.shape[1]), np.zeros(x.shape[1])])
            b[0] += x.shape[0] * x.shape[2] * x.shape[3]; b[1] += x.sum((0, 2, 3)); b[2] += (x ** 2).sum((0, 2, 3))
print('capture', round(time.time() - t), 's', flush=True)
A = {}
for n, d in acc.items():
    A[n] = dict(absmax=d['absmax'], absmin_nz=(None if d['absmin_nz'] == np.inf else d['absmin_nz']),
                frac_nz_subnormal=d['nsub'] / max(d['nnz'], 1), frac_nz_to_zero=d['nzero16'] / max(d['nnz'], 1),
                frac_over=d['nover'] / d['n'], frac_zero=1 - d['nnz'] / d['n'])
    if n in accb: A[n]['accum_L1_bound_max'] = accb[n]
for n, b in accb.items():
    if n not in A: A[n] = dict(accum_L1_bound_max=b)
Nm = {}
for n, d in nin.items():
    Nm[n] = dict(input_absmax=d['absmax'], input_sq_max=d['sqmax'])
    if n in bstat:
        c, s1, s2 = bstat[n]; m = s1 / c; v = s2 / c - m ** 2
        Nm[n].update(batch_var_min=float(v.min()), batch_var_max=float(v.max()), batch_absmean_max=float(np.abs(m).max()),
                     batch_E_x2_max=float((s2 / c).max()))
    if n in lnvar: Nm[n].update(ln_var_min=lnvar[n][0], ln_var_max=lnvar[n][1])
R['activations'] = A; R['norm_inputs'] = Nm
R['activations_summary'] = dict(
    global_absmax=max(v.get('absmax', 0) for v in A.values()),
    argmax_point=max(A, key=lambda k: A[k].get('absmax', 0)),
    global_accum_bound_max=max(v.get('accum_L1_bound_max', 0) for v in A.values()),
    argmax_accum=max(A, key=lambda k: A[k].get('accum_L1_bound_max', 0)),
    norm_input_sq_max=max(v['input_sq_max'] for v in Nm.values()),
    argmax_norm_sq=max(Nm, key=lambda k: Nm[k]['input_sq_max']))
print('act summary', R['activations_summary'], flush=True)
for n in A: print(f'  {n:14s}', {k: (f'{v:.3g}' if isinstance(v, float) else v) for k, v in A[n].items()}, flush=True)
for n in Nm: print(f'  norm {n:10s}', {k: f'{v:.3g}' for k, v in Nm[n].items()}, flush=True)

# ---------------- (c) emulations ----------------
def lsm(z): z = z - z.max(); return z - np.log(np.exp(z).sum())
names = list(forward(T, arch, X[:2], capture=True, sites=sites)['acts'].keys())
internal = frozenset(names) - {'p.convmm', 'v.fc2mm'}
v5 = 'p.pre_bn' in names and arch['block_groups'][0].get('output_norm') == 'layer_norm'
Tb = quantise_weights(T, bf16); Th = quantise_weights(T, f16); Tf = quantise_weights(T, f32)
t = time.time()
o64 = forward_batched(T, arch, X, sites=sites)
runs = {}
# (weights q, internal rounding set, quantiser, head-output quantiser, value-softmax quantiser)
runs['bf16 heads only'] = (T, frozenset(), ident, bf16, bf16)
runs['fp16 heads only'] = (T, frozenset(), ident, f16, f16)
if v5:
    runs['bf16 real (calibrated: p.pre_bn + fused heads)'] = (Tb, frozenset({'p.pre_bn'}), bf16, bf16, bf16)
    runs['fp16 analog of calibrated'] = (Th, frozenset({'p.pre_bn'}), f16, f16, f16)
runs['bf16 per-op'] = (Tb, internal, bf16, bf16, bf16)
runs['fp16 per-op'] = (Th, internal, f16, f16, f16)
runs['fp16 per-op, FTZ subnormals'] = (Th, internal, f16_ftz, f16_ftz, f16_ftz)
runs['bf16 per-op + fp32 head tails'] = (Tb, internal, bf16, f32, f32)
runs['fp16 per-op + fp32 head tails'] = (Th, internal, f16, f32, f32)
runs['fp32 per-op'] = (Tf, internal, f32, f32, f32)
outs = {}
for k, (Tq, Rs, q, qh, qs) in runs.items():
    oo = o64 if (Tq is T and not Rs) else forward_batched(Tq, arch, X, Rs, q, sites=sites)
    outs[k] = (qh(oo['pl_raw']), qh(oo['vl_raw']), qs)
    print('emu', k, round(time.time() - t), 's', flush=True)
p64 = softmax(o64['vl_raw']); v64 = p64[:, 0] - p64[:, 2]; ce64 = -np.log(p64[np.arange(N), labels])
la64 = [lsm(o64['pl_raw'][i][legal[i]]) for i in range(N)]
tg = [legal[i].tolist().index(targets[i]) for i in range(N)]
pce64 = np.array([-la64[i][tg[i]] for i in range(N)])
R['fp64'] = dict(value_ce_corpus=float(ce64[corpus].mean()), policy_ce_corpus=float(pce64[corpus].mean()))
vl = o64['vl_raw']; sh = vl.mean(1); spread = vl.max(1) - vl.min(1)
lm = np.array([o64['pl_raw'][i][legal[i]].mean() for i in range(N)]); lsd = np.array([o64['pl_raw'][i][legal[i]].std() for i in range(N)])
lmax = np.array([o64['pl_raw'][i][legal[i]].max() for i in range(N)])
R['head_levels'] = dict(value_shared_median=float(np.median(sh)), value_shared_p5=float(np.percentile(sh, 5)), value_shared_p95=float(np.percentile(sh, 95)),
    value_absmax=float(np.abs(vl).max()), value_spread_median=float(np.median(spread)),
    policy_legal_mean_median=float(np.median(lm)), policy_legal_std_median=float(np.median(lsd)), policy_legal_max_median=float(np.median(lmax)),
    policy_absmax=float(np.abs(o64['pl_raw']).max()), policy_allmove_mean_median=float(np.median(o64['pl_raw'].mean(1))),
    bf16_step_at_value_median=float(2.0 ** (math.floor(math.log2(abs(np.median(sh)))) - 7)),
    f16_step_at_value_median=float(2.0 ** (math.floor(math.log2(abs(np.median(sh)))) - 10)),
    bf16_step_at_policy_legal_max=float(2.0 ** (math.floor(math.log2(abs(np.median(lmax)))) - 7)),
    f16_step_at_policy_legal_max=float(2.0 ** (math.floor(math.log2(abs(np.median(lmax)))) - 10)))
print('levels', R['head_levels'], flush=True)
R['emu'] = {}
for k, (pl, vlq, qs) in outs.items():
    p = qs(softmax(vlq))
    fin = np.isfinite(p).all(1) & np.isfinite(pl).all(1)
    pn = p / p.sum(1, keepdims=True)
    kl = np.sum(p64 * (np.log(p64) - np.log(np.maximum(pn, 1e-30))), 1)
    ce = -np.log(np.maximum(p[np.arange(N), labels], 1e-30)); v = p[:, 0] - p[:, 2]
    ties = np.array([len(set(r)) < 3 for r in vlq])
    val = dict(nonfinite_rows=int((~fin).sum()), logit_ties=float(ties.mean()), kl_mean=float(np.nanmean(kl)), kl_max=float(np.nanmax(kl)),
               argmax_change=float(np.mean(p.argmax(1) != p64.argmax(1))), dce_corpus=float(np.nanmean((ce - ce64)[corpus])),
               dv_abs_mean=float(np.nanmean(np.abs(v - v64))), dv_abs_max=float(np.nanmax(np.abs(v - v64))))
    kl = []; top1 = []; t2 = []; t5 = []; dce = []
    for i in range(N):
        b = pl[i][legal[i]]; la = la64[i]
        if not np.isfinite(b).all(): kl.append(np.nan); top1.append(True); t2.append(False); t5.append(False); dce.append(np.nan); continue
        lb = lsm(b); pa = np.exp(la)
        kl.append(float(np.sum(pa * (la - lb)))); ia = int(np.argmax(la)); top1.append(b[ia] < b.max())
        ob = np.sort(b)[::-1]; t2.append(len(b) > 1 and ob[0] == ob[1]); t5.append(len(set(ob[:5])) < min(5, len(b)))
        dce.append(-lb[tg[i]] + la[tg[i]])
    kl = np.array(kl); dce = np.array(dce)
    pol = dict(kl_mean=float(np.nanmean(kl)), kl_p90=float(np.nanpercentile(kl, 90)), kl_max=float(np.nanmax(kl)), top1_lost=float(np.mean(top1)),
               top2_ties=float(np.mean(t2)), top5_ties=float(np.mean(t5)), dce_corpus=float(np.nanmean(dce[corpus])))
    R['emu'][k] = dict(value=val, policy=pol)
    print(f'{k:48s} V', {a: f'{b:.3g}' for a, b in val.items()}, flush=True)
    print(f'{"":48s} P', {a: f'{b:.3g}' for a, b in pol.items()}, flush=True)
json.dump(R, open(OUT, 'w'), indent=1)
print('done', flush=True)
