"""Value- and policy-head bf16 analysis of Ejp0 @ training_step 681000, with
every fix option quantified. Emulations:
  real   = calibrated fit to the bot's recorded outputs: float64 internals,
           bf16 rounding of the policy pre-BN output and of the FUSED head
           outputs (matmul+bias rounded once), bf16 value softmax.
  perop  = pessimistic: bf16 after every op except inside the fused head
           matmul+bias.
  fp32   = every op rounded to fp32 (whole network in fp32).
"""
import sys, os, pickle, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from fwd4 import load, bf16, f32, forward_batched, forward, softmax
from archnorm import norm_arch
MP = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/Library/Application Support/DrewsChessMachine/Models/20260702-Qeu8-resume3-replay-step681000.safetensors')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'heads681.json')
md, T = load(MP); arch = norm_arch(md['architecture'])
print(md['model_id'], md['training_step'], flush=True)
P = pickle.load(open(os.path.join(HERE, 'posset_ejp0.pkl'), 'rb'))
X = np.stack([p['x'] for p in P]).astype(np.float64); N = len(P)
corpus = np.array([p['src'] == 'corpus' for p in P]); lich = ~corpus
c900 = np.array([p['c900'] for p in P]); ours = np.array([p['ours'] for p in P])
labels = np.array([p['label'] for p in P]); legal = [p['legal'] for p in P]; targets = [p['target'] for p in P]
names = list(forward(T, arch, X[:2], capture=True)['acts'].keys())
t = time.time()
o64 = forward_batched(T, arch, X)
oreal = forward_batched(T, arch, X, frozenset({'p.pre_bn'}))
operop = forward_batched(T, arch, X, frozenset(names) - {'p.conv', 'v.fc2mm'})
o32 = forward_batched(T, arch, X, 'ALL', q=f32)
print('forwards', time.time()-t, flush=True)
W2 = T['value.wdl_fc2.weight']; b2 = T['value.wdl_fc2.bias']; m2 = W2.mean(0)
Wp = T['policy.conv.weight'][:, :, 0, 0]; bp = T['policy.conv.bias']
R = {}
# ---------------- value structure ----------------
R['value_struct'] = dict(mean_row_norm=float(np.linalg.norm(m2)), resid_row_norms=[float(v) for v in np.linalg.norm(W2-m2, axis=1)],
    bias=[float(v) for v in b2], bias_mean=float(b2.mean()), bias_mean_init=float(np.log(6)/3),
    row_norms=[float(v) for v in np.linalg.norm(W2, axis=1)], fc1_bias_mean=float(T['value.fc1.bias'].mean()))
vl64 = o64['vl_raw']; shared = vl64.mean(1); spread = vl64.max(1)-vl64.min(1)
f1 = o64['f1']; R['value_logit'] = dict(
    shared_pct={str(q): float(np.percentile(shared, q)) for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)},
    shared_frac_binade={'<256': float(np.mean(np.abs(shared) < 256)), '[256,512)': float(np.mean((np.abs(shared) >= 256) & (np.abs(shared) < 512))),
                        '[512,1024)': float(np.mean((np.abs(shared) >= 512) & (np.abs(shared) < 1024))), '>=1024': float(np.mean(np.abs(shared) >= 1024))},
    spread_pct={str(q): float(np.percentile(spread, q)) for q in (5, 50, 95)},
    shared_from_bias_mean=float(b2.mean()), shared_from_meanrow_med=float(np.median(f1 @ m2)),
    f1_norm_med=float(np.median(np.linalg.norm(f1, axis=1))), f1_active_med=float(np.median((f1 > 0).sum(1))))
# ---------------- value emulations / fixes ----------------
def vprobs_bf(vl): return bf16(softmax(vl))
Wr = bf16(W2-m2); br = bf16(b2-b2.mean())
V = {
 'fp64': (vl64, softmax(vl64)),
 'real (bf16 fused head output)': (bf16(oreal['vl_raw']), None),
 'per-op bf16 (pessimistic)': (bf16(operop['vl_raw']), None),
 '(a) recentered fc2, bf16-stored, bf16 output': (bf16(oreal['f1'] @ Wr.T + br), None),
 '(b) fp32 fc2 projection (bf16 internals)': (f32(oreal['vl_raw']), 'f32'),
 '(c) fp32 per-position mean-subtract, then bf16': (bf16(oreal['vl_raw']-oreal['vl_raw'].mean(1, keepdims=True)), None),
 '(d) whole network fp32': (f32(o32['vl_raw']), 'f32'),
}
p64 = softmax(vl64); v64 = p64[:, 0]-p64[:, 2]
ce64 = -np.log(p64[np.arange(N), labels])
R['value_fix'] = {}
for k, (vl, mode) in V.items():
    if k == 'fp64': p = p64
    elif mode == 'f32': p = f32(softmax(vl))
    else: p = vprobs_bf(vl)
    kl = np.sum(p64*(np.log(p64)-np.log(np.maximum(p, 1e-30))), 1)
    ce = -np.log(np.maximum(p[np.arange(N), labels], 1e-30)); v = p[:, 0]-p[:, 2]
    ties = np.array([len(set(r)) < 3 for r in vl])
    R['value_fix'][k] = dict(logit_tie=float(ties.mean()), logit_tie_ours=float(ties[ours].mean()),
        kl_mean=float(kl.mean()), kl_p90=float(np.percentile(kl, 90)), kl_max=float(kl.max()),
        argmax_change=float(np.mean(p.argmax(1) != p64.argmax(1))),
        ce_corpus=float(ce[corpus].mean()), dce_corpus=float((ce-ce64)[corpus].mean()), ce_c900=float(ce[c900].mean()),
        ce_lichess=float(ce[lich].mean()), dce_lichess=float((ce-ce64)[lich].mean()),
        dv_abs_mean=float(np.abs(v-v64).mean()), dv_abs_p90=float(np.percentile(np.abs(v-v64), 90)), dv_abs_max=float(np.abs(v-v64).max()),
        dW_abs_mean=float(np.abs(p[:, 0]-p64[:, 0]).mean()), dD_abs_mean=float(np.abs(p[:, 1]-p64[:, 1]).mean()), dL_abs_mean=float(np.abs(p[:, 2]-p64[:, 2]).mean()),
        maxlogit_abs_med=float(np.median(np.abs(vl).max(1))))
    print(k, R['value_fix'][k], flush=True)
# ground truth from the bot records (ours only)
oi = np.where(ours)[0]
rec = np.array([[P[i]['obs']['win'], P[i]['obs']['draw'], P[i]['obs']['loss']] for i in oi], dtype=np.float64)
rec_tie = np.array([len(set(r)) < 3 for r in rec.astype(np.float32).tolist()])
vr = rec[:, 0]-rec[:, 2]
real_p = vprobs_bf(bf16(oreal['vl_raw']))[oi]
R['records'] = dict(n=int(len(oi)), recorded_wdl_any_tie=float(rec_tie.mean()),
    recorded_vs_fp64_dv_abs_mean=float(np.abs(vr-v64[oi]).mean()), recorded_vs_fp64_dv_abs_p90=float(np.percentile(np.abs(vr-v64[oi]), 90)),
    recorded_vs_fp64_dv_abs_max=float(np.abs(vr-v64[oi]).max()),
    recorded_vs_fp64_argmax_change=float(np.mean(rec.argmax(1) != p64[oi].argmax(1))),
    recorded_vs_emulated_exact=float(np.mean(np.all(real_p.astype(np.float32) == rec.astype(np.float32), 1))),
    recorded_top2_tie=float(np.mean([len(P[i]['obs']['topMoves']) > 1 and P[i]['obs']['topMoves'][0]['probability'] == P[i]['obs']['topMoves'][1]['probability'] for i in oi])),
    recorded_pD_med=float(np.median(rec[:, 1])), fp64_pD_med_ours=float(np.median(p64[oi, 1])))
# start positions (ply 0, White = us)
sp = [i for i in oi if P[i]['ply'] == 0]
Ra = V['(a) recentered fc2, bf16-stored, bf16 output'][0]
R['start_position'] = dict(games=[P[i]['game'] for i in sp], vl_fp64=[float(v) for v in vl64[sp[0]]],
    wdl_fp64=[float(v) for v in p64[sp[0]]], wdl_bf16_emulated=[float(v) for v in vprobs_bf(bf16(oreal['vl_raw'][sp[0]]))],
    vl_bf16_emulated=[float(v) for v in bf16(oreal['vl_raw'][sp[0]])],
    wdl_recorded=[[P[i]['obs']['win'], P[i]['obs']['draw'], P[i]['obs']['loss']] for i in sp],
    wdl_recentered_bf16=[float(v) for v in vprobs_bf(Ra[sp[0]])], vl_recentered_bf16=[float(v) for v in Ra[sp[0]]])
print('records', R['records'], '\nstart', R['start_position'], flush=True)
# ---------------- policy structure ----------------
mp = Wp.mean(0)
R['policy_struct'] = dict(mean_row_norm=float(np.linalg.norm(mp)), resid_norm_med=float(np.median(np.linalg.norm(Wp-mp, axis=1))),
    bias_mean=float(bp.mean()), bias_std=float(bp.std()), bias_min=float(bp.min()), bias_max=float(bp.max()))
pl64 = o64['pl_raw']
lm = np.array([pl64[i][legal[i]].mean() for i in range(N)]); ls = np.array([pl64[i][legal[i]].std() for i in range(N)])
lmax = np.array([pl64[i][legal[i]].max() for i in range(N)]); gm = pl64.mean(1)
L3 = pl64.reshape(N, 76, 64); sq = L3.mean(1)
def ulp(v): return 2.0**(np.floor(np.log2(np.abs(v)))-7)
R['policy_logit'] = dict(legal_mean_pct={str(q): float(np.percentile(lm, q)) for q in (1, 5, 50, 95, 99)},
    legal_std_med=float(np.median(ls)), legal_max_med=float(np.median(lmax)), allmove_mean_med=float(np.median(gm)),
    allmove_mean_std_across_pos=float(gm.std()), legal_mean_std_across_pos=float(lm.std()),
    legal_minus_allmove_med=float(np.median(lm-gm)), bf16_ulp_at_legal_max_med=float(np.median(ulp(lmax))),
    persquare_component_std_within_pos_med=float(np.median(sq.std(1))), resid_std_within_pos_med=float(np.median((L3-sq[:, None, :]).std((1, 2)))),
    frac_legal_max_in_binade={k: float(v) for k, v in zip(('<32', '[32,64)', '[64,128)', '[128,256)', '>=256'),
        [np.mean(np.abs(lmax) < 32), np.mean((np.abs(lmax) >= 32) & (np.abs(lmax) < 64)), np.mean((np.abs(lmax) >= 64) & (np.abs(lmax) < 128)),
         np.mean((np.abs(lmax) >= 128) & (np.abs(lmax) < 256)), np.mean(np.abs(lmax) >= 256)])})
print('policy', R['policy_struct'], R['policy_logit'], flush=True)
# ---------------- policy emulations / fixes ----------------
def lsm(z): z = z-z.max(); return z-np.log(np.exp(z).sum())
K = float(np.round(np.median(lm)))
bK = bf16(bp-K); bM = bf16(bp-bp.mean())
pmr = oreal['pm']
chanrep = lambda b: np.repeat(b, 64)[None, :]
plr = oreal['pl_raw']
Pv = {
 'real (bf16 fused head output)': bf16(plr),
 'per-op bf16 (pessimistic)': bf16(operop['pl_raw']),
 'bias-mean recenter (b - mean b), bf16': bf16(pmr + chanrep(bM)),
 f'constant shift to legal level (b - {K:g}), bf16': bf16(pmr + chanrep(bK)),
 '(b) fp32 final projection (bf16 internals)': f32(plr),
 '(c1) fp32 per-position all-move mean-subtract, then bf16': bf16(plr - plr.mean(1, keepdims=True)),
 '(c2) fp32 per-position max-subtract, then bf16': bf16(plr - plr.max(1, keepdims=True)),
 '(d) whole network fp32': f32(o32['pl_raw']),
 'fp64 internals, fp32 output (reference floor)': f32(pl64),
}
# per-square weight recentering is NOT softmax-invariant: exact-math damage
plw = (L3 - L3.mean(1, keepdims=True)).reshape(N, -1)
Pv['weight recenter (W - mean row): exact math, NOT invariant'] = plw
R['policy_fix'] = {}
la64 = [lsm(pl64[i][legal[i]]) for i in range(N)]
for k, plq in Pv.items():
    kl = []; top1 = []; tie2 = []; tie5 = []; ce64l = []; ceql = []; tv = []
    for i in range(N):
        li = legal[i]; b = plq[i][li].astype(np.float32).astype(np.float64); la = la64[i]; lb = lsm(b)
        pa = np.exp(la); pb = np.exp(lb)
        kl.append(float(np.sum(pa*(la-lb)))); tv.append(0.5*np.abs(pa-pb).sum())
        ob = np.sort(b)[::-1]; ia = int(np.argmax(la))
        top1.append(b[ia] < b.max()); tie2.append(len(li) > 1 and ob[0] == ob[1]); tie5.append(len(set(ob[:5])) < min(5, len(li)))
        tt = li.tolist().index(targets[i]); ce64l.append(-la[tt]); ceql.append(-lb[tt])
    kl = np.array(kl); ce64l = np.array(ce64l); ceql = np.array(ceql)
    R['policy_fix'][k] = dict(kl_mean=float(kl.mean()), kl_p90=float(np.percentile(kl, 90)), kl_max=float(kl.max()), tv_mean=float(np.mean(tv)),
        top1_lost=float(np.mean(top1)), tie_top2=float(np.mean(tie2)), tie_top2_ours=float(np.mean(np.array(tie2)[ours])), tie_top5=float(np.mean(tie5)),
        ce_corpus=float(ceql[corpus].mean()), dce_corpus=float((ceql-ce64l)[corpus].mean()), ce_c900=float(ceql[c900].mean()), dce_c900=float((ceql-ce64l)[c900].mean()))
    print(k, R['policy_fix'][k], flush=True)
R['policy_ce_fp64_corpus'] = float(ce64l[corpus].mean()); R['value_ce_fp64_corpus'] = float(ce64[corpus].mean())
R['value_ce_fp64_c900'] = float(ce64[c900].mean()); R['policy_ce_fp64_c900'] = float(ce64l[c900].mean())
fp64top2 = np.array([len(legal[i]) > 1 and np.sort(pl64[i][legal[i]])[-1]-np.sort(pl64[i][legal[i]])[-2] < 1e-3 for i in range(N)])
R['policy_fp64_top2_within_1e-3'] = float(fp64top2.mean())
R['model'] = dict(model_id=md['model_id'], training_step=md['training_step'], n_positions=N, n_corpus=int(corpus.sum()), n_lichess=int(lich.sum()), n_ours=int(ours.sum()), K=K)
json.dump(R, open(OUT, 'w'), indent=1)
np.savez_compressed(OUT.replace('.json', '.npz'), vl64=vl64, pl_legal_mean=lm, shared=shared, vl_real=bf16(oreal['vl_raw']))
print('done', flush=True)
