"""Training-time effect of bf16 head outputs on the loss GRADIENT (not just
inference error), Ejp0 @ 681000, corpus positions, the trainer's own targets
(value eps=0.013 over 3 classes, policy eps=0.1 over legal; both built in
bf16 as ChessTrainer.buildTrainingOps does). Also: SE-gate and ReZero-tanh
saturation where a bf16 output of exactly 1.0 makes a derivative computed from
the output (s*(1-s), 1-t^2) exactly zero."""
import sys, os, pickle, json, math, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from fwd4 import load, bf16, forward_batched, forward, softmax
from archnorm import norm_arch
MP = os.path.expanduser('~/Library/Application Support/DrewsChessMachine/Models/20260702-Qeu8-resume3-replay-step681000.safetensors')
md, T = load(MP); arch = norm_arch(md['architecture'])
P = [p for p in pickle.load(open(os.path.join(HERE, 'posset_ejp0.pkl'), 'rb')) if p['src'] == 'corpus']
X = np.stack([p['x'] for p in P]).astype(np.float64); N = len(P)
labels = np.array([p['label'] for p in P])
o = forward_batched(T, arch, X, frozenset({'p.pre_bn'}), keep=('pl_raw', 'vl_raw', 'f1', 'pm', 'vm'))
o64 = forward_batched(T, arch, X, keep=('pl_raw', 'vl_raw', 'f1'))
R = {}
def cos(a, b): return float((a*b).sum()/np.linalg.norm(a)/np.linalg.norm(b))
# ---- value ----
eps = 0.013
yv = bf16((1-eps)*np.eye(3)[labels] + bf16(np.float64(eps)*bf16(1/3)))
p64 = softmax(o64['vl_raw']); pb = softmax(bf16(o['vl_raw']))
for tag, Y in (('bf16 target', yv), ('exact target', (1-eps)*np.eye(3)[labels]+eps/3)):
    g64 = p64 - Y; gb = pb - Y
    # remove the shared (softmax-invisible) direction to compare only the functional part
    c = lambda g: g - g.mean(1, keepdims=True)
    rel = np.linalg.norm(c(gb)-c(g64), axis=1)/np.linalg.norm(c(g64), axis=1)
    f1 = o['f1']; Gw64 = c(g64).T @ o64['f1']; Gwb = c(gb).T @ f1
    batches = [cos(c(gb[i:i+674]).T @ f1[i:i+674], c(g64[i:i+674]).T @ o64['f1'][i:i+674]) for i in range(0, N, 674)]
    R['value_'+tag] = dict(target_sum_minus_1_mean=float((Y.sum(1)-1).mean()),
        grad_rel_err_med=float(np.median(rel)), grad_rel_err_p90=float(np.percentile(rel, 90)),
        fc2W_grad_cosine_all=cos(Gwb, Gw64), fc2W_grad_cosine_batches=batches,
        shared_grad_per_pos_mean_bf16=float(gb.sum(1).mean()), shared_grad_per_pos_mean_fp64=float(g64.sum(1).mean()),
        shared_grad_if_probs_bf16_rounded=float((bf16(pb)-Y).sum(1).mean()))
    print('value', tag, R['value_'+tag], flush=True)
# ---- policy ----
peps = 0.1
pl64 = o64['pl_raw']; plb = bf16(o['pl_raw'])
rels = []; G64 = np.zeros((76, 512)); Gb = np.zeros((76, 512)); sh64 = []; shb = []; ysum = []
feat = None
fe = forward_batched(T, arch, X, frozenset({'p.pre_bn'}), keep=('feat',))['feat']  # [N,512,8,8] (bf16-rounded pre-BN, relu)
for i in range(N):
    li = P[i]['legal']; y = np.zeros(4864); y[li] = bf16(bf16(peps)/len(li)); y[P[i]['target']] += bf16(1-peps); y = bf16(y)
    ysum.append(y.sum()-1)
    a = softmax(pl64[i]); b = softmax(plb[i]); ga = a-y; gb_ = b-y
    rels.append(np.linalg.norm(gb_-ga)/np.linalg.norm(ga)); sh64.append(ga.sum()); shb.append(gb_.sum())
    F = fe[i].reshape(512, 64)
    G64 += ga.reshape(76, 64) @ F.T; Gb += gb_.reshape(76, 64) @ F.T
R['policy'] = dict(target_sum_minus_1_mean=float(np.mean(ysum)), grad_rel_err_med=float(np.median(rels)), grad_rel_err_p90=float(np.percentile(rels, 90)),
    convW_grad_cosine_all=cos(Gb, G64), shared_grad_per_pos_mean_bf16=float(np.mean(shb)), shared_grad_per_pos_mean_fp64=float(np.mean(sh64)))
print('policy', R['policy'], flush=True)
# ---- saturation of bf16 nonlinearities (derivative from output == 0) ----
sat = {}
for s in range(0, 1024, 256):
    a = forward(T, arch, X[s:s+256], capture=True)['acts']
    for k in ('b0.se.sig', 'b1.se.sig'):
        v = bf16(a[k]); d = sat.setdefault(k, [0, 0, 0]); d[0] += int((v == 1.0).sum()); d[1] += int((v == 0.0).sum()); d[2] += v.size
R['se_gate_bf16_exact_1_or_0'] = {k: dict(frac_eq1=v[0]/v[2], frac_eq0=v[1]/v[2]) for k, v in sat.items()}
C = arch['block_groups'][0]['rezero_alpha_init']
R['rezero'] = {f'b{i}': dict(alpha_raw=float(T[f'blocks.{i}.rezero_alpha'].reshape(-1)[0]),
    tanh_fp64=math.tanh(float(T[f'blocks.{i}.rezero_alpha'].reshape(-1)[0])/C),
    tanh_bf16=float(bf16(math.tanh(float(T[f'blocks.{i}.rezero_alpha'].reshape(-1)[0])/C))),
    dalpha_eff_dalpha_fp64=1-math.tanh(float(T[f'blocks.{i}.rezero_alpha'].reshape(-1)[0])/C)**2) for i in range(2)}
print(R['se_gate_bf16_exact_1_or_0'], R['rezero'], flush=True)
json.dump(R, open(os.path.join(HERE, 'gradcheck.json'), 'w'), indent=1)
