"""Whole-network bf16 audit of Ejp0 @ 681000.
Stage 1 (stats): for every named activation (float64 forward), dynamic range,
  per-channel offset-to-spread |mean_c|/std_c, bf16 rounding error relative to
  the per-channel spread, and — at every normalisation input — the error of a
  bf16-rounded input relative to the scale the normaliser divides by
  (cancellation risk).
Stage 2 (scan): bf16 rounding at ONE point, heads unrounded, vs float64:
  value KL / CE delta, policy KL / top-1 / CE delta."""
import sys, os, pickle, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from fwd4 import load, bf16, forward_batched, forward, softmax
from archnorm import norm_arch
MP = os.path.expanduser('~/Library/Application Support/DrewsChessMachine/Models/20260702-Qeu8-resume3-replay-step681000.safetensors')
md, T = load(MP); arch = norm_arch(md['architecture'])
P = pickle.load(open(os.path.join(HERE, 'posset_ejp0.pkl'), 'rb'))
X = np.stack([p['x'] for p in P]).astype(np.float64); N = len(P)
corpus = np.array([p['src'] == 'corpus' for p in P]); labels = np.array([p['label'] for p in P])
legal = [p['legal'] for p in P]; targets = [p['target'] for p in P]
names = list(forward(T, arch, X[:2], capture=True)['acts'].keys())
stage = sys.argv[1]
# which normaliser consumes which activation, and with what scale
BN_IN = {'stem.conv': 'stem.bn', 'b0.ln': 'blocks.1.bn1', 'b0.conv1': 'blocks.0.bn2', 'b1.conv1': 'blocks.1.bn2',
         'b1.ln': 'tower_final_bn', 'p.pre_conv': 'policy.pre_bn', 'v.conv': 'value.bn'}
BN_IN_STEM = {'stem.bn': 'blocks.0.bn1'}
LN_IN = {'b0.add', 'b1.add'}
if stage == 'stats':
    acc = {}
    lnr = {n: [] for n in LN_IN}; lnerr = {n: [] for n in LN_IN}
    for s in range(0, N, 128):
        a = forward(T, arch, X[s:s+128], capture=True)['acts']
        for n, x in a.items():
            x = np.asarray(x, dtype=np.float64)
            if x.ndim == 0: x = x.reshape(1, 1)
            xc = x if x.ndim == 2 else x.transpose(1, 0, 2, 3).reshape(x.shape[1], -1).T  # [samples, C]
            if x.ndim == 4: xc = x.transpose(0, 2, 3, 1).reshape(-1, x.shape[1])
            e = bf16(xc)-xc
            d = acc.setdefault(n, dict(cnt=0, s=0.0, ss=0.0, mx=0.0, cs=np.zeros(xc.shape[1]), css=np.zeros(xc.shape[1]), es=np.zeros(xc.shape[1]), shape=list(x.shape[1:])))
            d['cnt'] += xc.shape[0]; d['s'] += xc.sum(); d['ss'] += (xc**2).sum(); d['mx'] = max(d['mx'], float(np.abs(xc).max()))
            d['cs'] += xc.sum(0); d['css'] += (xc**2).sum(0); d['es'] += (e**2).sum(0)
            if n in LN_IN:
                mu = x.mean(1); sd = x.std(1)
                lnr[n].append((np.abs(mu)/sd).ravel())
                # error of LN output if its INPUT is bf16-rounded (relative to the LN divisor)
                lnerr[n].append((np.sqrt(((bf16(x)-x)**2).mean(1))/sd).ravel())
    out = {}
    for n, d in acc.items():
        c = d['cnt']; ce = c*len(d['cs']); m = d['s']/ce; sd = np.sqrt(max(d['ss']/ce-m*m, 0))
        cm = d['cs']/c; cv = np.maximum(d['css']/c-cm**2, 0); csd = np.sqrt(cv); er = np.sqrt(d['es']/c)
        ok = csd > 1e-12
        ratio = np.abs(cm[ok])/csd[ok]; eos = er[ok]/csd[ok]
        r = dict(shape=d['shape'], absmax=d['mx'], mean=float(m), std=float(sd), rms=float(np.sqrt(d['ss']/ce)), ch_offset_ratio_med=float(np.median(ratio)) if ok.any() else None,
                 ch_offset_ratio_max=float(ratio.max()) if ok.any() else None,
                 bf16_err_over_ch_spread_med=float(np.median(eos)) if ok.any() else None, bf16_err_over_ch_spread_max=float(eos.max()) if ok.any() else None,
                 dead_channels=int((~ok).sum()))
        if n in BN_IN or n in BN_IN_STEM:
            bnn = BN_IN.get(n) or BN_IN_STEM[n]
            rm = T[bnn+'.running_mean']; rv = T[bnn+'.running_var']
            r['bn_consumer'] = bnn
            r['bn_runmean_over_runsd_max'] = float(np.max(np.abs(rm)/np.sqrt(rv+1e-5)))
            r['bn_runmean_over_runsd_med'] = float(np.median(np.abs(rm)/np.sqrt(rv+1e-5)))
            r['bn_input_bf16err_over_runsd_max'] = float(np.max(er/np.sqrt(rv+1e-5)))
            r['bn_input_bf16err_over_runsd_med'] = float(np.median(er/np.sqrt(rv+1e-5)))
            r['bn_scale_max'] = float(np.max(np.abs(T[bnn+'.weight'])/np.sqrt(rv+1e-5)))
        if n in LN_IN:
            lr = np.concatenate(lnr[n]); le = np.concatenate(lnerr[n])
            r['ln_persquare_offset_ratio_med'] = float(np.median(lr)); r['ln_persquare_offset_ratio_p99'] = float(np.percentile(lr, 99)); r['ln_persquare_offset_ratio_max'] = float(lr.max())
            r['ln_input_bf16err_over_sd_med'] = float(np.median(le)); r['ln_input_bf16err_over_sd_max'] = float(le.max())
        out[n] = r
        print(n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    # weights: rezero alpha, LN gamma/beta, BN params
    W = {}
    for i in range(arch['block_groups'][0]['count']):
        a = float(T[f'blocks.{i}.rezero_alpha'].reshape(-1)[0]); C = arch['block_groups'][0]['rezero_alpha_init']
        W[f'b{i}.alpha_raw'] = a; W[f'b{i}.alpha_eff'] = C*np.tanh(a/C)
        W[f'b{i}.ln_gamma_absmax'] = float(np.abs(T[f'blocks.{i}.res_ln.weight']).max()); W[f'b{i}.ln_beta_absmax'] = float(np.abs(T[f'blocks.{i}.res_ln.bias']).max())
    out['_weights'] = W
    json.dump(out, open(os.path.join(HERE, 'net681_stats.json'), 'w'), indent=1)
elif stage == 'scan':
    def lsm(z): z = z-z.max(); return z-np.log(np.exp(z).sum())
    o64 = forward_batched(T, arch, X)
    p64 = softmax(o64['vl_raw']); ce64v = -np.log(p64[np.arange(N), labels])
    la64 = [lsm(o64['pl_raw'][i][legal[i]]) for i in range(N)]
    def metr(o, pol_round=None, val_round=None):
        vl = o['vl_raw'] if val_round is None else val_round(o['vl_raw']); pl = o['pl_raw'] if pol_round is None else pol_round(o['pl_raw'])
        p = softmax(vl) if val_round is None else bf16(softmax(vl))
        vkl = np.sum(p64*(np.log(p64)-np.log(np.maximum(p, 1e-30))), 1); cev = -np.log(np.maximum(p[np.arange(N), labels], 1e-30))
        kl = []; top1 = []; dce = []
        for i in range(N):
            li = legal[i]; lb = lsm(pl[i][li]); la = la64[i]; pa = np.exp(la)
            kl.append(float(np.sum(pa*(la-lb)))); top1.append(int(np.argmax(lb)) != int(np.argmax(la)) and lb.max() > lb[int(np.argmax(la))])
            t = li.tolist().index(targets[i]); dce.append(-lb[t]+la[t])
        kl = np.array(kl); dce = np.array(dce)
        return dict(v_kl_mean=float(vkl.mean()), v_kl_max=float(vkl.max()), v_dce_corpus=float((cev-ce64v)[corpus].mean()),
                    dv_abs_mean=float(np.abs((p[:, 0]-p[:, 2])-(p64[:, 0]-p64[:, 2])).mean()),
                    p_kl_mean=float(kl.mean()), p_kl_max=float(kl.max()), p_top1_changed=float(np.mean(top1)), p_dce_corpus=float(dce[corpus].mean()))
    out = {}
    runs = [('ALL internal points (heads unrounded)', frozenset(names) - {'p.conv', 'v.fc2mm'}, None),
            ('head outputs only (fused)', frozenset(), 'heads'),
            ('head matmul+bias rounded separately', frozenset({'p.conv', 'v.fc2mm'}), 'heads')]
    runs += [(n, frozenset({n}), None) for n in names if n not in ('p.conv', 'v.fc2mm')]
    for label, R, mode in runs:
        t = time.time(); o = forward_batched(T, arch, X, R)
        out[label] = metr(o, bf16, bf16) if mode == 'heads' else metr(o)
        print(f'{label:40s}', {k: f'{v:.3g}' for k, v in out[label].items()}, f'{time.time()-t:.0f}s', flush=True)
        json.dump(out, open(os.path.join(HERE, 'net681_scan.json'), 'w'), indent=1)
