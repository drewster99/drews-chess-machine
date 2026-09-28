import json,datetime
L={o['path']:o for o in json.load(open('latest.json'))}
rows=[json.loads(l) for l in open('all_out.jsonl')]
def cls(e):
    d=e['ceq']-e['ce64']
    if d<0.005 and e['tie_frac']<0.05: return 'fine'
    if d<0.03: return 'degraded'
    return 'BAD'
print(f"{'model_id':22} {'src':10} {'step':>8} {'nat':4} {'bfx':4} {'ph':5} | {'vMeanRow':>8} {'vResid(W/D/L)':>17} {'vBias mean':>9} | {'shared med':>10} {'|lg|max':>7} {'spread':>6} | {'CE64':>6} {'CEbf':>6} {'dCE':>6} {'KL':>7} {'ties':>5} {'argmx':>5} | {'recCE':>6} {'recTie':>6} | cls  || pol: {'rowM':>5} {'resMed':>6} {'bMean':>6} | {'legMean':>7} {'legStd':>6} {'legMax':>6} | {'KL':>8} {'top1':>5} {'tie2':>5} {'tie5':>5} {'dCE':>7} | orKL orTie2 orTie5")
for r in rows:
    if 'error' in r: print('ERROR',r['tag'],r['path'].split('/')[-1],r['error'][:120]); continue
    nat=r['native']; k=f'{nat}-out' if f'{nat}-out' in r['emu'] else 'bf16-out'
    e=r['emu'][k]['value']; rc=r['emu'][f'{nat}-out-vrecentered']['value']; p=r['emu'][k]['policy']; o=r['emu'][f'{nat}-out-poracle']['policy']
    vs=r['vstruct']; ps=r['pstruct']; vl=r['value_logit']; pm=r['policy_mag']
    src='model' if r['tag']=='models' else r['tag'][8:24]
    print(f"{r['model_id']:22} {src:10} {str(r['step']):>8} {nat:4} {r['bf16_exact_weight_frac']:.2f} {r['policy_style'][:5]:5} | {vs['mean_row_norm']:8.2f} {'/'.join(f'{x:.2f}' for x in vs['resid_norms']):>17} {vs['bias_mean']:+9.3f} | {vl['shared_med']:+10.1f} {vl['maxabs']:7.1f} {vl['spread_med']:6.2f} | {e['ce64']:.4f} {e['ceq']:.4f} {e['ceq']-e['ce64']:+.4f} {e['kl_mean']:.4f} {e['tie_frac']:.3f} {e['argmax_change']:.3f} | {rc['ceq']:.4f} {rc['tie_frac']:.4f} | {cls(e):4} || {ps['mean_row_norm']:5.2f} {ps['resid_norm_med']:6.2f} {ps['bias_mean']:+6.3f} | {pm['legal_mean_med']:+7.2f} {pm['legal_std_med']:6.2f} {pm['legal_max_med']:6.2f} | {p['kl_mean']:.2e} {p['top1_lost']:.3f} {p['tie_top2']:.3f} {p['tie_top5']:.3f} {p['ceq']-p['ce64']:+.4f} | {o['kl_mean']:.1e} {o['tie_top2']:.3f} {o['tie_top5']:.3f}")
