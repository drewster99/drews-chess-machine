"""Head-weight structure along the Ejp0 replay line and its predecessors
(Qeu8 seed -> GLu5 -> Lnji -> PVZp -> Ejp0), every checkpoint, identified by
safetensors __metadata__ (model_id + training_step), reading only the head
tensors."""
import os, glob, json, struct, numpy as np, csv
D = os.path.expanduser('~/Library/Application Support/DrewsChessMachine/Models')
LINE = ['20260702-7-Qeu8', '20260702-9-GLu5', '20260703-1-Lnji', '20260706-1-PVZp', '20260727-1-Ejp0']
WANT = ['value.wdl_fc2.weight', 'value.wdl_fc2.bias', 'policy.conv.weight', 'policy.conv.bias', 'value.fc1.bias',
        'blocks.0.rezero_alpha', 'blocks.1.rezero_alpha']
rows = []; seen = set()
for p in sorted(glob.glob(D+'/*.safetensors')):
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n)); base = 8+n
        md = h.get('__metadata__', {})
        if md.get('model_id') not in LINE: continue
        key = (md['model_id'], int(md.get('training_step') or 0))
        if key in seen: continue   # -latest duplicates
        seen.add(key)
        T = {}
        for k in WANT:
            if k not in h: continue
            a, b = h[k]['data_offsets']; f.seek(base+a)
            T[k] = np.frombuffer(f.read(b-a), dtype='<f4').reshape(h[k]['shape']).astype(np.float64)
    W = T['value.wdl_fc2.weight']; b = T['value.wdl_fc2.bias']; m = W.mean(0)
    Wp = T['policy.conv.weight'].reshape(76, -1); bp = T['policy.conv.bias']; mp = Wp.mean(0)
    rows.append(dict(model_id=md['model_id'], step=int(md.get('training_step') or 0), file=os.path.basename(p), created=md.get('created_at_unix'),
        v_meanrow_norm=float(np.linalg.norm(m)), v_resid_norm_max=float(np.linalg.norm(W-m, axis=1).max()),
        v_bias_mean=float(b.mean()), v_bias_w=float(b[0]), v_bias_d=float(b[1]), v_bias_l=float(b[2]),
        p_meanrow_norm=float(np.linalg.norm(mp)), p_resid_norm_med=float(np.median(np.linalg.norm(Wp-mp, axis=1))),
        p_bias_mean=float(bp.mean()), fc1_bias_mean=float(T['value.fc1.bias'].mean()),
        alpha0=float(T['blocks.0.rezero_alpha'].reshape(-1)[0]), alpha1=float(T['blocks.1.rezero_alpha'].reshape(-1)[0])))
order = {m: i for i, m in enumerate(LINE)}
rows.sort(key=lambda r: (order[r['model_id']], r['step']))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trend_struct.csv')
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(len(rows), 'rows ->', out)
