import json,collections,datetime,os
r=json.load(open('scan2.json'))
by=collections.defaultdict(list)
for x in r: by[x['model_id']].append(x)
def sk(x):
    s=x['step']; s=int(s) if s not in (None,'') else -1
    return (s, float(x['created'] or 0), x['mtime'])
out=[]
for mid,L in by.items():
    L.sort(key=sk); x=L[-1]
    from archnorm import norm_arch; a=norm_arch(x['arch']) if x['arch'] else None
    if a:
        g=a['block_groups']
        tower='+'.join(f"{b['count']}x{b['channels']}c{b['conv1_kernel_size']}{'/'+str(b['conv2_kernel_size']) if b['conv2_kernel_size']!=b['conv1_kernel_size'] else ''}" for b in g)
        feats=set()
        for b in g:
            feats.add(f"{b['activation_style']},{b['activation_function']},se={b['se_style']},rz={b['use_rezero']},on={b.get('output_norm')},sm={b.get('skip_merge')}")
        desc=f"stem{a['stem_conv_kernel_size']} {tower} [{';'.join(sorted(feats))}] vh={a['value_head_conv_channels']}c/{a['value_head_hidden_units']}h skip={a['feature_skip_source']}"
        dtype=a['compute_data_type']; ph=a['policy_head_style']+(f"(K={a['policy_pre_conv_channels']})" if a['policy_head_style']!='simple_conv' else ''); vh=a['value_head_style']; enc=a.get('input_encoding'); af=a.get('activation_function')
    else: desc=dtype=ph=vh=enc='?'
    cr=datetime.datetime.fromtimestamp(float(x['created'])).strftime('%Y-%m-%d %H:%M') if x['created'] else '?'
    out.append(dict(mid=mid,n=len(L),step=x['step'],file=os.path.basename(x['path']),path=x['path'],dtype=dtype,ph=ph,vh=vh,enc=enc,desc=desc,created=cr,corpus=x['corpus'],parent=x['parent'],build=x['build'],notes=(x['notes'] or '')[:80],dtypes=x['dtypes']))
out.sort(key=lambda o:o['created'])
json.dump(out,open('latest.json','w'))
print(len(out))
for o in out: print(f"{o['mid']:22} n={o['n']:4} step={str(o['step']):>8} {o['created']} {o['dtype']:9} {o['ph']:24} {o['vh']:12} {o['enc']} corpus={o['corpus']} parent={o['parent']} | {o['desc']} | {o['file']} {o['dtypes']}")
