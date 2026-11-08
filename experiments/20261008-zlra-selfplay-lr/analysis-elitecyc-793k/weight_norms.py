import json,struct,sys,numpy as np
D='/Users/andrew/Library/Application Support/DrewsChessMachine/Models/20261011-zlra-elitecyc-latest-step%d.safetensors'
G={'conv1':lambda k:'conv1.weight'==k[-12:] and k.startswith('blocks'),'conv2':lambda k:k.endswith('conv2.weight') and k.startswith('blocks'),
 'stem':lambda k:k=='stem.conv.weight','bn_gamma':lambda k:k.endswith('weight') and ('bn' in k) ,'res_ln_g':lambda k:'res_ln.weight' in k,
 'pol_heads':lambda k:k.startswith('policy.') and k.endswith('weight') and 'bn' not in k,'val_heads':lambda k:k.startswith('value.') and k.endswith('weight') and 'bn' not in k}
steps=[int(s) for s in sys.argv[1:]]
print('step', *[f'{g}:|W| |v|' for g in G])
for s in steps:
  f=open(D%s,'rb');n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));base=8+n
  def t(k):
    m=h[k];a,b=m['data_offsets'];f.seek(base+a);return np.frombuffer(f.read(b-a),dtype=np.float32).astype(np.float64)
  row=[str(s)]
  for g,fn in G.items():
    ks=[k for k in h if k!='__metadata__' and not k.startswith('opt.') and fn(k)]
    w=np.sqrt(sum((t(k)**2).sum() for k in ks)); v=np.sqrt(sum((t('opt.'+k+'.velocity')**2).sum() for k in ks))
    row.append(f'{w:.2f} {v:.3g}')
  print(*row,sep=' | ')
