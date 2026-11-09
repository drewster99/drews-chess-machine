import json,struct,glob,re,sys,os
import numpy as np
D=os.path.expanduser('~/Library/Application Support/DrewsChessMachine/Models/')
files=sorted(glob.glob(D+'20261011-zlra-elitecyc-latest-step*.safetensors'),key=lambda f:int(re.search(r'step(\d+)',f).group(1)))
out=open(sys.argv[1],'w'); out.write('step,tensor,n,w_norm,v_norm,lr,mom\n')
for f in files:
    s=int(re.search(r'step(\d+)',f).group(1))
    with open(f,'rb') as h:
        n=struct.unpack('<Q',h.read(8))[0]; hdr=json.loads(h.read(n)); base=8+n
        md=hdr.pop('__metadata__',{})
        lr=mom=''
        try:
            lin=json.loads(md['dcm_lineage']); sc=lin.get('configuration',{}).get('schedule_at_save') or {}
            lr=sc.get('learning_rate',sc.get('lr','')); mom=sc.get('momentum','')
            if s==10000: print('schedule_at_save keys:',list(sc.keys()),file=sys.stderr)
        except Exception as e: print(s,'lineage',e,file=sys.stderr)
        def arr(k):
            a,b=hdr[k]['data_offsets']; h.seek(base+a); return np.frombuffer(h.read(b-a),dtype='<f4').astype(np.float64)
        for k in sorted(hdr):
            if k.startswith('opt.') or 'running_' in k: continue
            vk='opt.'+k+'.velocity'
            w=arr(k); v=arr(vk) if vk in hdr else None
            out.write(f'{s},{k},{w.size},{np.sqrt((w*w).sum()):.6g},{"" if v is None else f"{np.sqrt((v*v).sum()):.6g}"},{lr},{mom}\n')
    out.flush()
