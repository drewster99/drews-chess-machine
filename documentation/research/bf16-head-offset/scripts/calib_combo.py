import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
exec(open(os.path.join(HERE,'calib_scan.py')).read().split('out = {}')[0])
N=set(names)
combos = {
 'fused heads + p.pre_bn': {'p.pre_bn'},
 'fused heads + all conv/norm/add outputs': {n for n in N if n.endswith(('conv','conv1','conv2','bn','bn1','bn2','.ln','add','pre_conv','pre_bn')) and n not in ('p.conv',)},
 'per-op everything except head mm (fused heads)': N - {'p.conv','v.fc2mm'},
 'per-op except head mm and norm .sub': {n for n in N if not n.endswith('.sub')} - {'p.conv','v.fc2mm'},
}
for k,R in combos.items():
    print(f'{k:50s}', score(frozenset(R)), flush=True)
