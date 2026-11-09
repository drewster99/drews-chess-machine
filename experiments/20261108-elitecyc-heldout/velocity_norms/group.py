import csv,collections,math
def grp(t):
    if t.startswith('stem.conv'): return 'stem.conv'
    if t.startswith('blocks.'):
        b=t.split('.')[1]
        if '.conv' in t: return f'b{b}.convs'
        if 'res_ln' in t: return 'tower.res_ln'
        return 'tower.bn'
    if t.startswith('stem.bn') or t.startswith('tower_final_bn'): return 'tower.bn'
    if t.startswith('policy.'): return 'policy.'+t.split('.')[1]
    if t.startswith('value.'):
        return 'value.'+t.split('.')[1]
    return t
W=collections.defaultdict(lambda: collections.defaultdict(float)); V=collections.defaultdict(lambda: collections.defaultdict(float))
for r in csv.DictReader(open('vnorm.csv')):
    s=int(r['step']); g=grp(r['tensor'])
    W[s][g]+=float(r['w_norm'])**2
    if r['v_norm']: V[s][g]+=float(r['v_norm'])**2
steps=[10000,50000,150000,250000,380000,450000,500000,600000,700000,790000,800000,845000]
groups=sorted(V[845000])
print('‖v‖ per group (momentum buffer ≈ smoothed gradient sum)')
print(f'{"group":16}'+''.join(f'{s//1000:>8}k' for s in steps))
for g in groups+['TOTAL']:
    print(f'{g:16}'+''.join(f'{math.sqrt(sum(V[s].values()) if g=="TOTAL" else V[s][g]):9.3f}' for s in steps))
print('\nshare of ‖v‖² (%)')
for g in groups:
    print(f'{g:16}'+''.join(f'{100*V[s][g]/sum(V[s].values()):9.1f}' for s in steps))
print('\n‖w‖ per group')
for g in groups:
    print(f'{g:16}'+''.join(f'{math.sqrt(W[s][g]):9.2f}' for s in steps))
print('\n‖v‖/‖w‖ per group (x1000)')
for g in groups:
    print(f'{g:16}'+''.join(f'{1000*math.sqrt(V[s][g]/W[s][g]):9.3f}' for s in steps))
