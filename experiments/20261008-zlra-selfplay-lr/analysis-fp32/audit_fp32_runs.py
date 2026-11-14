import json,collections,sys
LIMIT=2**31-1
runs=collections.OrderedDict(); nolineage=[]; dtypes=collections.Counter()
for line in open(sys.argv[1]):
    o=json.loads(line); d=o.get('description'); p=o.get('path','')
    if not d: continue
    a=d['architecture']; dt=a['compute_data_type']; dtypes[dt]+=1
    if dt!='float32': continue
    planes=d['input_planes']; ks=a['stem_conv_kernel_size']
    convs=[('stem',planes,ks)]
    prev=d['expanded_blocks'][0]['channels'] if d['expanded_blocks'] else 0
    prev=None
    stemC=a['block_groups'][0]['channels']
    prev=stemC
    for i,b in enumerate(d['expanded_blocks']):
        convs.append((f'b{i}.conv1',prev,b['conv1_kernel_size'])); convs.append((f'b{i}.conv2',b['channels'],b['conv2_kernel_size'])); prev=b['channels']
    rec=(d.get('lineage') or {}).get('record')
    batch=None; run='(no lineage)'; kind=None; step=d.get('step_reading',{}).get('trainer_step')
    if rec:
        run=rec['run']['lineage_run_id']; kind=rec['invocation'].get('path_kind')
        try: batch=json.loads(rec['parameters']['snapshot_json']).get('training_batch_size')
        except Exception: pass
    worst=max(convs,key=lambda c:c[1]*c[2]*c[2])
    per=worst[1]*worst[2]**2*64
    r=runs.setdefault(run,dict(kind=kind,batches=set(),files=0,steps=[],worst=worst,arch=f"{planes}pl stem{ks} "+ ' '.join(f"{b['count']}x{b['channels']}ch {b['conv1_kernel_size']}/{b['conv2_kernel_size']}" for b in a['block_groups']),example=p.split('/')[-1] if 'Sessions' not in p else '/'.join(p.split('/')[-2:])))
    r['files']+=1; r['batches'].add(batch)
    if isinstance(step,int): r['steps'].append(step)
    r['threshold_batch']=LIMIT//per+1
print('dtypes:',dict(dtypes))
for run,r in runs.items():
    bs=[b for b in r['batches'] if b]
    aff = any(b*r['worst'][1]*r['worst'][2]**2*64>LIMIT for b in bs) if bs else None
    st=(min(r['steps']),max(r['steps'])) if r['steps'] else None
    print(f"{'AFFECTED' if aff else ('unknown-batch' if aff is None else 'ok')} run={run[:8]} kind={r['kind']} files={r['files']} batch={sorted(r['batches'],key=str)} steps={st} arch=[{r['arch']}] worst={r['worst']} breaks_at_batch>={r['threshold_batch']} e.g. {r['example']}")
