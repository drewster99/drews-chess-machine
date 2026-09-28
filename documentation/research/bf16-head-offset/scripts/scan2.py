import os,json,struct,glob,sys
D=os.path.expanduser('~/Library/Application Support/DrewsChessMachine/Models')
rows=[]
for p in glob.glob(D+'/**/*.safetensors',recursive=True):
    try:
        with open(p,'rb') as f:
            n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n))
    except Exception as e:
        print('ERR',p,e,file=sys.stderr); continue
    md=h.pop('__metadata__',{})
    dt=sorted(set(v['dtype'] for v in h.values()))
    rows.append(dict(path=p,model_id=md.get('model_id'),step=md.get('training_step'),parent=md.get('parent_model_id'),created=md.get('created_at_unix'),creator=md.get('creator'),notes=md.get('notes',''),arch=md.get('architecture'),corpus=md.get('replay_corpus_id'),build=md.get('built_by_build'),mtime=os.path.getmtime(p),dtypes=dt,keys=sorted(h.keys())))
json.dump(rows,open('scan2.json','w'))
print(len(rows))
