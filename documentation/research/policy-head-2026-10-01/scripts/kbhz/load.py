import json, struct, hashlib, numpy as np, os
ROOT=os.path.expanduser("~/Library/Application Support/DrewsChessMachine/")
def read_st(path):
    with open(ROOT+path,"rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]; h=json.loads(f.read(n)); start=8+n
        meta=h.pop("__metadata__",{}); out={}
        for k,v in h.items():
            b,e=v["data_offsets"]; f.seek(start+b); raw=f.read(e-b)
            assert v["dtype"]=="F32"
            out[k]=np.frombuffer(raw,"<f4").astype(np.float64).reshape(v["shape"])
    return meta,out
def block_names(nblocks=8):
    names=["stem.conv.weight","stem.bn.weight","stem.bn.bias"]
    for i in range(nblocks):
        p=f"blocks.{i}."
        names+= [p+"conv1.weight",p+"bn1.weight",p+"bn1.bias",p+"conv2.weight",p+"bn2.weight",p+"bn2.bias",
                 p+"se_attenuate.fc1.weight",p+"se_attenuate.fc1.bias",p+"se_attenuate.fc2.weight",p+"se_attenuate.fc2.bias"]
    names+=["policy.conv.weight","policy.conv.bias","value.conv.weight","value.bn.weight","value.bn.bias","value.fc1.weight","value.fc1.bias","value.wdl_fc2.weight","value.wdl_fc2.bias"]
    return names
def read_dcm(path):
    data=open(ROOT+path,"rb").read()
    assert hashlib.sha256(data[:-32]).digest()==data[-32:]
    off=8; ver,ah,tc=struct.unpack_from("<III",data,off); off+=12+8
    il,=struct.unpack_from("<I",data,off); off+=4; mid=data[off:off+il].decode(); off+=il
    ml,=struct.unpack_from("<I",data,off); off+=4; meta=json.loads(data[off:off+ml]); off+=ml
    arrs=[]
    for i in range(tc):
        idx,cnt=struct.unpack_from("<II",data,off); off+=8
        arrs.append(np.frombuffer(data,"<f4",count=cnt,offset=off).astype(np.float64)); off+=4*cnt
    names=block_names(); nt=len(names); assert nt==92
    out={n:arrs[i] for i,n in enumerate(names)}
    nrun=36
    if tc==2*nt+nrun:
        for i,n in enumerate(names): out["opt."+n+".velocity"]=arrs[nt+nrun+i]
    meta["model_id"]=mid; meta["tensor_count"]=tc; meta["version"]=ver; meta["arch_hash"]=hex(ah)
    return meta,out
