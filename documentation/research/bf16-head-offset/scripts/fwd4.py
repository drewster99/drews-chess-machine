"""Instrumented numpy forward of ChessNetwork for the Ejp0 (v5: pre-act,
SE scale_and_bias, tanh-ReZero, clean_add, per-block LayerNorm, tower-end BN,
intermediate_conv policy, WDL value) architecture, mirroring the MPSGraph op
order in ChessNetwork.swift (inference-mode BN, running stats).

Every tensor the real bf16 graph materialises as a separate MPSGraph op is a
named *rounding point*. `forward(..., R=...)` rounds to bf16 exactly at the
points in R (everything else stays float64), so the error each point
contributes can be isolated. R='ALL' rounds at every point (pessimistic per-op
emulation). Head outputs are returned UNROUNDED (`pl_raw`, `vl_raw`) together
with the rounded variants, so head-output fixes can be applied afterwards.

Normalisation points: `<bn>.sub` is the (x - mean) intermediate (only rounded
if listed — whether MPSGraph's fused `normalize` materialises it in bf16 is not
known); `<bn>` is the normalize output. LayerNorm's mean and variance are
separate graph ops (`graph.mean`, `graph.variance`) whose outputs are bf16
tensors in the real graph: points `<ln>.mean`, `<ln>.var`.
"""
import numpy as np, json, struct, math

def load(p):
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n)); data = f.read()
    md = h.pop('__metadata__'); T = {}
    for k, v in h.items():
        a, b = v['data_offsets']; assert v['dtype'] == 'F32'
        T[k] = np.frombuffer(data[a:b], dtype='<f4').reshape(v['shape']).astype(np.float64)
    return md, T

def bf16(x):
    a = np.asarray(x, dtype=np.float32).copy(); u = a.view(np.uint32)
    u2 = ((u + (((u >> 16) & 1) + 0x7FFF)) >> 16) << 16
    return u2.astype(np.uint32).view(np.float32).astype(np.float64)

def conv(x, w):
    N, C = x.shape[:2]; O, _, k, _ = w.shape; p = (k-1)//2
    if k == 1:
        return np.einsum('nchw,oc->nohw', x, w[:, :, 0, 0], optimize=True)
    xp = np.zeros((N, C, 8+2*p, 8+2*p)); xp[:, :, p:p+8, p:p+8] = x
    y = np.zeros((N, 8, 8, O))
    for i in range(k):
        for j in range(k):
            y += np.tensordot(xp[:, :, i:i+8, j:j+8], w[:, :, i, j], axes=([1], [1]))
    return y.transpose(0, 3, 1, 2)

class Ctx:
    """Rounding policy + optional stats capture."""
    def __init__(self, R, capture=False, q=None):
        self.R = R; self.capture = capture; self.acts = {}; self.q = q or bf16
    def __call__(self, name, x):
        if self.capture: self.acts[name] = x
        if self.R == 'ALL' or name in self.R: return self.q(x)
        return x

def s4(a): return a[None, :, None, None]

def bn(c, x, T, n, key):
    m = T[n+'.running_mean']; v = T[n+'.running_var']; g = T[n+'.weight']; b = T[n+'.bias']
    d = c(key+'.sub', x - s4(m))
    return c(key, d / np.sqrt(s4(v)+1e-5) * s4(g) + s4(b))

def ln(c, h, T, n, key):
    mu = c(key+'.mean', h.mean(1, keepdims=True))
    var = c(key+'.var', ((h - h.mean(1, keepdims=True))**2).mean(1, keepdims=True))
    d = c(key+'.sub', h - mu)
    return c(key, d / np.sqrt(var+1e-5) * s4(T[n+'.weight']) + s4(T[n+'.bias']))

def relu(x): return np.maximum(x, 0)

def forward(T, arch, x, R=frozenset(), capture=False, vfc2=None, q=None):
    """T: float64 weights (already bf16-exact for this model). Returns dict with
    pl_raw/vl_raw (head outputs before their final rounding), f1, and — if
    capture — every named pre-rounding activation."""
    c = Ctx(R, capture, q)
    groups = arch['block_groups']; assert len(groups) == 1
    g = groups[0]
    assert g['activation_style'] == 'pre' and g['skip_merge'] == 'clean_add' and g['se_style'] == 'scale_and_bias'
    assert g.get('output_norm') == 'layer_norm' and g['use_rezero'] and arch['policy_head_style'] == 'intermediate_conv'
    # ReLU everywhere, SE FC1 included. A file without `se_activation`
    # predates it (format v4 or older): its FC1 used the group's activation.
    assert arch['activation_function'] == 'relu' and g['activation_function'] == 'relu'
    assert g.get('se_activation', g['activation_function']) == 'relu', 'only a ReLU SE FC1 is modelled'
    x = c('input', x)
    h = c('stem.conv', conv(x, T['stem.conv.weight']))
    h = bn(c, h, T, 'stem.bn', 'stem.bn')
    for i in range(g['count']):
        pre = f'blocks.{i}.'; k = f'b{i}.'
        y = relu(bn(c, h, T, pre+'bn1', k+'bn1'))
        y = c(k+'conv1', conv(y, T[pre+'conv1.weight']))
        y = relu(bn(c, y, T, pre+'bn2', k+'bn2'))
        z = c(k+'conv2', conv(y, T[pre+'conv2.weight']))
        # SE scale_and_bias (graph stores fc weights [in,out]; safetensors [out,in])
        C = z.shape[1]; base = pre+'se_scalebias.'
        s = c(k+'se.pool', z.mean((2, 3)))
        s = c(k+'se.fc1mm', s @ T[base+'fc1.weight'].T)
        s = relu(c(k+'se.fc1', s + T[base+'fc1.bias']))
        s = c(k+'se.fc2mm', s @ T[base+'fc2.weight'].T)
        s = c(k+'se.fc2', s + T[base+'fc2.bias'])
        sig = c(k+'se.sig', 1/(1+np.exp(-s[:, :C])))
        zs = c(k+'se.scaled', z * sig[:, :, None, None])
        zo = c(k+'se.out', zs + s[:, C:][:, :, None, None])
        Cc = g['rezero_alpha_init']*1.0
        al = Cc*c(k+'alpha', np.array(math.tanh(float(T[pre+'rezero_alpha'].reshape(-1)[0])/Cc)))
        br = c(k+'rezero', zo * al)
        h = c(k+'add', h + br)
        h = ln(c, h, T, pre+'res_ln', k+'ln')
    h = relu(bn(c, h, T, 'tower_final_bn', 'tower.bn'))
    tower = h
    # policy
    p = c('p.pre_conv', conv(h, T['policy.pre_conv.weight']))
    feat = relu(bn(c, p, T, 'policy.pre_bn', 'p.pre_bn'))
    pm = c('p.conv', conv(feat, T['policy.conv.weight']))
    pl_raw = pm + s4(T['policy.conv.bias'])
    N = x.shape[0]
    # value
    v = c('v.conv', conv(tower, T['value.conv.weight']))
    v = relu(bn(c, v, T, 'value.bn', 'v.bn'))
    f = v.reshape(N, -1)
    f1 = c('v.fc1mm', f @ T['value.fc1.weight'].T)
    f1 = relu(c('v.fc1', f1 + T['value.fc1.bias']))
    W2 = T['value.wdl_fc2.weight'] if vfc2 is None else vfc2[0]
    b2 = T['value.wdl_fc2.bias'] if vfc2 is None else vfc2[1]
    vm = c('v.fc2mm', f1 @ W2.T)
    vl_raw = vm + b2
    out = dict(pl_raw=pl_raw.reshape(N, -1), vl_raw=vl_raw, f1=f1, feat=feat, pm=pm.reshape(N, -1), vm=vm)
    if capture: out['acts'] = c.acts
    return out

def f32(x): return np.asarray(x, dtype=np.float32).astype(np.float64)

def forward_batched(T, arch, X, R=frozenset(), bs=256, vfc2=None, q=None, keep=('pl_raw', 'vl_raw', 'f1', 'pm', 'vm')):
    outs = [forward(T, arch, X[i:i+bs], R, vfc2=vfc2, q=q) for i in range(0, len(X), bs)]
    return {k: np.concatenate([o[k] for o in outs]) for k in keep}

def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e/e.sum(-1, keepdims=True)
