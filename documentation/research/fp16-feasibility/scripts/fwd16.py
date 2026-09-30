"""Instrumented numpy forward of ChessNetwork for fp16 / bf16 / fp32 rounding
studies. Generalises bf16-head-offset/scripts/fwd4.py (the v5 forward that was
calibrated against the Lichess bot's recorded bf16 outputs) to the older
post-activation / attenuate-SE / activation-gated / simple_conv architecture of
the fp32 KbHZ line, keeping fwd4's op order and rounding-point names for v5.

Every tensor the real MPSGraph graph materialises as a separate op is a named
rounding point. `forward(T, arch, x, R, q)` rounds with quantiser `q` exactly
at the points in R (R='ALL' = every point; everything else float64). Weights are
passed in already quantised by the caller (see `quantise_weights`). Head outputs
are returned unrounded (`pl_raw`, `vl_raw`) so head-tail options can be applied
afterwards.

With capture=True the forward also records, per point, the float64 value
before rounding and, for every conv / matmul, the L1 accumulation bound
sum_k |x_k| |w_k| (+|bias|): an upper bound on the magnitude of ANY partial sum
in ANY accumulation order, which is what decides whether an fp16 accumulator
could overflow inside the reduction even when the final output is in range.
"""
import numpy as np, json, struct, math, os

# Position set built by ../../bf16-head-offset/scripts/posset.py (900 corpus
# positions from shard 45 of corpus w3aA5b + every position of the first 4
# Lichess bot games). Override with the DCM_POSSET environment variable.
POSSET = os.environ.get('DCM_POSSET', 'posset.pkl')

F16_MAX = 65504.0
F16_MIN_NORMAL = 2.0 ** -14          # 6.103515625e-05
F16_MIN_SUB = 2.0 ** -24             # 5.960464477539063e-08

def load(p):
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n)); data = f.read()
    md = h.pop('__metadata__'); T = {}
    for k, v in h.items():
        a, b = v['data_offsets']; assert v['dtype'] == 'F32', (k, v['dtype'])
        T[k] = np.frombuffer(data[a:b], dtype='<f4').reshape(v['shape']).astype(np.float64)
    return md, T

def bf16(x):
    a = np.asarray(x, dtype=np.float32).copy(); u = a.view(np.uint32)
    u2 = ((u + (((u >> 16) & 1) + 0x7FFF)) >> 16) << 16
    return u2.astype(np.uint32).view(np.float32).astype(np.float64)

def f16(x):
    with np.errstate(over='ignore'):
        return np.asarray(x, dtype=np.float16).astype(np.float64)

def f16_ftz(x):
    """fp16 with subnormals flushed to zero (what the trainer's own comment
    says MPS does for fp16 denormals; unverified)."""
    y = f16(x); return np.where(np.abs(y) < F16_MIN_NORMAL, 0.0, y)

def f32(x): return np.asarray(x, dtype=np.float32).astype(np.float64)
def ident(x): return x
Q = {'f64': ident, 'f32': f32, 'bf16': bf16, 'f16': f16, 'f16ftz': f16_ftz}

def conv(x, w):
    N, C = x.shape[:2]; O, _, k, _ = w.shape; p = (k - 1) // 2
    if k == 1:
        return np.einsum('nchw,oc->nohw', x, w[:, :, 0, 0], optimize=True)
    xp = np.zeros((N, C, 8 + 2 * p, 8 + 2 * p)); xp[:, :, p:p + 8, p:p + 8] = x
    y = np.zeros((N, 8, 8, O))
    for i in range(k):
        for j in range(k):
            y += np.tensordot(xp[:, :, i:i + 8, j:j + 8], w[:, :, i, j], axes=([1], [1]))
    return y.transpose(0, 3, 1, 2)

class Ctx:
    def __init__(self, R, q, capture):
        self.R = R; self.q = q; self.capture = capture; self.acts = {}; self.acc = {}; self.norm_in = {}
    def __call__(self, name, x):
        if self.capture: self.acts[name] = x
        if self.R == 'ALL' or name in self.R: return self.q(x)
        return x
    def bound(self, name, b):
        if self.capture: self.acc[name] = float(np.max(b))

def s4(a): return a[None, :, None, None]

def bn(c, x, T, n, key):
    m = T[n + '.running_mean']; v = T[n + '.running_var']; g = T[n + '.weight']; b = T[n + '.bias']
    if c.capture: c.norm_in[key] = x
    d = c(key + '.sub', x - s4(m))
    return c(key, d / np.sqrt(s4(v) + 1e-5) * s4(g) + s4(b))

def ln(c, h, T, n, key):
    if c.capture: c.norm_in[key] = h
    mu = c(key + '.mean', h.mean(1, keepdims=True))
    var = c(key + '.var', ((h - h.mean(1, keepdims=True)) ** 2).mean(1, keepdims=True))
    d = c(key + '.sub', h - mu)
    return c(key, d / np.sqrt(var + 1e-5) * s4(T[n + '.weight']) + s4(T[n + '.bias']))

def relu(x): return np.maximum(x, 0)

def cconv(c, name, x, w):
    if c.capture: c.bound(name, conv(np.abs(x), np.abs(w)))
    return c(name, conv(x, w))

def cmm(c, name, x, W, b=None):
    """x @ W.T (+ b) with the matmul rounded at `name`+'mm' and the bias add at `name`."""
    if c.capture: c.bound(name + 'mm', np.abs(x) @ np.abs(W).T + (0 if b is None else np.abs(b)))
    y = c(name + 'mm', x @ W.T)
    return y if b is None else c(name, y + b)

def forward(T, arch, x, R=frozenset(), q=ident, capture=False):
    c = Ctx(R, q, capture)
    groups = arch['block_groups']; assert len(groups) == 1; g = groups[0]
    style = g['activation_style']
    x = c('input', x)
    h = cconv(c, 'stem.conv', x, T['stem.conv.weight'])
    h = bn(c, h, T, 'stem.bn', 'stem.bn')
    if style == 'post': h = relu(h)
    for i in range(g['count']):
        pre = f'blocks.{i}.'; k = f'b{i}.'
        if style == 'pre':
            y = relu(bn(c, h, T, pre + 'bn1', k + 'bn1'))
            y = cconv(c, k + 'conv1', y, T[pre + 'conv1.weight'])
            y = relu(bn(c, y, T, pre + 'bn2', k + 'bn2'))
            z = cconv(c, k + 'conv2', y, T[pre + 'conv2.weight'])
        else:
            y = cconv(c, k + 'conv1', h, T[pre + 'conv1.weight'])
            y = relu(bn(c, y, T, pre + 'bn1', k + 'bn1'))
            y = cconv(c, k + 'conv2', y, T[pre + 'conv2.weight'])
            z = bn(c, y, T, pre + 'bn2', k + 'bn2')
        C = z.shape[1]
        if g['se_style'] == 'scale_and_bias':
            base = pre + 'se_scalebias.'
            s = c(k + 'se.pool', z.mean((2, 3)))
            s = relu(cmm(c, k + 'se.fc1', s, T[base + 'fc1.weight'], T[base + 'fc1.bias']))
            s = cmm(c, k + 'se.fc2', s, T[base + 'fc2.weight'], T[base + 'fc2.bias'])
            sig = c(k + 'se.sig', 1 / (1 + np.exp(-s[:, :C])))
            zs = c(k + 'se.scaled', z * sig[:, :, None, None])
            z = c(k + 'se.out', zs + s[:, C:][:, :, None, None])
        elif g['se_style'] == 'attenuate_only':
            base = pre + 'se_attenuate.'
            s = c(k + 'se.pool', z.mean((2, 3)))
            s = relu(cmm(c, k + 'se.fc1', s, T[base + 'fc1.weight'], T[base + 'fc1.bias']))
            s = cmm(c, k + 'se.fc2', s, T[base + 'fc2.weight'], T[base + 'fc2.bias'])
            sig = c(k + 'se.sig', 1 / (1 + np.exp(-s)))
            z = c(k + 'se.out', z * sig[:, :, None, None])
        else:
            assert g['se_style'] == 'none'
        if g['use_rezero']:
            Cc = g['rezero_alpha_init'] * 1.0
            al = Cc * c(k + 'alpha', np.array(math.tanh(float(T[pre + 'rezero_alpha'].reshape(-1)[0]) / Cc)))
            z = c(k + 'rezero', z * al)
        h = c(k + 'add', h + z)
        if g['skip_merge'] == 'activation_gated': h = relu(h)
        else: assert g['skip_merge'] == 'clean_add'
        if g.get('output_norm') == 'layer_norm':
            h = ln(c, h, T, pre + 'res_ln', k + 'ln')
    if style == 'pre':
        h = relu(bn(c, h, T, 'tower_final_bn', 'tower.bn'))
    tower = h; N = x.shape[0]
    ps = arch['policy_head_style']
    if ps == 'simple_conv':
        feat = tower
    else:
        assert ps == 'intermediate_conv'
        p = cconv(c, 'p.pre_conv', tower, T['policy.pre_conv.weight'])
        feat = relu(bn(c, p, T, 'policy.pre_bn', 'p.pre_bn'))
    W = T['policy.conv.weight']
    if c.capture: c.bound('p.conv', conv(np.abs(feat), np.abs(W)) + s4(np.abs(T['policy.conv.bias'])))
    pm = c('p.convmm', conv(feat, W))
    pl_raw = pm + s4(T['policy.conv.bias'])
    v = cconv(c, 'v.conv', tower, T['value.conv.weight'])
    v = relu(bn(c, v, T, 'value.bn', 'v.bn'))
    f = v.reshape(N, -1)
    f1 = relu(cmm(c, 'v.fc1', f, T['value.fc1.weight'], T['value.fc1.bias']))
    W2 = T['value.wdl_fc2.weight']; b2 = T['value.wdl_fc2.bias']
    if c.capture: c.bound('v.fc2', np.abs(f1) @ np.abs(W2).T + np.abs(b2))
    vm = c('v.fc2mm', f1 @ W2.T)
    vl_raw = vm + b2
    out = dict(pl_raw=pl_raw.reshape(N, -1), vl_raw=vl_raw, f1=f1, feat=feat, tower=tower)
    if capture: out['acts'] = c.acts; out['acc'] = c.acc; out['norm_in'] = c.norm_in
    return out

def quantise_weights(T, q):
    return {k: q(v) for k, v in T.items() if not k.startswith('opt.')}

def forward_batched(T, arch, X, R=frozenset(), q=ident, bs=128, keep=('pl_raw', 'vl_raw', 'f1')):
    outs = [forward(T, arch, X[i:i + bs], R, q) for i in range(0, len(X), bs)]
    return {k: np.concatenate([o[k] for o in outs]) for k in keep}

def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

def norm_arch(s):
    a = json.loads(s) if isinstance(s, str) else dict(s)
    if 'block_groups' not in a:
        a['block_groups'] = [dict(count=a['num_blocks'], channels=a['channels'], conv1_kernel_size=a['block_conv1_kernel_size'],
            conv2_kernel_size=a['block_conv2_kernel_size'], se_style=a['block_se_style'], se_reduction_ratio=a['block_se_reduction_ratio'],
            use_rezero=a['block_use_rezero'], rezero_alpha_init=a['rezero_alpha_init'], activation_function=a['activation_function'],
            activation_style=a['block_activation_style'], skip_merge=a['block_skip_merge'], dropout_multiplier=1)]
    # `se_beta_init` (format v4+) only changes the random init; the stored
    # weights already carry it, so the forward ignores it. Legacy files omit it.
    for g in a['block_groups']:
        g.setdefault('se_beta_init', 'glorot')
    a.setdefault('feature_skip_source', 'none')
    return a
