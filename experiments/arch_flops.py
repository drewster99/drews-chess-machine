"""Per-position and per-step FLOPs for the R1-R16 start nets, from each file's own
architecture metadata and tensor shapes, following ChessNetwork.swift's graph."""
import json, os, struct
M = os.path.expanduser("~/Library/Application Support/DrewsChessMachine/Models")
NETS = [("R1 R10 R12", "20260929-test_SE_scale+bias-fresh"),
        ("R2 R11", "20260929-test_SE_scale+bias-seed2-fresh"),
        ("R3", "20260929-test_SE_attenuate-only-fresh"),
        ("R4", "20261001-test_SE_scale+bias-fc1leaky-fresh"),
        ("R5", "20260929-test_SE_none-fresh"), ("R6", "20260929-test_SE_none-seed2-fresh"),
        ("R7", "20261002-bench_v5s3_noSE_noReZero-fresh"),
        ("R8", "20261002-bench_v5s3_noSE_noReZero-seed2-fresh"),
        ("R9", "20260929-test_SE_none-rz0cap1-fresh"),
        ("R13 fatty", "20261003-fatty216-b2275-fresh"), ("R14 skinny", "20261003-skinny48-b2275-fresh"),
        ("R15 slim-neck fatty", "20261003-fatty224s3-b2275-fresh"),
        ("R16 fatconv", "20261004-fatconv98-b2275-fresh")]
BATCH = 4096
SQ = 64
# elementwise FLOPs per element, forward (training-mode BN/LN: batch stats + normalize)
NORM, ACT, ADD, MUL = 8, 1, 1, 1

def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    md = h.pop("__metadata__")
    return json.loads(md["architecture"]), {k: v["shape"] for k, v in h.items()}

def valid_taps(k):
    r = k // 2
    per_dim = sum(sum(1 for d in range(-r, r + 1) if 0 <= p + d < 8) for p in range(8))
    return (per_dim / 8) ** 2  # mean valid taps per output square

def count(arch, shapes):
    mac = useful = 0
    for name, s in shapes.items():
        if not name.endswith("weight") or len(s) not in (2, 4):
            continue
        if len(s) == 4:
            o, i, kh, kw = s
            mac += o * i * kh * kw * SQ
            useful += o * i * valid_taps(kh) * SQ
        else:
            mac += s[0] * s[1]; useful += s[0] * s[1]
    params = sum(eval("*".join(map(str, s))) for n, s in shapes.items() if "running_" not in n)
    ew = 0; act_elems = 0
    g = arch["block_groups"]
    pre = g[0]["activation_style"] == "pre"
    c0 = g[0]["channels"]
    ew += NORM * c0 * SQ + (0 if pre else ACT * c0 * SQ); act_elems += 2 * c0 * SQ
    cin = c0
    for grp in g:
        c = grp["channels"]; e = c * SQ
        for _ in range(grp["count"]):
            # bn1+act, conv1, bn2+act, conv2
            ew += (NORM + ACT) * cin * SQ + (NORM + ACT) * e; act_elems += 2 * cin * SQ + 4 * e
            se = grp["se_style"]
            if se != "none":
                ew += e + MUL * e + (ADD * e if se == "scale_and_bias" else 0) + 4 * c
                act_elems += e if se == "attenuate_only" else 2 * e
            if grp["use_rezero"]:
                ew += MUL * e; act_elems += e
            ew += ADD * e; act_elems += e
            if grp.get("output_norm") == "layer_norm":
                ew += NORM * e; act_elems += e
            cin = c
    if pre:
        ew += (NORM + ACT) * cin * SQ; act_elems += 2 * cin * SQ
    pk = arch["policy_pre_conv_channels"]
    ew += (NORM + ACT) * pk * SQ + 76 * SQ + 4864 * 5; act_elems += 2 * pk * SQ + 76 * SQ
    vc = arch["value_head_conv_channels"]; vh = arch["value_head_hidden_units"]
    ew += (NORM + ACT) * vc * SQ + 2 * vh + 3 + 3 * 5; act_elems += 2 * vc * SQ
    return params, mac, useful, ew, act_elems

rows = []
for runs, net in NETS:
    arch, shapes = header(os.path.join(M, net + ".safetensors"))
    g = arch["block_groups"][0]
    desc = f'{g["count"]}x{g["conv1_kernel_size"]}x{g["conv1_kernel_size"]}@{g["channels"]} SE={g["se_style"]} ReZero={g["use_rezero"]}'
    rows.append((runs, desc) + count(arch, shapes))
base = [r for r in rows if r[0] == "R7"][0]
fwd = lambda r: 2 * r[3] + r[5]
print("| runs | tower | params | conv/FC MACs/pos (M) | useful MACs/pos (M) | elementwise FLOPs/pos (M) | fwd GFLOP/pos | train TFLOP/step | vs R7/R8 | activation elems/pos (k) | vs R7/R8 |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    f = fwd(r); step = 3 * f * BATCH + 8 * r[2]
    bstep = 3 * fwd(base) * BATCH + 8 * base[2]
    print(f"| {r[0]} | {r[1]} | {r[2]:,} | {r[3]/1e6:.1f} | {r[4]/1e6:.1f} | {r[5]/1e6:.2f} | {f/1e9:.4f} | {step/1e12:.3f} | {step/bstep:.3f} | {r[6]/1e3:.0f} | {r[6]/base[6]:.2f} |")
