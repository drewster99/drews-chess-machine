#!/usr/bin/env python3
"""Per-site batch-norm health for the LR-schedule arms, for every activation function.

`[LAYER-HEALTH]` classifies a BN channel as dead / mostly off / always on from β/|γ|, which
only means something for ReLU and leaky ReLU, so it prints n/a for SiLU and GELU sites.
This script reads the enumerated checkpoints directly and reports, for every BN site that
feeds an activation, two activation-aware numbers alongside β/|γ|:

- **pass-through**: the expected magnitude of the activation's derivative, E|f'(γz + β)|,
  with z ~ N(0, 1) standing for the BN-normalized input. It is the fraction of the
  incoming gradient the channel passes back on average: Φ(β/|γ|) for ReLU,
  α + (1 − α)Φ(β/|γ|) for leaky ReLU with slope α, and a Gauss-Hermite integral for SiLU.
  A ReLU channel at the `[LAYER-HEALTH]` dead line (β/|γ| = −3) passes Φ(−3) = 0.00135.
- **low pass-through count**: channels whose pass-through is below that same 0.00135
  (ReLU's dead line), the activation-independent analogue of "dead".

The Gaussian input is a model, not a measurement: BN makes each channel's input mean 0 and
variance 1 over the running statistics, but the true distribution need not be normal.
The checkpoint's own architecture says which activation each BN site feeds; nothing is
assumed from the run's name.

Usage: bn_liveness.py [--steps 1000,6000,...] [--site blocks.2.bn1] [--selftest]
"""
import argparse
import json
import math
import os
import struct
import sys

import numpy as np

MODELS = os.path.expanduser("~/Library/Application Support/DrewsChessMachine/Models")

# Run label -> list of (checkpoint stem, step offset). A resumed segment's files are named by
# segment step; the offset turns them into trainer steps.
RUNS = {
    "A (ReLU, const 0.01)": [("20261005-lrA-const01-replay-step", 0), ("20261005-lrA-const01-r1-replay-seg1-step", 36000)],
    "B (ReLU)": [("20261005-lrB-cyc1-replay-step", 0), ("20261005-lrB-cyc1-r1-replay-seg1-step", 36000)],
    "B-leaky (value head)": [("20261005-lrBleaky-cyc1-replay-step", 0)],
    "B-leakyall": [("20261005-lrBleakyall-cyc1-replay-step", 0)],
    "B-silu": [("20261005-lrBsilu-cyc1-replay-step", 0)],
}

RELU_DEAD_LINE = -3.0  # [LAYER-HEALTH] dead threshold on β/|γ|
LOW_PASS = 0.5 * math.erfc(3.0 / math.sqrt(2.0))  # Φ(−3): ReLU pass-through at the dead line
LEAKY_SLOPE = 0.01  # ActivationFunction.leakyReLUNegativeSlope in the Swift app

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(80)  # probabilists' Hermite: ∫ f(z) φ(z) dz
_GH_W = _GH_W / math.sqrt(2.0 * math.pi)


def phi_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=np.float64) / math.sqrt(2.0)))


def silu_derivative(y):
    s = 1.0 / (1.0 + np.exp(-y))
    return s * (1.0 + y * (1.0 - s))


def pass_through(activation, gamma, beta):
    """E|f'(γz + β)| for z ~ N(0, 1), per channel."""
    g = np.asarray(gamma, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    ratio = b / np.abs(g)
    if activation == "relu":
        return phi_cdf(ratio)
    if activation == "leaky_relu":
        return LEAKY_SLOPE + (1.0 - LEAKY_SLOPE) * phi_cdf(ratio)
    if activation == "silu":
        y = g[:, None] * _GH_X[None, :] + b[:, None]
        return (np.abs(silu_derivative(y)) * _GH_W[None, :]).sum(axis=1)
    raise ValueError(f"no pass-through model for activation {activation!r}")


def read_tensors(path, names):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name in names:
            spec = header[name]
            if spec["dtype"] != "F32":
                raise ValueError(f"{path}: {name} is {spec['dtype']}, expected F32")
            f.seek(base + spec["data_offsets"][0])
            out[name] = np.frombuffer(f.read(spec["data_offsets"][1] - spec["data_offsets"][0]), dtype=np.float32)
    return out, header


def site_activations(arch):
    """BN site name -> activation it feeds, from the checkpoint's own architecture.

    Only the shapes this experiment uses are supported (one pre-activation block group, no SE,
    no feature skip, intermediate_conv policy head, WDL value head); anything else is refused
    rather than guessed."""
    groups = arch["block_groups"]
    if len(groups) != 1 or groups[0]["activation_style"] != "pre" or groups[0]["se_style"] != "none":
        raise ValueError("unsupported architecture for this script (expects one pre-activation, SE-less block group)")
    if arch.get("feature_skip_source", "none") != "none" or arch["policy_head_style"] != "intermediate_conv":
        raise ValueError("unsupported architecture for this script (feature skip or policy head style)")
    if "activation_function" in arch:
        raise ValueError("pre-v9 architecture: per-site activations are not stated; this script does not resolve legacy files")
    block_act = groups[0]["activation_function"]
    sites = {}
    for i in range(groups[0]["count"]):
        sites[f"blocks.{i}.bn1"] = block_act
        sites[f"blocks.{i}.bn2"] = block_act
    sites["tower_final_bn"] = arch["tower_end_activation"]
    sites["policy.pre_bn"] = arch["policy_head_activation"]
    sites["value.bn"] = arch["value_head_conv_activation"]
    for site, act in sites.items():
        if act == "does_not_apply":
            raise ValueError(f"{site} is does_not_apply in a topology where it exists")
    return sites


def legacy_relu_sites(arch_json):
    """Pre-v9 files (A and B were written by build 2320) state one activation for every site."""
    arch = json.loads(arch_json)
    if "activation_function" not in arch:
        return None
    act = arch["activation_function"]
    count = arch["block_groups"][0]["count"]
    sites = {f"blocks.{i}.bn{j}": arch["block_groups"][0]["activation_function"] for i in range(count) for j in (1, 2)}
    sites.update({"tower_final_bn": act, "policy.pre_bn": act, "value.bn": act})
    return sites


def checkpoints(run):
    found = {}
    for stem, offset in RUNS[run]:
        for name in os.listdir(MODELS):
            if name.startswith(stem) and name.endswith(".safetensors"):
                digits = name[len(stem):-len(".safetensors")]
                if digits.isdigit():
                    found[int(digits) + offset] = os.path.join(MODELS, name)
    return found


def analyze(path):
    _, header = read_tensors(path, [])
    arch_json = header["__metadata__"]["architecture"]
    sites = legacy_relu_sites(arch_json) or site_activations(json.loads(arch_json))
    names = [f"{s}.{p}" for s in sites for p in ("weight", "bias")]
    t, _ = read_tensors(path, names)
    rows = {}
    for site, act in sites.items():
        g, b = t[f"{site}.weight"], t[f"{site}.bias"]
        ratio = b.astype(np.float64) / np.abs(g.astype(np.float64))
        p = pass_through(act, g, b)
        rows[site] = dict(act=act, ch=len(g), min_ratio=float(ratio.min()), median_beta=float(np.median(b)),
                          median_abs_gamma=float(np.median(np.abs(g))), below_dead=int((ratio < RELU_DEAD_LINE).sum()),
                          mostly_off=int(((ratio >= RELU_DEAD_LINE) & (ratio < -2.0)).sum()),
                          min_pass=float(p.min()), median_pass=float(np.median(p)), low_pass=int((p < LOW_PASS).sum()))
    return rows


def selftest():
    rng = np.random.default_rng(20261006)
    z = rng.standard_normal(2_000_000)
    for act in ("relu", "leaky_relu", "silu"):
        for g, b in [(1.0, 0.0), (1.3, -2.0), (0.7, 0.5), (-0.9, -1.1), (2.0, -6.0)]:
            y = g * z + b
            if act == "relu":
                d = (y > 0).astype(float)
            elif act == "leaky_relu":
                d = np.where(y > 0, 1.0, LEAKY_SLOPE)
            else:
                d = np.abs(silu_derivative(y))
            mc = d.mean()
            model = float(pass_through(act, np.array([g]), np.array([b]))[0])
            assert abs(mc - model) < 2e-3, (act, g, b, mc, model)
    # SiLU derivative against a central finite difference.
    y = np.linspace(-8, 8, 2001)
    silu = lambda v: v / (1.0 + np.exp(-v))
    fd = (silu(y + 1e-5) - silu(y - 1e-5)) / 2e-5
    assert np.max(np.abs(fd - silu_derivative(y))) < 1e-6
    assert abs(LOW_PASS - 0.0013499) < 1e-6
    print("selftest ok: pass-through matches Monte Carlo for relu / leaky_relu / silu; SiLU derivative matches finite differences")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="1000,6000,11000,16000,21000,26000,31000,36000,40000")
    ap.add_argument("--site", default=None, help="one BN site; default: a per-run summary over all sites")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    steps = [int(s) for s in args.steps.split(",")]
    for run in RUNS:
        cps = checkpoints(run)
        if not cps:
            continue
        print(f"\n### {run}\n")
        if args.site:
            print(f"| step | {args.site} act | min β/|γ| | median β | median |γ| | β/|γ| < −3 | −3 ≤ β/|γ| < −2 | min pass-through | median pass-through | pass-through < Φ(−3) |")
            print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        else:
            print("| step | sites | channels β/|γ| < −3 | mostly off (−3…−2) | channels pass-through < Φ(−3) | lowest pass-through (site) | lowest median pass-through (site) |")
            print("|---:|---:|---:|---:|---:|---|---|")
        for s in steps:
            if s not in cps:
                continue
            rows = analyze(cps[s])
            if args.site:
                r = rows[args.site]
                print(f"| {s:,} | {r['act']} | {r['min_ratio']:.2f} | {r['median_beta']:.3f} | {r['median_abs_gamma']:.3f} | {r['below_dead']} | {r['mostly_off']} | {r['min_pass']:.4f} | {r['median_pass']:.3f} | {r['low_pass']} |")
            else:
                lo = min(rows, key=lambda k: rows[k]["min_pass"])
                lm = min(rows, key=lambda k: rows[k]["median_pass"])
                print(f"| {s:,} | {len(rows)} | {sum(r['below_dead'] for r in rows.values())} | {sum(r['mostly_off'] for r in rows.values())} | "
                      f"{sum(r['low_pass'] for r in rows.values())} | {rows[lo]['min_pass']:.4f} ({lo}, {rows[lo]['act']}) | {rows[lm]['median_pass']:.3f} ({lm}, {rows[lm]['act']}) |")


if __name__ == "__main__":
    main()
