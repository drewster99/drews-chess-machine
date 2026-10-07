#!/usr/bin/env python3
"""Per-site batch-norm health for the LR-schedule arms, for every activation function.

`[LAYER-HEALTH]` classifies a BN channel as dead / mostly off / always on from β/|γ|, which
only means something for ReLU and leaky ReLU, so it prints n/a for SiLU and GELU sites
(its activation-aware `parked` counts, from the app's `BatchNormPassThrough`, cover every
activation; this script mirrors that model). This script reads the enumerated checkpoints
directly and reports, for every BN site that feeds an activation, two activation-aware
numbers:

- **pass-through** P: the expected magnitude of the activation's derivative, E|f'(γz + β)|,
  with z ~ N(0, 1) standing for the BN-normalized input — the fraction of the incoming
  gradient the channel passes back on average. Φ(β/|γ|) for ReLU, α + (1 − α)Φ(β/|γ|) for
  leaky ReLU with slope α, and a numerical integral for SiLU and (exact, erf) GELU.
- **excess pass-through** X: the part of P above what the activation passes whatever its
  input (its floor: 0 for ReLU, SiLU and GELU, α for leaky ReLU), as a share of the range above
  that floor, X = (P − floor) / (1 − floor). For ReLU and leaky ReLU X is exactly Φ(β/|γ|),
  the share of inputs on the unit-slope side; for SiLU and GELU it is P.

A channel is **parked** when X < Φ(−3) and **mostly off** when Φ(−3) ≤ X < Φ(−2). For ReLU
and leaky ReLU these are `[LAYER-HEALTH]`'s dead (β/|γ| < −3) and mostly off (−3 ≤ β/|γ| < −2)
counts exactly, γ = 0 included; a parked ReLU channel passes no gradient, a parked leaky one
only its α leak — it is trainable but computes the linear αy, its switch unused. For SiLU,
whose derivative is not scale-invariant, the same X line sits at a β that depends on |γ|
(about −9 at |γ| = 1, far beyond ReLU's −3); a parked SiLU channel passes less than Φ(−3) of
its gradient and outputs about 0. Counts are reported per activation and never summed across
activations.

The Gaussian input is a model, not a measurement: BN makes each channel's input mean 0 and
variance 1 over the running statistics, but the true distribution need not be normal.
The checkpoint's own architecture says which activation each BN site feeds (through
`scripts/dcm_arch.py`, the app's rules); the trainer step is its lineage record's
`cum_trainer_step` (through `scripts/dcm_lineage.py`). Nothing is taken from a file name but
where to look.

Usage: bn_liveness.py [--steps 1000,6000,...] [--site blocks.2.bn1] [--selftest]
"""
import argparse
import copy
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import dcm_arch  # noqa: E402
import dcm_lineage  # noqa: E402

MODELS = os.path.expanduser("~/Library/Application Support/DrewsChessMachine/Models")
SAFETENSORS = ".safetensors"

# Run label -> the name prefixes of its enumerated checkpoints, one per segment. A run's
# first segment writes `<stem>-replay-step<N>`. A resumed segment written before
# architecture format v11 wrote `<stem>-replay-seg<k>-step<N>` (N its own step); one
# resumed on a build from v11 on continues `<stem>-replay-step<trainer step>` under its
# stem, so the run's first prefix covers it.
RUNS = {
    "A (ReLU, const 0.01)": ("20261005-lrA-const01-replay-step", "20261005-lrA-const01-r1-replay-seg1-step"),
    "B (ReLU)": ("20261005-lrB-cyc1-replay-step", "20261005-lrB-cyc1-r1-replay-seg1-step"),
    "C (ReLU, cycle peak 10)": ("20261005-lrC-cyc10-replay-step", "20261005-lrC-cyc10-r1-replay-seg1-step"),
    "B-leaky (value head)": ("20261005-lrBleaky-cyc1-replay-step",),
    "B-leakyall": ("20261005-lrBleakyall-cyc1-replay-step",),
    "B-silu": ("20261005-lrBsilu-cyc1-replay-step",),
    "B-silu clip 1.0 (from 18k)": ("20261006-lrBsilu-clip1-replay-seg1-step",),
    "B-silu control cap 15 (from 18k)": ("20261006-lrBsilu-ctl15-replay-seg1-step",),
    "C-leaky (cycle peak 10)": ("20261005-lrCleaky-cyc10-replay-step",),
}

# Activations this script has a pass-through model for, in report order. The Swift app's
# `BatchNormPassThrough` (Training/BatchNormPassThrough.swift) is the single source of these
# models and of the integration below; this script mirrors it, and the app's tests check the
# two agree (DrewsChessMachineTests/BatchNormPassThroughTests.swift).
MODELED_ACTIVATIONS = ("relu", "leaky_relu", "silu", "gelu")
LEAKY_SLOPE = 0.01  # ActivationFunction.leakyReLUNegativeSlope in the Swift app
PASS_THROUGH_FLOOR = {"relu": 0.0, "leaky_relu": LEAKY_SLOPE, "silu": 0.0, "gelu": 0.0}
# LayerHealth.deadBetaOverAbsGamma / mostlyOffBetaOverAbsGamma, as excess pass-through.
PARKED_BELOW = 0.5 * math.erfc(3.0 / math.sqrt(2.0))  # Φ(−3)
MOSTLY_OFF_BELOW = 0.5 * math.erfc(2.0 / math.sqrt(2.0))  # Φ(−2)

# SiLU / GELU integration: Y = γz + β ~ N(β, γ²) is integrated in y over β ± SMOOTH_SIGMA_SPAN·|γ|,
# clipped to ±SMOOTH_SATURATION; beyond it |f'(y)| equals 1 (above) or 0 (below) to well
# under double precision, so the mass there is added in closed form. Panels are at most
# SMOOTH_PANEL_Y wide in y and |γ| / SMOOTH_PANELS_PER_SIGMA wide relative to the Gaussian, with
# a panel edge at f''s sign change so |f'| is smooth inside each panel.
SMOOTH_SIGMA_SPAN = 9.0
SMOOTH_SATURATION = 50.0
SMOOTH_PANEL_Y = 0.25
SMOOTH_PANELS_PER_SIGMA = 4.0
_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)


def phi_cdf(x):
    """Φ(x), accurate in the far lower tail (erfc, not 1 + erf)."""
    return 0.5 * np.vectorize(math.erfc, otypes=[np.float64])(-np.asarray(x, dtype=np.float64) / math.sqrt(2.0))


def logistic(y):
    """1 / (1 + e^−y) without overflow for any finite y."""
    e = np.exp(-np.abs(y))
    return np.where(y >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def silu_derivative(y):
    s = logistic(y)
    return s * (1.0 + y * (1.0 - s))


def gelu_derivative(y):
    """Exact (erf) GELU, as the app builds it: d/dy [y·Φ(y)] = Φ(y) + y·φ(y)."""
    y = np.asarray(y, dtype=np.float64)
    return phi_cdf(y) + y * np.exp(-0.5 * y * y) / math.sqrt(2.0 * math.pi)


SMOOTH_DERIVATIVES = {"silu": silu_derivative, "gelu": gelu_derivative}
# A bracket holding the one y where each derivative changes sign (negative below it).
SMOOTH_ROOT_BRACKETS = {"silu": (-2.0, -1.0), "gelu": (-1.0, -0.5)}


def _derivative_root(activation):
    """The one y where f' changes sign (f' < 0 below it), by bisection."""
    derivative = SMOOTH_DERIVATIVES[activation]
    lo, hi = SMOOTH_ROOT_BRACKETS[activation]
    if not (derivative(np.array([lo]))[0] < 0 < derivative(np.array([hi]))[0]):
        raise AssertionError(f"{activation}' does not change sign in the bisection bracket")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if derivative(np.array([mid]))[0] < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


SMOOTH_DERIVATIVE_ROOTS = {act: _derivative_root(act) for act in SMOOTH_DERIVATIVES}
SILU_DERIVATIVE_ROOT = SMOOTH_DERIVATIVE_ROOTS["silu"]


def smooth_pass_through(activation, gamma, beta):
    """E|f'(Y)| for Y ~ N(β, γ²), one channel, f SiLU or GELU; γ = 0 is the constant input β."""
    derivative = SMOOTH_DERIVATIVES[activation]
    root = SMOOTH_DERIVATIVE_ROOTS[activation]
    sigma = abs(gamma)
    if sigma == 0.0:
        return float(abs(derivative(np.array([beta]))[0]))
    lo = max(beta - SMOOTH_SIGMA_SPAN * sigma, -SMOOTH_SATURATION)
    hi = min(beta + SMOOTH_SIGMA_SPAN * sigma, SMOOTH_SATURATION)
    saturated_mass = float(phi_cdf((beta - SMOOTH_SATURATION) / sigma)) if hi == SMOOTH_SATURATION else 0.0
    if lo >= hi:
        return saturated_mass
    width = min(SMOOTH_PANEL_Y, sigma / SMOOTH_PANELS_PER_SIGMA)
    edges = [lo, root, hi] if lo < root < hi else [lo, hi]
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        bounds = np.linspace(a, b, max(1, math.ceil((b - a) / width)) + 1)
        middle = 0.5 * (bounds[:-1] + bounds[1:])
        half = 0.5 * (bounds[1:] - bounds[:-1])
        y = (middle[:, None] + half[:, None] * _GL_X[None, :]).ravel()
        w = (half[:, None] * _GL_W[None, :]).ravel()
        density = np.exp(-0.5 * ((y - beta) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))
        total += float((np.abs(derivative(y)) * density * w).sum())
    return total + saturated_mass


def silu_pass_through(gamma, beta):
    """E|silu'(Y)| for Y ~ N(β, γ²), one channel; γ = 0 is the constant input β."""
    return smooth_pass_through("silu", gamma, beta)


def excess_pass_through(activation, gamma, beta):
    """X per channel (see the module doc); γ = 0 is the constant input β. Finite inputs only."""
    g = np.asarray(gamma, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    if activation in ("relu", "leaky_relu"):
        x = np.empty_like(b)
        zero = g == 0
        x[zero] = (b[zero] > 0).astype(np.float64)
        x[~zero] = phi_cdf(b[~zero] / np.abs(g[~zero]))
        return x
    if activation in SMOOTH_DERIVATIVES:
        return np.array([smooth_pass_through(activation, gi, bi) for gi, bi in zip(g, b)])
    raise ValueError(f"no pass-through model for activation {activation!r}")


def pass_through(activation, gamma, beta):
    """P = E|f'(γz + β)| for z ~ N(0, 1), per channel: floor + (1 − floor)·X."""
    floor = PASS_THROUGH_FLOOR[activation]
    return floor + (1.0 - floor) * excess_pass_through(activation, gamma, beta)


def bn_site_activations(metadata, source):
    """BN site name -> the activation that consumes its output, from the file's own
    architecture as the app resolves it: the block group's `activation_function` from
    `dcm_arch.norm_arch_md`, the tower-end and head sites from `dcm_arch.site_activations_md`.
    Both gate on `dcm_format_version`, refuse what the app refuses, and resolve a pre-v9 file's
    single `activation_function`.

    Only the topology this experiment uses is mapped — one pre-activation, SE-less block group,
    no feature skip, an intermediate_conv policy head — because only its BN → activation wiring
    was checked against `ChessNetwork`; any other file is refused rather than guessed. Its
    `stem.bn` feeds no activation and is not a site."""
    arch = dcm_arch.norm_arch_md(metadata)
    sites_of_file = dcm_arch.site_activations_md(metadata)
    groups = arch["block_groups"]
    if len(groups) != 1 or groups[0]["activation_style"] != "pre" or groups[0]["se_style"] != "none":
        raise dcm_arch.ArchitectureError(f"{source}: unsupported architecture for this script "
                                         f"(expects one pre-activation, SE-less block group)")
    if arch["feature_skip_source"] != "none" or arch["policy_head_style"] != "intermediate_conv":
        raise dcm_arch.ArchitectureError(f"{source}: unsupported architecture for this script "
                                         f"(feature skip or policy head style)")
    block = groups[0]["activation_function"]
    sites = {f"blocks.{i}.bn{j}": block for i in range(int(groups[0]["count"])) for j in (1, 2)}
    sites["tower_final_bn"] = sites_of_file["tower_end_activation"]
    sites["policy.pre_bn"] = sites_of_file["policy_head_activation"]
    sites["value.bn"] = sites_of_file["value_head_conv_activation"]
    return sites


def read_bn_parameters(path, header, data_start, sites):
    """Site -> (γ, β) as float64 vectors, read from the tensor index `dcm_arch.read_header`
    returned. Refuses a header whose BN layers are not exactly `sites` plus `stem.bn`, a
    tensor that is not an F32 vector of its stated length, a short read, and γ and β of
    different lengths."""
    in_file = {name[:-len(".running_var")] for name in header if name.endswith(".running_var")}
    expected = set(sites) | {"stem.bn"}
    if in_file != expected:
        raise ValueError(f"{path}: BN layers {sorted(in_file ^ expected)} disagree with the architecture's sites")
    out = {}
    with open(path, "rb") as handle:
        for site in sites:
            vectors = []
            for part in ("weight", "bias"):
                name = f"{site}.{part}"
                spec = header.get(name)
                if spec is None:
                    raise ValueError(f"{path}: no tensor {name}")
                start, end = spec["data_offsets"]
                if spec["dtype"] != "F32" or len(spec["shape"]) != 1 or end - start != 4 * spec["shape"][0]:
                    raise ValueError(f"{path}: {name} is {spec['dtype']} {spec['shape']} in {end - start} bytes; "
                                     f"expected an F32 vector")
                handle.seek(data_start + start)
                raw = handle.read(end - start)
                if len(raw) != end - start:
                    raise ValueError(f"{path}: {name} truncated ({len(raw)} of {end - start} bytes)")
                vectors.append(np.frombuffer(raw, dtype="<f4").astype(np.float64))
            if len(vectors[0]) != len(vectors[1]):
                raise ValueError(f"{path}: {site} has {len(vectors[0])} γ and {len(vectors[1])} β values")
            out[site] = tuple(vectors)
    return out


def index_checkpoints(run, entries):
    """Trainer step -> path for one run, from (file name, path, filename step, metadata, lineage
    record) entries. The step is the record's `cum_trainer_step`. The file's `training_step` must
    equal its filename step and agree with the record under the file's step basis
    (`dcm_lineage.step_reading`): from format v11 it is the trainer step and must equal
    `cum_trainer_step`; before v11 it is the segment's step and must equal `segment_local_step`.
    Two files on one trainer step are refused. That the files are one lineage run whose
    segments agree (one `model_id` per segment among them) is `checkpoints`' check, through
    `dcm_lineage.derive_runs`."""
    found = {}
    for name, path, filename_step, metadata, record in entries:
        reading = dcm_lineage.step_reading(metadata, name)
        stated = reading.stated_training_step
        cum = record["steps"]["cum_trainer_step"]
        if reading.basis == dcm_lineage.BASIS_TRAINER_STEP:
            if stated != cum or filename_step != stated:
                raise ValueError(f"{name}: filename step {filename_step}, training_step {stated}, lineage "
                                 f"cum_trainer_step {cum} disagree (from format v11 all three are the trainer step)")
        else:
            local = record["steps"]["segment_local_step"]
            if stated != local or filename_step != local:
                raise ValueError(f"{name}: filename step {filename_step}, training_step {stated}, "
                                 f"lineage segment_local_step {local} disagree")
        if cum is None:
            raise ValueError(f"{name}: its lineage record holds cum_trainer_step as null (unrecorded history)")
        if cum in found:
            raise ValueError(f"{name} and {os.path.basename(found[cum])} are both trainer step {cum}")
        found[cum] = path
    return found


def checkpoints(run, models_dir=MODELS):
    """Trainer step -> path for a run's enumerated checkpoints (see `index_checkpoints`).
    A file under one of the run's prefixes whose name does not end in a step, a file that
    cannot be read or has no lineage record, files of more than one lineage run, and files
    of one segment that disagree (`dcm_lineage.derive_runs`) are refused."""
    filename_steps = {}
    for prefix in RUNS[run]:
        for name in sorted(os.listdir(models_dir)):
            if not (name.startswith(prefix) and name.endswith(SAFETENSORS)):
                continue
            digits = name[len(prefix):-len(SAFETENSORS)]
            if not digits.isdigit():
                raise ValueError(f"{name}: under {prefix!r} but not an enumerated step checkpoint")
            filename_steps[os.path.join(models_dir, name)] = int(digits)
    recorded, unrecorded, errors = dcm_lineage.scan_files(sorted(filename_steps))
    if errors:
        raise ValueError(f"{run}: unreadable checkpoints: {errors}")
    if unrecorded:
        raise ValueError(f"{run}: checkpoints without a lineage record (trainer step not recorded): "
                         f"{sorted(os.path.basename(p) for p in unrecorded)}")
    lineage_runs = dcm_lineage.derive_runs(recorded)
    if len(lineage_runs) > 1:
        raise ValueError(f"{run}: files from more than one lineage run: {sorted(lineage_runs)}")
    return index_checkpoints(run, [(f.name, f.path, filename_steps[f.path], f.metadata, f.record)
                                   for f in recorded])


def analyze(path):
    """Site -> row of statistics for one checkpoint; one header read."""
    header, data_start = dcm_arch.read_header(path)
    name = os.path.basename(path)
    sites = bn_site_activations(header["__metadata__"], name)
    unmodeled = sorted({act for act in sites.values() if act not in MODELED_ACTIVATIONS})
    if unmodeled:
        raise ValueError(f"{name}: no pass-through model for {', '.join(unmodeled)}")
    parameters = read_bn_parameters(path, header, data_start, sites)
    rows = {}
    for site, act in sites.items():
        g, b = parameters[site]
        finite = np.isfinite(g) & np.isfinite(b)
        nonzero = finite & (g != 0)
        x = excess_pass_through(act, g[finite], b[finite])
        p = PASS_THROUGH_FLOOR[act] + (1.0 - PASS_THROUGH_FLOOR[act]) * x
        ratio = b[nonzero] / np.abs(g[nonzero])
        rows[site] = dict(
            act=act, ch=len(g), non_finite=int((~finite).sum()), zero_gamma=int((finite & (g == 0)).sum()),
            min_ratio=float(ratio.min()) if ratio.size else math.nan,
            median_beta=float(np.median(b[finite])) if x.size else math.nan,
            median_abs_gamma=float(np.median(np.abs(g[finite]))) if x.size else math.nan,
            parked=int((x < PARKED_BELOW).sum()),
            mostly_off=int(((x >= PARKED_BELOW) & (x < MOSTLY_OFF_BELOW)).sum()),
            min_pass=float(p.min()) if p.size else math.nan,
            median_pass=float(np.median(p)) if p.size else math.nan)
    return rows


def selftest():
    rng = np.random.default_rng(20261006)
    z = rng.standard_normal(2_000_000)
    for act in MODELED_ACTIVATIONS:
        for g, b in [(1.0, 0.0), (1.3, -2.0), (0.7, 0.5), (-0.9, -1.1), (2.0, -6.0)]:
            y = g * z + b
            if act == "relu":
                d = (y > 0).astype(float)
            elif act == "leaky_relu":
                d = np.where(y > 0, 1.0, LEAKY_SLOPE)
            else:
                d = np.abs(SMOOTH_DERIVATIVES[act](y))
            model = float(pass_through(act, np.array([g]), np.array([b]))[0])
            assert abs(d.mean() - model) < 2e-3, (act, g, b, d.mean(), model)
    # SiLU at large |γ| and near the parked line, against a dense trapezoid in y.
    for g, b in [(1.0, -9.06), (5.0, -16.64), (10.0, -25.0), (20.0, -60.0), (50.0, -150.0), (100.0, -300.0),
                 (605.0, -1000.0), (605.0, 2000.0), (1e-3, -2.4), (1000.0, 0.0)]:
        y = np.linspace(b - 12 * abs(g), b + 12 * abs(g), 4_000_001)
        dense = np.exp(-0.5 * ((y - b) / g) ** 2) / (abs(g) * math.sqrt(2 * math.pi)) * np.abs(silu_derivative(y))
        reference = float(np.trapezoid(dense, y))
        assert abs(silu_pass_through(g, b) - reference) <= 1e-5 * reference, (g, b, reference)
    # GELU likewise, near its parked line and at large |γ|.
    for g, b in [(1.0, -4.0), (1.0, -5.5), (5.0, -14.0), (20.0, -60.0), (605.0, -1000.0), (1e-3, -0.4),
                 (1000.0, 0.0)]:
        y = np.linspace(b - 12 * abs(g), b + 12 * abs(g), 4_000_001)
        dense = np.exp(-0.5 * ((y - b) / g) ** 2) / (abs(g) * math.sqrt(2 * math.pi)) * np.abs(gelu_derivative(y))
        reference = float(np.trapezoid(dense, y))
        assert abs(smooth_pass_through("gelu", g, b) - reference) <= 1e-5 * reference, (g, b, reference)
    # SiLU derivative against a central finite difference, and its sign change.
    y = np.linspace(-8, 8, 2001)
    silu = lambda v: v * logistic(v)
    fd = (silu(y + 1e-5) - silu(y - 1e-5)) / 2e-5
    assert np.max(np.abs(fd - silu_derivative(y))) < 1e-6
    assert abs(float(silu_derivative(np.array([SILU_DERIVATIVE_ROOT]))[0])) < 1e-12
    gelu = lambda v: v * phi_cdf(v)
    fd = (gelu(y + 1e-5) - gelu(y - 1e-5)) / 2e-5
    assert np.max(np.abs(fd - gelu_derivative(y))) < 1e-6
    assert abs(float(gelu_derivative(np.array([SMOOTH_DERIVATIVE_ROOTS["gelu"]]))[0])) < 1e-12
    # Parked / mostly off equal LayerHealth's β/|γ| bands for ReLU and leaky ReLU, γ = 0 included
    # (LayerHealth: γ = 0 is always on when β > 0, else dead).
    g = np.concatenate([rng.normal(0, 2, 20000), np.zeros(4)])
    b = np.concatenate([rng.normal(-2, 3, 20000), np.array([1.0, 0.0, -1.0, -0.0])])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(g == 0, np.where(b > 0, np.inf, -np.inf), b / np.abs(g))
    for act in ("relu", "leaky_relu"):
        x = excess_pass_through(act, g, b)
        assert np.array_equal(x < PARKED_BELOW, ratio < -3.0), act
        assert np.array_equal((x >= PARKED_BELOW) & (x < MOSTLY_OFF_BELOW), (ratio >= -3.0) & (ratio < -2.0)), act
        assert np.all(np.isfinite(x))
        assert excess_pass_through(act, np.array([]), np.array([])).size == 0
        assert np.array_equal(excess_pass_through(act, np.zeros(2), np.array([1.0, -1.0])), np.array([1.0, 0.0]))
    # Site mapping: a legacy (pre-v9) file, a v9 per-site file, and the refusals.
    legacy_group = dict(count=2, channels=8, conv1_kernel_size=3, conv2_kernel_size=3, se_style="none",
                        se_reduction_ratio=4, use_rezero=False, rezero_alpha_init=0.1, rezero_alpha_cap=0.1,
                        activation_function="relu", activation_style="pre", skip_merge="add",
                        dropout_multiplier=1, se_beta_init="glorot", se_activation="relu")
    legacy = dict(block_groups=[legacy_group], activation_function="relu", policy_head_style="intermediate_conv")
    md = dict(architecture=json.dumps(legacy), dcm_format_version="8")
    assert bn_site_activations(md, "legacy") == {
        "blocks.0.bn1": "relu", "blocks.0.bn2": "relu", "blocks.1.bn1": "relu", "blocks.1.bn2": "relu",
        "tower_final_bn": "relu", "policy.pre_bn": "relu", "value.bn": "relu"}
    v9 = copy.deepcopy(legacy)
    del v9["activation_function"]
    v9["block_groups"][0].update(activation_function="silu", se_activation="silu")
    v9.update(stem_activation="does_not_apply", tower_end_activation="silu", feature_skip_activation="does_not_apply",
              policy_head_activation="leaky_relu", value_head_conv_activation="leaky_relu",
              value_head_fc1_hidden_activation="leaky_relu")
    sites = bn_site_activations(dict(architecture=json.dumps(v9), dcm_format_version="9"), "v9")
    assert sites["blocks.1.bn2"] == "silu" and sites["tower_final_bn"] == "silu" and sites["value.bn"] == "leaky_relu"
    refused = []
    retired = dict(v9, activation_function="relu")
    refused.append(dict(architecture=json.dumps(retired), dcm_format_version="9"))
    two_groups = copy.deepcopy(legacy)
    two_groups["block_groups"].append(dict(legacy_group))
    refused.append(dict(architecture=json.dumps(two_groups), dcm_format_version="8"))
    post = copy.deepcopy(legacy)
    post["block_groups"][0]["activation_style"] = "post"
    refused.append(dict(architecture=json.dumps(post), dcm_format_version="8"))
    refused.append(dict(architecture=json.dumps(legacy), dcm_format_version="12"))
    for case in refused:
        try:
            bn_site_activations(case, "refused")
        except dcm_arch.ArchitectureError:
            continue
        raise AssertionError(f"not refused: {case}")
    # Checkpoint identity: the step from the record, never the name (run and segment agreement
    # is dcm_lineage.derive_runs', tested in documentation/dashboards/tests/test_lineage.py).
    def entry(name, filename_step, local, cum):
        return (name, name, filename_step, dict(training_step=str(local)),
                dict(steps=dict(segment_local_step=local, cum_trainer_step=cum)))

    def v11_entry(name, filename_step, stated, local, cum):
        # A format v11 header: `training_step` is the trainer step, and the segment step is
        # the record's (which the reading takes from the header's own dcm_lineage).
        record = dict(schema=dcm_lineage.SUPPORTED_SCHEMA,
                      run=dict(lineage_run_id="r", segment_index=1, segment_id="s", segment_started_unix=0,
                               start="resume", exact_resume=True, not_exact_items=[],
                               continues_unrecorded_history=False, recorded_unix=0),
                      parent=None, steps=dict(cum_trainer_step=cum, segment_start_trainer_step=None,
                                              segment_local_step=local),
                      fed=dict(cum_games=None, cum_positions=None, segment_games=0, segment_positions=0, corpus=None),
                      time=dict(cum_train_step_sec=None, cum_wall_sec=None, segment_train_step_sec=0.0,
                                segment_wall_sec=0.0),
                      device=dict(hw_model="m", chip="c", is_vm=False, os_version="v", gpu_name="g"),
                      invocation=dict(argv=[], path_kind="replay"), segments=[])
        metadata = dict(training_step=str(stated), dcm_format_version="11", creator="replay",
                        dcm_lineage=json.dumps(record))
        return (name, name, filename_step, metadata, record)
    assert index_checkpoints("t", [entry("a", 1000, 1000, 1000), entry("b", 1000, 1000, 37000)]) == \
        {1000: "a", 37000: "b"}
    # A resumed segment on a v11 build: name step = training_step = cum_trainer_step.
    assert index_checkpoints("t", [entry("a", 1000, 1000, 1000), v11_entry("b", 37000, 37000, 1000, 37000)]) == \
        {1000: "a", 37000: "b"}
    for bad in ([entry("a", 1000, 1000, 1000), entry("b", 2000, 2000, 1000)],
                [entry("a", 1000, 999, 1000)],
                [entry("a", 999, 1000, 1000)],
                [entry("a", 1000, 1000, None)],
                [v11_entry("a", 1000, 1000, 1000, 37000)],
                [v11_entry("a", 1000, 37000, 1000, 37000)]):
        try:
            index_checkpoints("t", bad)
        except ValueError:
            continue
        raise AssertionError(f"not refused: {bad}")
    print("selftest ok: pass-through (Monte Carlo; SiLU and GELU against a dense integral to |γ| = 1000); parked / mostly off "
          "= LayerHealth bands for relu / leaky_relu; site mapping and refusals; checkpoint identity")


def counts_cell(rows, act, key):
    """`n/channels` over the run's sites of one activation, or empty when it has none."""
    channels = sum(r["ch"] for r in rows.values() if r["act"] == act)
    return f"{sum(r[key] for r in rows.values() if r['act'] == act)}/{channels}" if channels else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=None, help="comma-separated trainer steps; default: every checkpoint")
    ap.add_argument("--site", default=None, help="one BN site; default: a per-run summary over all sites")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    requested = None if args.steps is None else [int(s) for s in args.steps.split(",")]
    for run in RUNS:
        cps = checkpoints(run)
        print(f"\n### {run}\n")
        if not cps:
            print(f"No checkpoints under {', '.join(RUNS[run])} in {MODELS}.")
            continue
        steps = sorted(cps) if requested is None else requested
        analyses = {s: analyze(cps[s]) for s in steps if s in cps}
        if not analyses:
            print(f"None of the requested trainer steps has a checkpoint; this run's are {', '.join(f'{s:,}' for s in sorted(cps))}.")
            continue
        if args.site:
            for rows in analyses.values():
                if args.site not in rows:
                    raise SystemExit(f"{run}: no BN site {args.site!r}; its sites are {', '.join(rows)}")
        acts = [a for a in MODELED_ACTIVATIONS if any(r["act"] == a for rows in analyses.values() for r in rows.values())]
        if args.site:
            print(f"| step | {args.site} act | min β/|γ| | median β | median |γ| | parked | mostly off | "
                  f"min pass-through | median pass-through | γ = 0 | non-finite |")
            print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        else:
            print("| step | " + " | ".join(f"{a} parked | {a} mostly off | {a} lowest pass-through (site)" for a in acts)
                  + " | γ = 0 | non-finite |")
            print("|---:|" + "---:|---:|---|" * len(acts) + "---:|---:|")
        for s in steps:
            if s not in analyses:
                print(f"| {s:,} | no checkpoint at this trainer step |")
                continue
            rows = analyses[s]
            if args.site:
                r = rows[args.site]
                print(f"| {s:,} | {r['act']} | {r['min_ratio']:.2f} | {r['median_beta']:.3f} | {r['median_abs_gamma']:.3f} | "
                      f"{r['parked']} | {r['mostly_off']} | {r['min_pass']:.4f} | {r['median_pass']:.3f} | "
                      f"{r['zero_gamma']} | {r['non_finite']} |")
            else:
                cells = []
                for a in acts:
                    of_act = {k: r for k, r in rows.items() if r["act"] == a and not math.isnan(r["min_pass"])}
                    if of_act:
                        lowest = min(of_act, key=lambda k: of_act[k]["min_pass"])
                        lowest_cell = f"{of_act[lowest]['min_pass']:.4f} ({lowest})"
                    else:
                        lowest_cell = "no finite channel"
                    cells += [counts_cell(rows, a, "parked"), counts_cell(rows, a, "mostly_off"), lowest_cell]
                print(f"| {s:,} | " + " | ".join(cells) + f" | {sum(r['zero_gamma'] for r in rows.values())} | "
                      f"{sum(r['non_finite'] for r in rows.values())} |")


if __name__ == "__main__":
    main()
