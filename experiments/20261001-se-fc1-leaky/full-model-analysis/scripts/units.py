#!/usr/bin/env python3
"""Per-unit dead / stuck / always-on measurements for every layer of every checkpoint.

For each run (leaky-FC1, ReLU seed 1, ReLU seed 2) and each checkpoint (fresh
net + every enumerated 1k checkpoint; the leaky and ReLU-seed-1 runs only up to
the latest leaky step, so the two arms are compared at matched steps), writes:

- results/units.csv.gz -- one row per (checkpoint, site, unit). A *site* is one
  way of slicing one tensor into units: a conv's output rows (`*.out`), a
  conv's input columns (`*.in`, i.e. how much the layer still reads each input
  channel), a BN / LayerNorm channel, an FC unit, etc. Columns are described
  in the module docstring of summarize.py.
- results/tensors.csv -- one row per (checkpoint, tensor): non-finite count,
  max |value|, RMS, exact-zero fraction, bf16-grid fraction.

Definitions (thresholds live in fma_lib.py):

- unmoved (decayed slices only: conv / FC weights): cosine with the fresh
  slice > UNMOVED_COSINE and norm ratio within DECAY_RATIO_RELATIVE_TOLERANCE
  of the decay-only factor `f` measured on the stem weights that read the
  always-zero input planes. Such a slice got (essentially) no gradient for the
  whole run -- weight decay is all that ever touched it.
- unchanged (undecayed parameters: BN / LN gamma, beta, biases, ReZero alpha):
  bit-identical to the fresh value.
- vel0: every velocity element of the unit is exactly 0 (zero gradient for the
  last several hundred+ steps). Only runs whose checkpoints save velocity.
- vel_low: unit velocity L2 norm < LOW_VELOCITY_FRACTION x the site median.
- vel_high: unit velocity L2 norm > HIGH_VELOCITY_MULTIPLE x the site median.
- BN followed by ReLU: dead (beta/|gamma| < -3), mostly_off (< -2),
  always_on (> +3). P(on) = Phi(beta/|gamma|) assumes a Gaussian BN input.
- rv_high / rv_low: BN running variance above / below a multiple of that
  BN's median running variance.
- ln_gamma_small: LayerNorm |gamma| < LN_GAMMA_NEAR_ZERO.

Usage: python3 scripts/units.py
"""
import csv
import gzip
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fma_lib as L  # noqa: E402

UNIT_COLUMNS = [
    "run", "step", "model_id", "site", "index", "label", "next_op", "decayed",
    "w_norm", "w_norm_fresh", "ratio_over_decay", "cos_fresh",
    "v_norm", "v_rel_median", "v_zero_elem_frac",
    "gamma", "beta", "beta_over_absgamma", "p_on", "v_gamma", "v_beta",
    "running_mean", "running_var", "rv_rel_median",
    "value", "value_fresh", "v_value",
    "flags",
]
TENSOR_COLUMNS = [
    "run", "step", "model_id", "file", "tensor", "shape", "count", "nonfinite", "abs_max",
    "rms", "zero_frac", "bf16_grid_frac",
]


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "nan"
        return f"{float(value):.9g}"
    return str(value)


class SiteWriter:
    def __init__(self, run, ckpt, fresh, decay_factor, rows):
        self.run, self.ckpt, self.fresh, self.f, self.rows = run, ckpt, fresh, decay_factor, rows

    def _base(self, site, index, label, next_op, decayed):
        return {"run": self.run, "step": self.ckpt.training_step, "model_id": self.ckpt.model_id,
                "site": site, "index": index, "label": label, "next_op": next_op, "decayed": decayed}

    def weight_slices(self, site, now, fresh, velocity, next_op, labels=None):
        """`now`, `fresh`, `velocity` already sliced to [units, ...]."""
        norm = L.rows_norm(now)
        norm0 = L.rows_norm(fresh)
        cos = L.rows_cos(now, fresh)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = norm / norm0 / self.f
        v_norm = L.rows_norm(velocity) if velocity is not None else None
        median = float(np.median(v_norm)) if v_norm is not None else None
        for u in range(now.shape[0]):
            flags = []
            unmoved = self.ckpt.training_step > 0 and cos[u] > L.UNMOVED_COSINE and abs(ratio[u] - 1.0) < L.DECAY_RATIO_RELATIVE_TOLERANCE
            if unmoved:
                flags.append("unmoved")
            row = self._base(site, u, labels[u] if labels else "", next_op, True)
            row.update(w_norm=norm[u], w_norm_fresh=norm0[u], ratio_over_decay=ratio[u], cos_fresh=cos[u])
            if v_norm is not None:
                vz = float(np.mean(velocity[u] == 0.0))
                row.update(v_norm=v_norm[u], v_rel_median=v_norm[u] / median if median > 0 else float("nan"),
                           v_zero_elem_frac=vz)
                if v_norm[u] == 0.0:
                    flags.append("vel0")
                elif v_norm[u] < L.LOW_VELOCITY_FRACTION * median:
                    flags.append("vel_low")
                if v_norm[u] > L.HIGH_VELOCITY_MULTIPLE * median:
                    flags.append("vel_high")
                if 0.0 < vz < 1.0:
                    flags.append("vel0_partial")
            row["flags"] = ";".join(flags)
            self.rows.append(row)

    def norm_channels(self, site, prefix, next_op, has_running):
        gamma = self.ckpt[f"{prefix}.weight"]
        beta = self.ckpt[f"{prefix}.bias"]
        gamma0 = self.fresh[f"{prefix}.weight"]
        beta0 = self.fresh[f"{prefix}.bias"]
        vg = self.ckpt.velocity(f"{prefix}.weight")
        vb = self.ckpt.velocity(f"{prefix}.bias")
        if has_running:
            rm = self.ckpt[f"{prefix}.running_mean"]
            rv = self.ckpt[f"{prefix}.running_var"]
            rv_median = float(np.median(rv))
        v_norm = np.sqrt(vg ** 2 + vb ** 2) if vg is not None else None
        v_median = float(np.median(v_norm)) if v_norm is not None else None
        followed_by_relu = next_op == "relu"
        for c in range(gamma.shape[0]):
            flags = []
            row = self._base(site, c, "", next_op, False)
            ratio = beta[c] / abs(gamma[c]) if gamma[c] != 0 else float("-inf") * np.sign(beta[c] or 1)
            row.update(gamma=gamma[c], beta=beta[c], beta_over_absgamma=ratio)
            if followed_by_relu:
                row["p_on"] = L.phi(ratio) if np.isfinite(ratio) else (1.0 if ratio > 0 else 0.0)
                if ratio < L.DEAD_BETA_OVER_GAMMA:
                    flags.append("dead_bn")
                elif ratio < L.MOSTLY_OFF_BETA_OVER_GAMMA:
                    flags.append("mostly_off")
                if ratio > L.ALWAYS_ON_BETA_OVER_GAMMA:
                    flags.append("always_on")
            if site.endswith("res_ln") and abs(gamma[c]) < L.LN_GAMMA_NEAR_ZERO:
                flags.append("ln_gamma_small")
            if self.ckpt.training_step > 0 and gamma[c] == gamma0[c] and beta[c] == beta0[c]:
                flags.append("unchanged")
            if has_running:
                row.update(running_mean=rm[c], running_var=rv[c], rv_rel_median=rv[c] / rv_median)
                if rv[c] > L.RUNNING_VAR_HIGH_RATIO * rv_median:
                    flags.append("rv_high")
                if rv[c] < L.RUNNING_VAR_LOW_RATIO * rv_median:
                    flags.append("rv_low")
            if v_norm is not None:
                row.update(v_gamma=vg[c], v_beta=vb[c], v_norm=v_norm[c],
                           v_rel_median=v_norm[c] / v_median if v_median > 0 else float("nan"))
                if vg[c] == 0.0 and vb[c] == 0.0:
                    flags.append("vel0")
                elif v_norm[c] < L.LOW_VELOCITY_FRACTION * v_median:
                    flags.append("vel_low")
                if v_norm[c] > L.HIGH_VELOCITY_MULTIPLE * v_median:
                    flags.append("vel_high")
            row["flags"] = ";".join(flags)
            self.rows.append(row)

    def vector(self, site, name, next_op, labels=None):
        """Undecayed bias-like vector, one unit per element."""
        value = self.ckpt[name].reshape(-1)
        value0 = self.fresh[name].reshape(-1)
        v = self.ckpt.velocity(name)
        v = v.reshape(-1) if v is not None else None
        v_median = float(np.median(np.abs(v))) if v is not None else None
        for u in range(value.shape[0]):
            flags = []
            row = self._base(site, u, labels[u] if labels else "", next_op, False)
            row.update(value=value[u], value_fresh=value0[u])
            if self.ckpt.training_step > 0 and value[u] == value0[u]:
                flags.append("unchanged")
            if v is not None:
                row.update(v_value=v[u], v_norm=abs(v[u]),
                           v_rel_median=abs(v[u]) / v_median if v_median > 0 else float("nan"))
                if v[u] == 0.0:
                    flags.append("vel0")
                elif abs(v[u]) < L.LOW_VELOCITY_FRACTION * v_median:
                    flags.append("vel_low")
                if abs(v[u]) > L.HIGH_VELOCITY_MULTIPLE * v_median:
                    flags.append("vel_high")
            row["flags"] = ";".join(flags)
            self.rows.append(row)


def require_supported_architecture(ckpt):
    """Refuses a checkpoint whose topology or head activations `analyze` does
    not model: it labels tower_final_bn, policy.pre_bn, value.bn and the value
    FC1 (and its reader) as ReLU-fed, and assumes the heads read the tower output
    directly. A simple_conv model has no policy pre-block, so its
    policy_head_activation is 'does_not_apply' and it is refused here too."""
    if ckpt.architecture["feature_skip_source"] != "none":
        raise ValueError(f"{ckpt.file}: a feature skip is not modelled (feature_skip_source "
                         f"{ckpt.architecture['feature_skip_source']!r})")
    sites = L.dcm_arch.site_activations_md(ckpt.metadata)
    for key in ("tower_end_activation", "policy_head_activation", "value_head_conv_activation",
                "value_head_fc1_hidden_activation"):
        if sites[key] != "relu":
            raise ValueError(f"{ckpt.file}: {key} is {sites[key]!r}; only a ReLU {key} is modelled")


def analyze(run, ckpt, fresh):
    require_supported_architecture(ckpt)
    rows = []
    stem_now = ckpt["stem.conv.weight"]
    stem0 = fresh["stem.conv.weight"]
    f = float(np.linalg.norm(stem_now[:, L.ALWAYS_ZERO_PLANES]) / np.linalg.norm(stem0[:, L.ALWAYS_ZERO_PLANES]))
    w = SiteWriter(run, ckpt, fresh, f, rows)
    groups = ckpt.architecture["block_groups"]
    if len(groups) != 1:
        raise ValueError(f"{ckpt.file}: expected one block group, got {len(groups)}")
    group = groups[0]
    if group["activation_style"] != "pre" or group["activation_function"] != "relu":
        raise ValueError(f"{ckpt.file}: unexpected block activation {group}")
    if "se_activation" in group:
        se_act = group["se_activation"]
    elif int(ckpt.metadata["dcm_format_version"]) < 5:
        # Same rule as NetworkArchitecture's decoder: files older than format 5
        # predate the field, and their SE FC1 used the group's activation.
        se_act = group["activation_function"]
    else:
        raise ValueError(f"{ckpt.file}: format >= 5 without se_activation")
    blocks = group["count"]
    channels = group["channels"]
    reduced = channels // group["se_reduction_ratio"]

    def weight(name, transpose_in=False):
        now, init, vel = ckpt[name], fresh[name], ckpt.velocity(name)
        if transpose_in:
            now, init = np.swapaxes(now, 0, 1), np.swapaxes(init, 0, 1)
            vel = np.swapaxes(vel, 0, 1) if vel is not None else None
        return now, init, vel

    # Stem: conv (30 -> 128, 7x7) -> BN; no activation (pre-act tower).
    w.weight_slices("stem.conv.out", *weight("stem.conv.weight"), next_op="stem.bn")
    w.weight_slices("stem.conv.in", *weight("stem.conv.weight", transpose_in=True),
                    next_op="stem.conv", labels=[f"plane {p} {L.PLANE_LABELS[p]}" for p in range(30)])
    w.norm_channels("stem.bn", "stem.bn", next_op="stream x0 (no activation)", has_running=True)
    for b in range(blocks):
        p = f"blocks.{b}"
        w.norm_channels(f"{p}.bn1", f"{p}.bn1", next_op="relu", has_running=True)
        w.weight_slices(f"{p}.conv1.out", *weight(f"{p}.conv1.weight"), next_op=f"{p}.bn2")
        w.weight_slices(f"{p}.conv1.in", *weight(f"{p}.conv1.weight", transpose_in=True), next_op="reads relu(bn1)")
        w.norm_channels(f"{p}.bn2", f"{p}.bn2", next_op="relu", has_running=True)
        w.weight_slices(f"{p}.conv2.out", *weight(f"{p}.conv2.weight"), next_op="SE + branch")
        w.weight_slices(f"{p}.conv2.in", *weight(f"{p}.conv2.weight", transpose_in=True), next_op="reads relu(bn2)")
        se = f"{p}.se_scalebias"
        w.weight_slices(f"{p}.se.fc1", *weight(f"{se}.fc1.weight"), next_op=se_act)
        w.vector(f"{p}.se.fc1.bias", f"{se}.fc1.bias", next_op=se_act)
        labels = [f"gamma c{c}" for c in range(channels)] + [f"beta c{c}" for c in range(channels)]
        w.weight_slices(f"{p}.se.fc2.out", *weight(f"{se}.fc2.weight"), next_op="sigmoid (gamma) / add (beta)", labels=labels)
        w.vector(f"{p}.se.fc2.bias", f"{se}.fc2.bias", next_op="sigmoid (gamma) / add (beta)", labels=labels)
        fc2_now, fc2_init, fc2_vel = weight(f"{se}.fc2.weight")
        half = channels
        for part, sl in (("gamma", slice(0, half)), ("beta", slice(half, 2 * half))):
            w.weight_slices(f"{p}.se.fc2.in_{part}", fc2_now[sl].T, fc2_init[sl].T,
                            fc2_vel[sl].T if fc2_vel is not None else None, next_op=f"reads {se_act}(fc1)")
        alpha = float(ckpt[f"{p}.rezero_alpha"][0])
        alpha0 = float(fresh[f"{p}.rezero_alpha"][0])
        rezero = ckpt.rezero_blocks[b]
        w.vector(f"{p}.rezero_alpha", f"{p}.rezero_alpha", next_op="C*tanh(alpha/C) x branch",
                 labels=[f"raw {alpha:.6f} eff {rezero.effective(alpha):.6f} (init raw {alpha0:.6f} eff {rezero.effective(alpha0):.6f}, cap {rezero.alpha_cap:.6f})"])
        w.norm_channels(f"{p}.res_ln", f"{p}.res_ln", next_op="residual stream", has_running=False)
    w.norm_channels("tower_final_bn", "tower_final_bn", next_op="relu", has_running=True)
    # Policy head (intermediate_conv): pre_conv 1x1 -> pre_bn -> relu -> conv 1x1 + bias.
    w.weight_slices("policy.pre_conv.out", *weight("policy.pre_conv.weight"), next_op="policy.pre_bn")
    w.weight_slices("policy.pre_conv.in", *weight("policy.pre_conv.weight", transpose_in=True), next_op="reads relu(tower_final_bn)")
    w.norm_channels("policy.pre_bn", "policy.pre_bn", next_op="relu", has_running=True)
    from_policy_lib = _policy_labels()
    w.weight_slices("policy.conv.out", *weight("policy.conv.weight"), next_op="logits", labels=from_policy_lib)
    w.vector("policy.conv.bias", "policy.conv.bias", next_op="logits", labels=from_policy_lib)
    w.weight_slices("policy.conv.in", *weight("policy.conv.weight", transpose_in=True), next_op="reads relu(policy.pre_bn)")
    # Value head: conv 1x1 (128 -> 16) -> BN -> relu -> flatten (c*64 + square) -> fc1 128 -> relu -> fc2 3.
    w.weight_slices("value.conv.out", *weight("value.conv.weight"), next_op="value.bn")
    w.weight_slices("value.conv.in", *weight("value.conv.weight", transpose_in=True), next_op="reads relu(tower_final_bn)")
    w.norm_channels("value.bn", "value.bn", next_op="relu", has_running=True)
    w.weight_slices("value.fc1", *weight("value.fc1.weight"), next_op="relu")
    w.vector("value.fc1.bias", "value.fc1.bias", next_op="relu")
    fc1_now, fc1_init, fc1_vel = weight("value.fc1.weight")
    vc = ckpt["value.conv.weight"].shape[0]

    def group_cols(a):
        return a.reshape(a.shape[0], vc, 64).transpose(1, 0, 2).reshape(vc, -1)

    w.weight_slices("value.fc1.in_group", group_cols(fc1_now), group_cols(fc1_init),
                    group_cols(fc1_vel) if fc1_vel is not None else None,
                    next_op="reads relu(value.bn) channel x 64 squares")
    wdl = ["win", "draw", "loss"]
    w.weight_slices("value.wdl_fc2.out", *weight("value.wdl_fc2.weight"), next_op="softmax", labels=wdl)
    w.vector("value.wdl_fc2.bias", "value.wdl_fc2.bias", next_op="softmax", labels=wdl)
    w.weight_slices("value.wdl_fc2.in", *weight("value.wdl_fc2.weight", transpose_in=True), next_op="reads relu(value.fc1)")
    return rows, f


def _policy_labels():
    sys.path.insert(0, os.path.abspath(os.path.join(L.HERE, "..", "..", "..", "..",
                                                     "documentation", "research", "policy-head-2026-10-01", "scripts")))
    import policy_head_lib  # noqa: E402
    return [f"{c} {policy_head_lib.channel_label(c)}" for c in range(76)]


def tensor_rows(run, ckpt):
    rows = []
    for name in sorted(ckpt.tensors):
        values = ckpt.tensors[name].reshape(-1)
        finite = np.isfinite(values)
        v = values[finite]
        rows.append({
            "run": run, "step": ckpt.training_step, "model_id": ckpt.model_id, "file": ckpt.file,
            "tensor": name, "shape": "x".join(str(d) for d in ckpt.tensors[name].shape),
            "count": values.size, "nonfinite": int(values.size - finite.sum()),
            "abs_max": float(np.abs(v).max()) if v.size else float("nan"),
            "rms": float(np.sqrt(np.mean(v * v))) if v.size else float("nan"),
            "zero_frac": float(np.mean(values == 0.0)),
            "bf16_grid_frac": L.bf16_grid_fraction(ckpt.f32[name]),
        })
    return rows


def main():
    unit_rows, tensor_rows_all = [], []
    leaky_files, _, _ = L.discover("leaky")
    leaky_latest = max(s for s in leaky_files)
    manifest = []
    fresh_cache = {}
    for run in ("leaky", "relu", "relu_s2"):
        files, fresh_id, trained_id = L.discover(run)
        fresh = L.Checkpoint(files[0])
        fresh_cache[run] = fresh
        for step, path in files.items():
            if run in ("leaky", "relu") and step > leaky_latest:
                continue
            ckpt = fresh if step == 0 else L.Checkpoint(path)
            rows, f = analyze(run, ckpt, fresh)
            unit_rows += rows
            tensor_rows_all += tensor_rows(run, ckpt)
            manifest.append((run, step, ckpt.model_id, ckpt.metadata.get("built_by_build", ""),
                             ckpt.metadata.get("built_by_git", ""), ckpt.has_velocity, f, ckpt.file))
            print(f"{run} step {step}: {ckpt.file} model {ckpt.model_id} velocity={ckpt.has_velocity} decay f={f:.6f}", file=sys.stderr)
    # The leaky fresh net is a bit-exact derived copy of the ReLU fresh net.
    differing = [n for n in fresh_cache["relu"].tensors
                 if not np.array_equal(fresh_cache["relu"].f32[n], fresh_cache["leaky"].f32[n])]
    os.makedirs(L.RESULTS_DIR, exist_ok=True)
    with gzip.open(os.path.join(L.RESULTS_DIR, "units.csv.gz"), "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNIT_COLUMNS)
        writer.writeheader()
        for row in unit_rows:
            writer.writerow({k: fmt(row.get(k)) for k in UNIT_COLUMNS})
    with open(os.path.join(L.RESULTS_DIR, "tensors.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TENSOR_COLUMNS)
        writer.writeheader()
        for row in tensor_rows_all:
            writer.writerow({k: fmt(row.get(k)) for k in TENSOR_COLUMNS})
    with open(os.path.join(L.RESULTS_DIR, "manifest.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "step", "model_id", "built_by_build", "built_by_git", "has_velocity",
                         "decay_only_factor_vs_fresh", "file"])
        for m in manifest:
            writer.writerow([fmt(x) for x in m])
    print(f"leaky fresh vs relu fresh: {len(differing)} differing tensors {differing}", file=sys.stderr)
    print(f"wrote {len(unit_rows)} unit rows, {len(tensor_rows_all)} tensor rows", file=sys.stderr)


if __name__ == "__main__":
    main()
