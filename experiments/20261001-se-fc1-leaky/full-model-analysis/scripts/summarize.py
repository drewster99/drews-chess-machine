#!/usr/bin/env python3
"""Turn results/units.csv.gz (+ tensors.csv, input-feature CSVs) into the report tables.

Every table is written in full (every row) as markdown under results/, plus
machine-readable CSVs where a table is long. Run units.py and
input_features.py first.

Derived definitions used here (in addition to units.py's flags):

- weak (velocity runs only): unit velocity L2 norm < 5% of the site's 90th
  percentile unit velocity norm. The site *median* is the wrong reference for
  the SE bottleneck: more than half of each block's FC1 units carry only the
  leaky slope's trickle, so the median unit is itself a weak one and
  "< 5% of median" (the README's earlier measure) never fires there.
- SE unit usage (all runs, weights-only): the angle between unit u's FC2 input
  column (the 256 weights reading FC1 unit u: 128 gamma + 128 beta) and its
  fresh value. FC2's gradient for that column is proportional to unit u's
  output, so the angle integrates how much u was ever used. used >= 5 deg,
  trickle 1-5 deg, unused < 1 deg.
- energy share of a BN input channel: (running_var + running_mean^2) / sum
  over channels (uniform = 1/C).

units.csv.gz columns:
  run, step, model_id     -- checkpoint identity (from __metadata__)
  site                    -- tensor slicing, e.g. blocks.1.conv1.out (output rows),
                             blocks.1.conv1.in (input columns), blocks.1.bn1 (channels),
                             blocks.1.se.fc1 (FC1 weight rows), .se.fc2.in_gamma /
                             .in_beta (FC2 columns reading each FC1 unit, split by half),
                             value.fc1.in_group (fc1 columns grouped by value-conv channel)
  index, label            -- unit index within the site; plane / policy-channel / WDL names
  next_op                 -- what consumes the unit (relu, residual stream, ...)
  decayed                 -- 1 for weight-decayed slices (conv / FC weights)
  w_norm, w_norm_fresh    -- slice L2 norm now / in the fresh net
  ratio_over_decay        -- (w_norm / w_norm_fresh) / decay-only factor (1 = decay only)
  cos_fresh               -- cosine of the slice with its fresh value
  v_norm, v_rel_median    -- velocity L2 norm of the unit; / site median
  v_zero_elem_frac        -- share of the unit's velocity elements that are exactly 0
  gamma, beta, beta_over_absgamma, p_on, v_gamma, v_beta  -- BN / LN channel parameters
  running_mean, running_var, rv_rel_median               -- BN running statistics
  value, value_fresh, v_value                            -- scalar parameters (biases, alpha)
  flags                   -- ';'-joined flags defined in units.py
"""
import math
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fma_lib as L  # noqa: E402

R = L.RESULTS_DIR
RUN_LABEL = {"leaky": "leaky", "relu": "ReLU s1", "relu_s2": "ReLU s2"}
WEAK_FRACTION_OF_P90 = 0.05
SE_USED_DEG = 5.0
SE_UNUSED_DEG = 1.0


def md_table(df, float_format="{:.4g}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else float_format.format(v))
            else:
                cells.append("" if (v is None or (isinstance(v, float) and math.isnan(v))) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write(name, text):
    with open(os.path.join(R, name), "w") as handle:
        handle.write(text.rstrip() + "\n")
    print(f"wrote {name}", file=sys.stderr)


def load():
    units = pd.read_csv(os.path.join(R, "units.csv.gz"), low_memory=False)
    units["flags"] = units["flags"].fillna("")
    # weak = < 5% of the site's p90 velocity norm (velocity runs only).
    p90 = units.groupby(["run", "step", "site"])["v_norm"].transform(lambda s: s.quantile(0.9))
    units["v_rel_p90"] = units["v_norm"] / p90
    # False where no velocity was saved; counts only consider rows with velocity.
    units["weak"] = (units["v_rel_p90"] < WEAK_FRACTION_OF_P90).astype(bool)
    return units


def has_flag(series, flag):
    return series.str.split(";").apply(lambda fs: flag in fs)


FLAGS = ["unmoved", "unchanged", "vel0", "weak", "vel_low", "vel_high", "dead_bn", "mostly_off",
         "always_on", "rv_high", "rv_low", "ln_gamma_small"]


def counts_at(units, run, step):
    x = units[(units.run == run) & (units.step == step)]
    out = {}
    for site, g in x.groupby("site", sort=False):
        c = {"units": len(g)}
        for f in FLAGS:
            if f == "weak":
                c[f] = int(g["weak"].sum()) if g["v_norm"].notna().any() else None
            elif f in ("vel0", "vel_low", "vel_high") and not g["v_norm"].notna().any():
                c[f] = None
            else:
                c[f] = int(has_flag(g["flags"], f).sum())
        out[site] = c
    return out


def site_order(units):
    seen = []
    for s in units[units.run == "leaky"]["site"]:
        if s not in seen:
            seen.append(s)
    return seen


def counts_table(units, latest, s2_step):
    leaky = counts_at(units, "leaky", latest)
    relu = counts_at(units, "relu", latest)
    s2 = counts_at(units, "relu_s2", s2_step)
    rows = []
    for site in site_order(units):
        row = {"site": site, "units": leaky[site]["units"]}
        for f in FLAGS:
            vals = [leaky[site][f], relu[site][f], s2[site][f]]
            if all(v in (0, None) for v in vals):
                row[f] = "."
            else:
                row[f] = " / ".join("-" if v is None else str(v) for v in vals)
        rows.append(row)
    return pd.DataFrame(rows)


def counts_by_step(units, steps_leaky, steps_s2):
    rows = []
    for run, steps in (("leaky", steps_leaky), ("relu", steps_leaky), ("relu_s2", steps_s2)):
        for step in steps:
            c = counts_at(units, run, step)
            for site, cc in c.items():
                for f in FLAGS:
                    if cc[f] is not None:
                        rows.append({"run": run, "step": step, "site": site, "flag": f, "count": cc[f], "units": cc["units"]})
    return pd.DataFrame(rows)


def se_unit_tables(units, latest, s2_step, steps):
    """Per block, per FC1 unit: FC2-column angle (both seed-1 arms, same init),
    FC1-row angle, leaky velocity / p90, weak persistence."""
    import fma_lib as LL
    lf, _, _ = LL.discover("leaky")
    rf, _, _ = LL.discover("relu")
    sf, _, _ = LL.discover("relu_s2")
    fresh = LL.Checkpoint(lf[0])
    fresh_s2 = LL.Checkpoint(sf[0])

    def angles(now, init, name, axis):
        a, b = now[name], init[name]
        if axis == 0:  # columns of the saved [out, in] FC2 weight = one per FC1 unit
            cos = (a * b).sum(0) / np.linalg.norm(a, axis=0) / np.linalg.norm(b, axis=0)
        else:
            cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    blocks = 3
    detail_rows = []
    usage_rows = []
    cache = {}

    def ck(files, step):
        key = (files[step],)
        if key not in cache:
            cache[key] = LL.Checkpoint(files[step])
        return cache[key]

    for step in steps:
        for run, files, init in (("leaky", lf, fresh), ("relu", rf, fresh)):
            c = ck(files, step)
            for b in range(blocks):
                ang = angles(c, init, f"blocks.{b}.se_scalebias.fc2.weight", 0)
                usage_rows.append({"run": run, "step": step, "block": b,
                                   "used (>=5 deg)": int((ang >= SE_USED_DEG).sum()),
                                   "trickle (1-5 deg)": int(((ang >= SE_UNUSED_DEG) & (ang < SE_USED_DEG)).sum()),
                                   "unused (<1 deg)": int((ang < SE_UNUSED_DEG).sum())})
    for step in sorted(s for s in sf if s > 0):
        c = ck(sf, step)
        for b in range(blocks):
            ang = angles(c, fresh_s2, f"blocks.{b}.se_scalebias.fc2.weight", 0)
            usage_rows.append({"run": "relu_s2", "step": step, "block": b,
                               "used (>=5 deg)": int((ang >= SE_USED_DEG).sum()),
                               "trickle (1-5 deg)": int(((ang >= SE_UNUSED_DEG) & (ang < SE_USED_DEG)).sum()),
                               "unused (<1 deg)": int((ang < SE_UNUSED_DEG).sum())})
    usage = pd.DataFrame(usage_rows)

    # Per-unit detail at the latest matched step.
    a_l = ck(lf, latest)
    a_r = ck(rf, latest)
    leaky_steps = [s for s in sorted(lf) if s > 0]
    weak_hist = units[(units.run == "leaky") & (units.step > 0)]
    for b in range(blocks):
        fc2 = f"blocks.{b}.se_scalebias.fc2.weight"
        fc1 = f"blocks.{b}.se_scalebias.fc1.weight"
        ang2_l = angles(a_l, fresh, fc2, 0)
        ang2_r = angles(a_r, fresh, fc2, 0)
        ang1_l = angles(a_l, fresh, fc1, 1)
        ang1_r = angles(a_r, fresh, fc1, 1)
        site = f"blocks.{b}.se.fc1"
        latest_rows = units[(units.run == "leaky") & (units.step == latest) & (units.site == site)].set_index("index")
        relu_rows = units[(units.run == "relu") & (units.step == latest) & (units.site == site)].set_index("index")
        hist = weak_hist[weak_hist.site == site].groupby("index")["weak"].apply(lambda s: float(s.astype(float).mean()))
        bias_l = a_l[f"blocks.{b}.se_scalebias.fc1.bias"]
        bias_r = a_r[f"blocks.{b}.se_scalebias.fc1.bias"]
        for u in range(ang2_l.shape[0]):
            def cls(a):
                return "used" if a >= SE_USED_DEG else ("trickle" if a >= SE_UNUSED_DEG else "unused")
            detail_rows.append({
                "block": b, "unit": u,
                "leaky FC2-col deg": ang2_l[u], "ReLU FC2-col deg": ang2_r[u],
                "leaky class": cls(ang2_l[u]), "ReLU class": cls(ang2_r[u]),
                "leaky FC1-row deg": ang1_l[u], "ReLU FC1-row deg": ang1_r[u],
                "leaky FC1 bias": bias_l[u], "ReLU FC1 bias": bias_r[u],
                "leaky v/p90": latest_rows.loc[u, "v_rel_p90"],
                f"leaky weak share of {len(leaky_steps)} ckpts": hist.loc[u],
                "leaky unmoved": "unmoved" in latest_rows.loc[u, "flags"],
                "ReLU unmoved": "unmoved" in relu_rows.loc[u, "flags"],
            })
    detail = pd.DataFrame(detail_rows)
    return usage, detail


def weak_trend(units, steps):
    rows = []
    for run in ("leaky", "relu_s2"):
        x = units[(units.run == run) & (units.step > 0)]
        for step in sorted(x.step.unique()):
            if run == "leaky" and step not in steps and step % 1000:
                continue
            row = {"run": run, "step": step}
            for b in range(3):
                g = x[(x.step == step) & (x.site == f"blocks.{b}.se.fc1")]
                row[f"b{b} vel0"] = int(has_flag(g["flags"], "vel0").sum())
                row[f"b{b} weak"] = int(g.weak.astype(bool).sum())
                row[f"b{b} <5% median"] = int((has_flag(g["flags"], "vel_low") | has_flag(g["flags"], "vel0")).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def bn_tables(units, latest, s2_step):
    rows = []
    hot_rows = []
    for run, step in (("leaky", latest), ("relu", latest), ("relu_s2", s2_step)):
        x = units[(units.run == run) & (units.step == step) & units.gamma.notna()]
        for site, g in x.groupby("site", sort=False):
            r = g["beta_over_absgamma"]
            row = {"site": site, "run": RUN_LABEL[run], "next op": g.next_op.iloc[0],
                   "|gamma| min": g.gamma.abs().min(), "|gamma| max": g.gamma.abs().max(),
                   "beta/|gamma| min": r.min(), "p5": r.quantile(0.05), "median": r.median(),
                   "p95": r.quantile(0.95), "max": r.max()}
            if g.next_op.iloc[0] == "relu":
                row["P(on) min"] = L.phi(r.min())
                row["P(on) max"] = L.phi(r.max())
            if g.running_var.notna().any():
                rv = g.running_var
                energy = rv + g.running_mean ** 2
                share = energy / energy.sum()
                top = int(g["index"].iloc[int(np.argmax(share.values))])
                row["rv max/median"] = rv.max() / rv.median()
                row["rv max ch"] = int(g["index"].iloc[int(np.argmax(rv.values))])
                row["rv min/median"] = rv.min() / rv.median()
                row["hottest ch"] = top
                row["hottest energy %"] = 100 * share.max()
                snr = (g.running_mean.abs() / np.sqrt(g.running_var)).values
                near_constant = g["index"].values[snr > 3.0]
                row["near-constant ch (|mean|/std > 3)"] = " ".join(str(int(c)) for c in near_constant) if len(near_constant) else "none"
                hot_rows.append({"site": site, "run": RUN_LABEL[run], "step": step, "channel": top,
                                 "energy share %": 100 * share.max(), "running_mean": float(g.running_mean.iloc[int(np.argmax(share.values))]),
                                 "running_var": float(rv.iloc[int(np.argmax(share.values))]),
                                 "rv / median rv": float(rv.iloc[int(np.argmax(share.values))] / rv.median()),
                                 "uniform %": 100 / len(g)})
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(hot_rows)


def rezero_table(units):
    x = units[units.site.str.endswith("rezero_alpha")]
    rows = []
    for (run, step), g in x.groupby(["run", "step"]):
        row = {"run": RUN_LABEL[run], "step": step}
        for _, r in g.iterrows():
            b = int(re.match(r"blocks\.(\d)", r.site).group(1))
            raw = r.value
            row[f"b{b} raw"] = raw
            row[f"b{b} eff"] = L.rezero_effective(raw, 0.4472136)
            row[f"b{b} d(eff)/d(raw)"] = 1.0 / math.cosh(raw / 0.4472136) ** 2
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["run", "step"])


def persistence_table(units):
    x = units[(units.run == "leaky") & (units.step > 0) & units.v_norm.notna()]
    n = x.step.nunique()
    g = x.groupby(["site", "index", "label"], dropna=False).agg(
        weak_share=("weak", lambda s: float(s.astype(float).mean())),
        vel_low_share=("flags", lambda s: float((has_flag(s, "vel_low") | has_flag(s, "vel0")).mean())),
        v_rel_p90_median=("v_rel_p90", "median"),
        v_rel_p90_max=("v_rel_p90", "max")).reset_index()
    g = g[(g.weak_share >= 0.5) | (g.vel_low_share >= 0.5)]
    return g.sort_values(["site", "index"]), n


def arm_similarity(steps):
    lf, _, _ = L.discover("leaky")
    rf, _, _ = L.discover("relu")
    fresh = L.Checkpoint(lf[0])
    rows = []
    for step in steps:
        a, b = L.Checkpoint(lf[step]), L.Checkpoint(rf[step])
        for name in sorted(n for n in a.tensors if not n.startswith("opt.") and "running" not in n):
            x, y, f = a[name].ravel(), b[name].ravel(), fresh[name].ravel()
            dx, dy = x - f, y - f
            nd = np.linalg.norm(dx) * np.linalg.norm(dy)
            rows.append({"step": step, "tensor": name,
                         "cos(leaky, ReLU)": float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))) if np.linalg.norm(x) * np.linalg.norm(y) > 0 else float("nan"),
                         "cos(leaky - fresh, ReLU - fresh)": float(dx @ dy / nd) if nd > 0 else float("nan"),
                         "|leaky - ReLU| / |ReLU - fresh|": float(np.linalg.norm(x - y) / np.linalg.norm(dy)) if np.linalg.norm(dy) > 0 else float("nan")})
    return pd.DataFrame(rows)


def tensor_summary(latest, s2_step):
    t = pd.read_csv(os.path.join(R, "tensors.csv"))
    nonfinite = t.groupby("run")["nonfinite"].sum()
    sel = t[((t.run.isin(["leaky", "relu"])) & (t.step == latest)) | ((t.run == "relu_s2") & (t.step == s2_step))]
    params = sel[~sel.tensor.str.startswith("opt.")]
    wide = params.pivot_table(index="tensor", columns="run", values=["abs_max", "rms"]).reset_index()
    wide.columns = [" ".join(c).strip() if isinstance(c, tuple) else c for c in wide.columns]
    vel = sel[sel.tensor.str.startswith("opt.")].pivot_table(index="tensor", columns="run", values=["abs_max", "rms", "zero_frac"]).reset_index()
    vel.columns = [" ".join(c).strip() if isinstance(c, tuple) else c for c in vel.columns]
    grid = t[~t.tensor.str.startswith("opt.") & (t.step > 0)].groupby("run")["bf16_grid_frac"].agg(["min", "max"]).reset_index()
    return nonfinite, wide, vel, grid


def velocity_outliers(units, latest, s2_step):
    rows = []
    for run, step in (("leaky", latest), ("relu_s2", s2_step)):
        x = units[(units.run == run) & (units.step == step) & units.v_norm.notna()]
        for site, g in x.groupby("site", sort=False):
            med = g.v_norm.median()
            i = int(np.argmax(g.v_norm.values))
            rows.append({"run": RUN_LABEL[run], "site": site, "units": len(g),
                         "max/median": g.v_norm.max() / med if med > 0 else float("nan"),
                         "argmax": int(g["index"].iloc[i]), "label": g["label"].iloc[i] if isinstance(g["label"].iloc[i], str) else "",
                         "min/p90": g.v_rel_p90.min(), "weak": int(g.weak.astype(bool).sum()),
                         "vel0": int(has_flag(g["flags"], "vel0").sum())})
    return pd.DataFrame(rows)


def head_unit_tables(units, latest):
    """Per value hidden unit and per policy channel: leaky vs ReLU s1 movement
    (same init) plus leaky velocity persistence."""
    lf, _, _ = L.discover("leaky")
    rf, _, _ = L.discover("relu")
    fresh, a_l, a_r = L.Checkpoint(lf[0]), L.Checkpoint(lf[latest]), L.Checkpoint(rf[latest])
    f_l = float(np.linalg.norm(a_l["stem.conv.weight"][:, L.ALWAYS_ZERO_PLANES]) / np.linalg.norm(fresh["stem.conv.weight"][:, L.ALWAYS_ZERO_PLANES]))
    f_r = float(np.linalg.norm(a_r["stem.conv.weight"][:, L.ALWAYS_ZERO_PLANES]) / np.linalg.norm(fresh["stem.conv.weight"][:, L.ALWAYS_ZERO_PLANES]))
    hist = units[(units.run == "leaky") & (units.step > 0)]

    def row_angle(c, name, axis):
        a, b = c[name], fresh[name]
        a2 = a.reshape(a.shape[0], -1) if axis == 1 else a.reshape(a.shape[0], -1).T
        b2 = b.reshape(b.shape[0], -1) if axis == 1 else b.reshape(b.shape[0], -1).T
        cos = (a2 * b2).sum(1) / np.linalg.norm(a2, axis=1) / np.linalg.norm(b2, axis=1)
        return np.degrees(np.arccos(np.clip(cos, -1, 1))), np.linalg.norm(a2, axis=1) / np.linalg.norm(b2, axis=1)

    def persistence(site):
        g = hist[hist.site == site].groupby("index")
        return g["v_rel_p90"].median(), g["weak"].apply(lambda s: float(s.astype(float).mean()))

    v_rows = []
    fc1_l, _ = row_angle(a_l, "value.fc1.weight", 1)
    fc1_r, _ = row_angle(a_r, "value.fc1.weight", 1)
    _, fc2n_l = row_angle(a_l, "value.wdl_fc2.weight", 0)
    _, fc2n_r = row_angle(a_r, "value.wdl_fc2.weight", 0)
    med_in, weak_in = persistence("value.wdl_fc2.in")
    med_row, _ = persistence("value.fc1")
    for u in range(fc1_l.shape[0]):
        v_rows.append({"unit": u, "leaky fc1-row deg": fc1_l[u], "ReLU fc1-row deg": fc1_r[u],
                       "leaky fc2-col norm/decay": fc2n_l[u] / f_l, "ReLU fc2-col norm/decay": fc2n_r[u] / f_r,
                       "leaky bias": a_l["value.fc1.bias"][u], "ReLU bias": a_r["value.fc1.bias"][u],
                       "leaky fc2-col v/p90 (median over ckpts)": med_in.loc[u],
                       "leaky fc1-row v/p90 (median)": med_row.loc[u],
                       "leaky fc2-col weak share": weak_in.loc[u]})
    vdf = pd.DataFrame(v_rows).sort_values("leaky fc2-col v/p90 (median over ckpts)")
    labels = _policy_labels()
    p_rows = []
    ang_l, nrm_l = row_angle(a_l, "policy.conv.weight", 1)
    ang_r, nrm_r = row_angle(a_r, "policy.conv.weight", 1)
    med_out, weak_out = persistence("policy.conv.out")
    med_b, weak_b = persistence("policy.conv.bias")
    for ch in range(ang_l.shape[0]):
        p_rows.append({"channel": labels[ch], "leaky row deg": ang_l[ch], "ReLU row deg": ang_r[ch],
                       "leaky row norm/decay": nrm_l[ch] / f_l, "ReLU row norm/decay": nrm_r[ch] / f_r,
                       "leaky bias": a_l["policy.conv.bias"][ch], "ReLU bias": a_r["policy.conv.bias"][ch],
                       "leaky row v/p90 (median over ckpts)": med_out.loc[ch], "leaky row weak share": weak_out.loc[ch],
                       "leaky bias weak share": weak_b.loc[ch]})
    pdf = pd.DataFrame(p_rows).sort_values("leaky row v/p90 (median over ckpts)")
    return vdf, pdf


def _policy_labels():
    sys.path.insert(0, os.path.abspath(os.path.join(L.HERE, "..", "..", "..", "..",
                                                     "documentation", "research", "policy-head-2026-10-01", "scripts")))
    import policy_head_lib  # noqa: E402
    return [f"{c} {policy_head_lib.channel_label(c)}" for c in range(76)]


def se_overlap(detail):
    rows = []
    for b, g in detail.groupby("block"):
        used_l = set(g[g["leaky class"] == "used"].unit)
        used_r = set(g[g["ReLU class"] == "used"].unit)
        rows.append({"block": b, "used leaky": len(used_l), "used ReLU": len(used_r), "used in both": len(used_l & used_r),
                     "leaky only": " ".join(map(str, sorted(used_l - used_r))), "ReLU only": " ".join(map(str, sorted(used_r - used_l))),
                     "Spearman(leaky deg, ReLU deg)": L.spearman(g["leaky FC2-col deg"].values, g["ReLU FC2-col deg"].values)})
    return pd.DataFrame(rows)


def dead_turnover(units):
    """ReLU s2 (velocity saved): which SE FC1 units have exactly-zero velocity at
    each checkpoint, and which entered / left that set since the previous one."""
    x = units[(units.run == "relu_s2") & (units.step > 0)]
    rows = []
    for b in range(3):
        s = x[x.site == f"blocks.{b}.se.fc1"]
        previous = set()
        for step, g in s.groupby("step"):
            dead = set(int(i) for i in g[has_flag(g["flags"], "vel0")]["index"])
            rows.append({"block": b, "step": step, "vel0 units": " ".join(map(str, sorted(dead))), "count": len(dead),
                         "newly dead": " ".join(map(str, sorted(dead - previous))),
                         "revived (left the set)": " ".join(map(str, sorted(previous - dead)))})
            previous = dead
    return pd.DataFrame(rows)


def main():
    units = load()
    leaky_steps = sorted(s for s in units[units.run == "leaky"].step.unique())
    latest = max(leaky_steps)
    s2_steps = sorted(s for s in units[units.run == "relu_s2"].step.unique())
    s2_step = max(s2_steps)
    matched = [s for s in (1000, 5000, 10000, 15000, 20000) if s in leaky_steps] + [latest]
    header = (f"Generated by `scripts/summarize.py` from `units.csv.gz`. leaky / ReLU s1 at step {latest} "
              f"(same init, ModelIDs in manifest.csv); ReLU s2 at step {s2_step} (its last checkpoint).\n")

    ct = counts_table(units, latest, s2_step)
    write("counts_latest.md", "# Per-site flag counts at the latest step\n\n" + header +
          "\nCells are `leaky / ReLU s1 / ReLU s2`; `-` = not measurable (no velocity saved); `.` = zero in all three.\n\n" +
          md_table(ct))
    cbs = counts_by_step(units, [s for s in leaky_steps if s > 0], [s for s in s2_steps if s > 0])
    cbs.to_csv(os.path.join(R, "counts_by_step.csv"), index=False)
    # Trend table: every nonzero (site, flag) at matched steps.
    trend = cbs[cbs.step.isin(matched)]
    keys = trend.groupby(["site", "flag"])["count"].sum()
    keys = keys[keys > 0].index
    rows = []
    for site, flag in keys:
        row = {"site": site, "flag": flag}
        for step in matched:
            for run in ("leaky", "relu"):
                v = trend[(trend.site == site) & (trend.flag == flag) & (trend.run == run) & (trend.step == step)]["count"]
                row[f"{RUN_LABEL[run]} {step // 1000}k"] = int(v.iloc[0]) if len(v) else None
        rows.append(row)
    tt = pd.DataFrame(rows)
    order = {s: i for i, s in enumerate(site_order(units))}
    tt = tt.sort_values(["site", "flag"], key=lambda c: c.map(order) if c.name == "site" else c)
    write("counts_trend.md", "# Flag counts at matched steps, leaky vs ReLU s1\n\n" + header +
          "\nEvery (site, flag) that is nonzero at any matched step. Blank = not measurable for that run "
          "(ReLU s1 saves no velocity). Full per-step data: `counts_by_step.csv`.\n\n" + md_table(tt))

    usage, detail = se_unit_tables(units, latest, s2_step, [s for s in leaky_steps if s > 0])
    usage.to_csv(os.path.join(R, "se_usage_by_step.csv"), index=False)
    piv = usage[usage.run.isin(["leaky", "relu"]) & usage.step.isin(matched)].copy()
    piv["cell"] = piv["used (>=5 deg)"].astype(str) + " / " + piv["trickle (1-5 deg)"].astype(str) + " / " + piv["unused (<1 deg)"].astype(str)
    piv = piv.pivot_table(index=["block", "step"], columns="run", values="cell", aggfunc="first").reset_index()
    s2u = usage[usage.run == "relu_s2"].copy()
    s2u["cell"] = s2u["used (>=5 deg)"].astype(str) + " / " + s2u["trickle (1-5 deg)"].astype(str) + " / " + s2u["unused (<1 deg)"].astype(str)
    s2u = s2u.pivot_table(index="step", columns="block", values="cell", aggfunc="first").reset_index()
    s2u.columns = ["step"] + [f"block {b}" for b in s2u.columns[1:]]
    detail.to_csv(os.path.join(R, "se_units_latest.csv"), index=False)
    write("se_units.md", "# SE bottleneck: per-unit usage, leaky vs ReLU s1 (same init)\n\n" + header +
          f"\nUsage class from the FC2 input-column angle vs fresh (used >= {SE_USED_DEG:g} deg, trickle "
          f"{SE_UNUSED_DEG:g}-{SE_USED_DEG:g}, unused < {SE_UNUSED_DEG:g}). Cells `used / trickle / unused`.\n\n"
          "## Usage counts at matched steps (seed 1 arms)\n\n" + md_table(piv) +
          "\n\n## ReLU s2 usage counts by step (different init; for scale)\n\n" + md_table(s2u) +
          "\n\n## Every unit at the latest step\n\n`leaky v/p90` = FC1 weight-row velocity norm / block 90th percentile. "
          "`weak share` = fraction of leaky checkpoints (1k..latest) where the unit was weak (< 5% of p90).\n\n" +
          md_table(detail) + "\n\n## Same units? (used = FC2-column angle >= 5 deg at the latest step)\n\n" + md_table(se_overlap(detail)))
    vdf, pdf = head_unit_tables(units, latest)
    vdf.to_csv(os.path.join(R, "value_hidden_units.csv"), index=False)
    pdf.to_csv(os.path.join(R, "policy_channels.csv"), index=False)
    write("value_hidden_units.md", "# Value head: every FC1 hidden unit, leaky vs ReLU s1 (same init)\n\n" + header +
          "\n`fc2-col` = the 3 WDL weights reading the unit; its gradient is proportional to the unit's output, so its "
          "velocity measures how often/strongly the unit fires. `norm/decay` = norm ratio to fresh divided by the "
          "decay-only factor (1 = decay only). `v/p90` medians are over the leaky checkpoints 1k..latest. Sorted by usage.\n\n" +
          md_table(vdf))
    write("policy_channels.md", "# Policy head: every final-conv channel, leaky vs ReLU s1 (same init)\n\n" + header +
          "\nSorted by leaky row velocity (median over checkpoints, / site p90).\n\n" + md_table(pdf))
    wt = weak_trend(units, matched)
    write("se_weak_trend.md", "# SE FC1 velocity classes by step (velocity runs)\n\n" + header +
          "\n`vel0` = exactly zero velocity; `weak` = < 5% of the block's p90; `<5% median` = the README's earlier "
          "measure (vel0 included).\n\n" + md_table(wt))

    write("se_relu_s2_dead_turnover.md", "# ReLU s2: SE FC1 units with exactly-zero velocity, per checkpoint\n\n" + header +
          "\nA unit leaves the set when its pre-activation turns positive for some inputs again -- possible under ReLU "
          "because FC1's input (the pooled conv2 output) keeps changing even when the unit's own weights get no gradient.\n\n" +
          md_table(dead_turnover(units)))
    bn, hot = bn_tables(units, latest, s2_step)
    write("bn_ln.md", "# BN and LayerNorm channels\n\n" + header +
          "\n`P(on)` = Phi(beta/|gamma|) for BNs followed by ReLU (Gaussian-input approximation). "
          "LayerNorm rows: gamma/beta of the per-position LN at each block output (no running stats, no activation).\n\n" +
          md_table(bn) + "\n\n## Hottest (largest E[x^2]) channel per BN input\n\n" + md_table(hot))
    write("rezero.md", "# ReZero alpha by step\n\n" + header +
          "\n`eff` = C tanh(raw/C), C = alpha init = 0.4472136. `d(eff)/d(raw)` = sech^2(raw/C): the gradient "
          "attenuation at the current raw value.\n\n" + md_table(rezero_table(units), "{:.5f}"))
    pers, n = persistence_table(units)
    pers.to_csv(os.path.join(R, "persistent_low_velocity.csv"), index=False)
    write("persistent_low_velocity.md", "# Leaky units with persistently low velocity\n\n" + header +
          f"\nUnits (any site) that were weak (< 5% of site p90) or below 5% of the site median in at least half "
          f"of the {n} leaky checkpoints with velocity. `v_rel_p90_*` = velocity / site p90 across those checkpoints.\n\n" +
          md_table(pers))
    sim = arm_similarity(matched)
    sim.to_csv(os.path.join(R, "arm_similarity.csv"), index=False)
    write("arm_similarity.md", "# How far apart are the leaky and ReLU s1 arms?\n\n" + header +
          "\nPer tensor at matched steps. Both arms start from bit-identical weights.\n\n" + md_table(sim[sim.step == latest]))
    nonfinite, wide, vel, grid = tensor_summary(latest, s2_step)
    vo = velocity_outliers(units, latest, s2_step)
    write("tensors_summary.md", "# Tensor-level checks\n\n" + header +
          "\n## Non-finite values (all checkpoints analysed, incl. optimizer state)\n\n" +
          md_table(nonfinite.reset_index()) +
          "\n\n## Share of stored parameter values on the bf16 grid (trained checkpoints)\n\n" + md_table(grid) +
          "\n\n## Parameters: max |value| and RMS\n\n" + md_table(wide) +
          "\n\n## Velocity: max |v|, RMS, exact-zero share\n\n" + md_table(vel) +
          "\n\n## Per-site velocity spread (max/median, min/p90)\n\n" + md_table(vo))

    # Every flagged unit at the latest step (all runs).
    lat = units[((units.run.isin(["leaky", "relu"])) & (units.step == latest)) | ((units.run == "relu_s2") & (units.step == s2_step))]
    flagged = lat[(lat["flags"] != "") | lat["weak"]]
    cols = ["run", "step", "site", "index", "label", "flags", "weak", "v_rel_p90", "v_rel_median", "cos_fresh",
            "ratio_over_decay", "gamma", "beta", "beta_over_absgamma", "running_var", "rv_rel_median", "value", "value_fresh"]
    flagged[cols].to_csv(os.path.join(R, "flagged_units_latest.csv"), index=False)
    print(f"flagged rows at latest: {len(flagged)}", file=sys.stderr)


if __name__ == "__main__":
    main()
