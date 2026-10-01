"""Per-checkpoint static analysis of the DCM policy head.

`analyze(path, reference_path=None)` returns a dict with
  - identity (metadata, file sha256, architecture summary),
  - per-tensor statistics (every policy tensor, velocities included),
  - per pre-block channel arrays (K rows) and per move-type channel arrays
    (76 rows) for the full detail tables,
  - scalar summaries used by the trajectory and cross-lineage tables.

Definitions (all weights-only, no forward pass):
  * Pre-block channel k (intermediate_conv): post-BN value ~ N(beta_k,
    gamma_k^2) under the assumption that BN normalizes its input exactly.
    a_k = beta_k / |gamma_k|. For ReLU, P(on) = Phi(a_k); dead: a_k < -3
    (P(on) < 0.13%), mostly-off: a_k < -2 (P(on) < 2.3%). For leaky ReLU
    (slope 0.01) the same thresholds mark channels that are "on" equally
    rarely, but such channels still pass 1% of the negative side.
    "Flat" channel: |gamma_k| < 0.05 * median |gamma| — its output is
    nearly constant (beta), so it carries almost no position information
    whatever its sign.
  * Post-activation moments E[a], Var[a], E[a^2] by Gauss-Hermite.
  * Static logit level of move type c: L_c = b_c + sum_k W[c,k] E[a_k] —
    the exact mean of logit c over positions and squares if each BN output
    is N(beta, gamma^2) (only the marginals are assumed; linearity does the
    rest). mean_c L_c is the static estimate of the shared policy offset.
  * Mean row m = mean_c W[c,:]; shared-row ratio = ||m|| / mean_c ||W[c]||;
    residual norm = ||W[c] - m||.
  * Shared-row feature-rounding noise: when the K features are rounded to
    bf16 before the final projection (the default mixed_final_projection
    tail), each square's logits all receive m . (a * eps), eps ~ relative
    rounding error (std ~ 2^-8/sqrt(3)). This does not cancel in softmax
    because it differs per square. Std = 2^-8/sqrt(3) * sqrt(sum m_k^2 E[a_k^2]),
    assuming independent rounding errors.
  * BN cancellation risk: |running_mean| / sqrt(running_var); the bf16 noise
    of the normalized value from rounding the BN input is about
    (|mean| + std) / std * 2^-9 (in units of the normalized std).
"""
import json
import math
import os

import numpy as np

import policy_head_lib as lib

ROUNDING_REL_STD = lib.BF16_UNIT_ROUNDOFF / math.sqrt(3.0)


def tensor_stats(array, raw_bits=None, reference=None):
    flat = np.asarray(array, dtype=np.float64).ravel()
    finite = flat[np.isfinite(flat)]
    stats = dict(shape=list(np.shape(array)), count=int(flat.size),
                 nonfinite=lib.nonfinite_count(flat),
                 min=float(finite.min()), max=float(finite.max()),
                 max_abs=float(np.abs(finite).max()), mean=float(finite.mean()),
                 std=float(finite.std()), l2=float(np.linalg.norm(finite)),
                 exact_zero_fraction=float(np.mean(flat == 0.0)),
                 bf16_exact_fraction=lib.bf16_exact_fraction(raw_bits))
    if reference is not None and np.shape(reference) == np.shape(array):
        ref = np.asarray(reference, dtype=np.float64).ravel()
        stats["equal_to_reference_fraction"] = float(np.mean(flat == ref))
        denominator = np.linalg.norm(ref)
        stats["relative_change_vs_reference"] = (float(np.linalg.norm(flat - ref) / denominator)
                                                 if denominator > 0 else None)
        stats["norm_ratio_vs_reference"] = (float(np.linalg.norm(flat) / denominator)
                                            if denominator > 0 else None)
    return stats


def row_cosine(a, b):
    numerator = (a * b).sum(1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def analyze(path, reference_path=None, compute_file_sha256=True):
    metadata, tensors, raw_bits = lib.read_policy(path)
    architecture = json.loads(metadata["architecture"])
    reference = None
    if reference_path is not None and os.path.abspath(reference_path) != os.path.abspath(path):
        reference_metadata, reference, _ = lib.read_policy(reference_path)
    style = architecture["policy_head_style"]
    if style == "fc_bottleneck":
        raise NotImplementedError("fc_bottleneck heads are not present in any saved checkpoint")
    activation = architecture["activation_function"]

    result = dict(path=path, model_id=metadata["model_id"],
                  parent_model_id=metadata.get("parent_model_id", ""),
                  training_step=int(metadata["training_step"]) if metadata.get("training_step") not in (None, "") else None,
                  creator=metadata.get("creator", ""), created_at_unix=metadata.get("created_at_unix"),
                  content_sha256=metadata.get("content_sha256"), file_sha256=lib.sha256_of_file(path) if compute_file_sha256 else None,
                  architecture=architecture, arch_summary=lib.architecture_summary(architecture),
                  policy_style=style, activation=activation,
                  compute_dtype=architecture.get("compute_data_type"),
                  reference_path=reference_path if reference is not None else None,
                  reference_model_id=reference_metadata["model_id"] if reference is not None else None,
                  reference_step=(reference_metadata.get("training_step") or "fresh") if reference is not None else None)

    W = tensors["policy.conv.weight"].reshape(lib.POLICY_CHANNELS, -1)
    b = tensors["policy.conv.bias"].reshape(-1)
    K = W.shape[1]
    result["K"] = K
    result["input_width"] = lib.policy_input_width(architecture, tensors)
    result["has_velocity"] = any(name.startswith("opt.") for name in tensors)

    # ------------------------------------------------ every tensor
    result["tensors"] = {}
    for name in sorted(tensors):
        ref = reference.get(name) if reference is not None else None
        result["tensors"][name] = tensor_stats(tensors[name], raw_bits.get(name), ref)
    result["nonfinite_total"] = sum(s["nonfinite"] for s in result["tensors"].values())

    # ------------------------------------------------ pre-block
    pre = None
    if style == "intermediate_conv":
        Wp = tensors["policy.pre_conv.weight"].reshape(K, -1)
        gamma = tensors["policy.pre_bn.weight"].reshape(-1)
        beta = tensors["policy.pre_bn.bias"].reshape(-1)
        running_mean = tensors["policy.pre_bn.running_mean"].reshape(-1)
        running_var = tensors["policy.pre_bn.running_var"].reshape(-1)
        abs_gamma = np.abs(gamma)
        if np.any(abs_gamma == 0):
            raise ValueError(f"{path}: policy.pre_bn.weight has exact zeros; beta/|gamma| undefined")
        on_ratio = beta / abs_gamma
        p_on = lib.standard_normal_cdf(on_ratio)
        act_mean, act_var, act_second = lib.post_activation_moments(gamma, beta, activation)
        act_std = np.sqrt(act_var)
        pre_row_norm = np.linalg.norm(Wp, axis=1)
        pre_col_norm = np.linalg.norm(Wp, axis=0)
        final_col_norm = np.linalg.norm(W, axis=0)
        median_abs_gamma = float(np.median(abs_gamma))
        mean_over_std = np.abs(running_mean) / np.sqrt(np.maximum(running_var, 1e-30))
        pre = dict(gamma=gamma, beta=beta, running_mean=running_mean, running_var=running_var,
                   on_ratio=on_ratio, p_on=p_on, act_mean=act_mean, act_std=act_std,
                   act_second=act_second, pre_row_norm=pre_row_norm,
                   final_col_norm=final_col_norm,
                   effective_contribution=final_col_norm * act_std,
                   mean_over_std=mean_over_std,
                   bn_input_rounding_noise=(mean_over_std + 1.0) * 2.0 ** -9,
                   dead=on_ratio < -3.0, mostly_off=(on_ratio >= -3.0) & (on_ratio < -2.0),
                   flat=abs_gamma < 0.05 * median_abs_gamma,
                   always_on=on_ratio > 3.0,
                   negative_gamma=gamma < 0,
                   mean_row_component=W.mean(0),
                   shared_contribution=W.mean(0) * act_mean,
                   column_shared_fraction=np.abs(W.mean(0)) * math.sqrt(lib.POLICY_CHANNELS) / np.maximum(final_col_norm, 1e-30))
        if reference is not None and "policy.pre_conv.weight" in reference:
            Wp0 = reference["policy.pre_conv.weight"].reshape(K, -1)
            pre["pre_row_rel_change"] = np.linalg.norm(Wp - Wp0, axis=1) / np.maximum(np.linalg.norm(Wp0, axis=1), 1e-30)
            pre["pre_row_cos_ref"] = row_cosine(Wp, Wp0)
            pre["pre_row_norm_ratio_ref"] = pre_row_norm / np.maximum(np.linalg.norm(Wp0, axis=1), 1e-30)
            pre["gamma_delta"] = gamma - reference["policy.pre_bn.weight"].reshape(-1)
            pre["beta_delta"] = beta - reference["policy.pre_bn.bias"].reshape(-1)
            pre["final_col_rel_change"] = (np.linalg.norm(W - reference["policy.conv.weight"].reshape(76, -1), axis=0)
                                           / np.maximum(np.linalg.norm(reference["policy.conv.weight"].reshape(76, -1), axis=0), 1e-30))
        if "opt.policy.pre_conv.weight.velocity" in tensors:
            Vp = tensors["opt.policy.pre_conv.weight.velocity"].reshape(K, -1)
            pre["pre_row_velocity_norm"] = np.linalg.norm(Vp, axis=1)
            pre["pre_row_velocity_cos_w"] = row_cosine(Vp, Wp)
            pre["gamma_velocity"] = tensors["opt.policy.pre_bn.weight.velocity"].reshape(-1)
            pre["beta_velocity"] = tensors["opt.policy.pre_bn.bias.velocity"].reshape(-1)
        if "opt.policy.conv.weight.velocity" in tensors:
            Vc = tensors["opt.policy.conv.weight.velocity"].reshape(76, K)
            pre["final_col_velocity_norm"] = np.linalg.norm(Vc, axis=0)
        result["pre"] = {key: [float(x) for x in np.asarray(value, dtype=np.float64)] for key, value in pre.items()}
        result["pre_summary"] = dict(
            dead=int(pre["dead"].sum()), mostly_off=int(pre["mostly_off"].sum()), flat=int(pre["flat"].sum()),
            negative_gamma=int(pre["negative_gamma"].sum()),
            dead_channels=[int(k) for k in np.nonzero(pre["dead"])[0]],
            mostly_off_channels=[int(k) for k in np.nonzero(pre["mostly_off"])[0]],
            flat_channels=[int(k) for k in np.nonzero(pre["flat"])[0]],
            gamma=lib.percentile_summary(gamma), abs_gamma=lib.percentile_summary(abs_gamma),
            beta=lib.percentile_summary(beta), on_ratio=lib.percentile_summary(on_ratio[np.isfinite(on_ratio)]),
            p_on=lib.percentile_summary(p_on),
            running_mean=lib.percentile_summary(running_mean), running_var=lib.percentile_summary(running_var),
            hottest_ratio=float(running_var.max() / running_var.mean()),
            rv_max_over_median=float(running_var.max() / np.median(running_var)),
            rv_span=float(running_var.max() / running_var.min()),
            always_on=int(pre["always_on"].sum()),
            always_on_channels=[int(k) for k in np.nonzero(pre["always_on"])[0]],
            shared_level_from_always_on=float(pre["shared_contribution"][pre["always_on"]].sum()),
            top_shared_contributors=[dict(k=int(k), contribution=float(pre["shared_contribution"][k]),
                                          beta=float(beta[k]), gamma=float(gamma[k]),
                                          running_var=float(running_var[k]),
                                          final_col_norm=float(final_col_norm[k]),
                                          column_shared_fraction=float(pre["column_shared_fraction"][k]))
                                     for k in np.argsort(-np.abs(pre["shared_contribution"]))[:5]],
            hottest_channel=int(np.argmax(running_var)),
            coldest_ratio=float(running_var.min() / running_var.mean()),
            coldest_channel=int(np.argmin(running_var)),
            mean_over_std=lib.percentile_summary(mean_over_std),
            worst_mean_over_std_channel=int(np.argmax(mean_over_std)),
            pre_row_norm=lib.percentile_summary(pre_row_norm),
            pre_row_near_zero=[int(k) for k in np.nonzero(pre_row_norm < 0.05 * np.median(pre_row_norm))[0]],
            pre_col_norm=lib.percentile_summary(pre_col_norm),
            pre_col_weak=[int(k) for k in np.nonzero(pre_col_norm < 0.1 * np.median(pre_col_norm))[0]],
            act_std=lib.percentile_summary(act_std),
            effective_contribution=lib.percentile_summary(pre["effective_contribution"]),
        )
        if "gamma_velocity" in pre:
            zero_both = (pre["gamma_velocity"] == 0) & (pre["beta_velocity"] == 0)
            zero_row = pre["pre_row_velocity_norm"] == 0
            result["pre_summary"].update(
                zero_velocity_gamma_beta_channels=[int(k) for k in np.nonzero(zero_both)[0]],
                zero_velocity_pre_rows=[int(k) for k in np.nonzero(zero_row)[0]],
                pre_row_velocity_norm=lib.percentile_summary(pre["pre_row_velocity_norm"]),
                pre_row_velocity_cos_w=lib.percentile_summary(np.nan_to_num(pre["pre_row_velocity_cos_w"])),
                gamma_velocity_abs=lib.percentile_summary(np.abs(pre["gamma_velocity"])),
                beta_velocity_abs=lib.percentile_summary(np.abs(pre["beta_velocity"])))
        if "pre_row_rel_change" in pre:
            result["pre_summary"].update(
                pre_row_rel_change=lib.percentile_summary(pre["pre_row_rel_change"]),
                pre_row_cos_ref=lib.percentile_summary(pre["pre_row_cos_ref"]),
                pre_row_norm_ratio_ref=lib.percentile_summary(pre["pre_row_norm_ratio_ref"]),
                gamma_delta_abs=lib.percentile_summary(np.abs(pre["gamma_delta"])),
                beta_delta_abs=lib.percentile_summary(np.abs(pre["beta_delta"])),
                gamma_beta_unchanged=[int(k) for k in np.nonzero((pre["gamma_delta"] == 0) & (pre["beta_delta"] == 0))[0]])

    # ------------------------------------------------ final projection
    row_norm = np.linalg.norm(W, axis=1)
    mean_row = W.mean(0)
    residual = W - mean_row
    residual_norm = np.linalg.norm(residual, axis=1)
    conv = dict(row_norm=row_norm, bias=b, residual_norm=residual_norm,
                row_cos_mean_row=row_cosine(W, np.broadcast_to(mean_row, W.shape)),
                row_max_abs=np.abs(W).max(1))
    if pre is not None:
        static_level = b + W @ pre["act_mean"]
        conv["static_logit_level"] = static_level
        conv["indep_logit_std_estimate"] = np.sqrt((W ** 2) @ (pre["act_std"] ** 2))
        conv["residual_logit_std_estimate"] = np.sqrt((residual ** 2) @ (pre["act_std"] ** 2))
        conv["feature_rounding_noise"] = ROUNDING_REL_STD * np.sqrt((W ** 2) @ pre["act_second"])
    if reference is not None:
        W0 = reference["policy.conv.weight"].reshape(76, -1)
        b0 = reference["policy.conv.bias"].reshape(-1)
        conv["row_rel_change"] = np.linalg.norm(W - W0, axis=1) / np.maximum(np.linalg.norm(W0, axis=1), 1e-30)
        conv["row_cos_ref"] = row_cosine(W, W0)
        conv["row_norm_ratio_ref"] = row_norm / np.maximum(np.linalg.norm(W0, axis=1), 1e-30)
        conv["bias_delta"] = b - b0
    if "opt.policy.conv.weight.velocity" in tensors:
        Vc = tensors["opt.policy.conv.weight.velocity"].reshape(76, K)
        conv["row_velocity_norm"] = np.linalg.norm(Vc, axis=1)
        conv["row_velocity_cos_w"] = row_cosine(Vc, W)
        conv["bias_velocity"] = tensors["opt.policy.conv.bias.velocity"].reshape(-1)
    result["conv"] = {key: [float(x) for x in np.asarray(value, dtype=np.float64)] for key, value in conv.items()}

    families = lib.channel_families()
    queen_median = float(np.median(row_norm[:56]))
    family_rows = {}
    for family, channels in families.items():
        channels = np.array(channels)
        entry = dict(row_norm_mean=float(row_norm[channels].mean()),
                     row_norm_min=float(row_norm[channels].min()),
                     row_norm_max=float(row_norm[channels].max()),
                     row_norm_vs_queen_median=float(row_norm[channels].mean() / queen_median),
                     bias_mean=float(b[channels].mean()), bias_min=float(b[channels].min()),
                     bias_max=float(b[channels].max()),
                     residual_norm_mean=float(residual_norm[channels].mean()))
        if "static_logit_level" in conv:
            entry["static_logit_level_mean"] = float(conv["static_logit_level"][channels].mean())
        if "row_rel_change" in conv:
            entry["row_rel_change_mean"] = float(conv["row_rel_change"][channels].mean())
            entry["bias_delta_mean"] = float(conv["bias_delta"][channels].mean())
        if "row_velocity_norm" in conv:
            entry["row_velocity_norm_mean"] = float(conv["row_velocity_norm"][channels].mean())
        family_rows[family] = entry
    result["families"] = family_rows

    shared_ratio = float(np.linalg.norm(mean_row) / row_norm.mean())
    summary = dict(row_norm=lib.percentile_summary(row_norm), bias=lib.percentile_summary(b),
                   bias_mean=float(b.mean()), bias_std=float(b.std()),
                   mean_row_norm=float(np.linalg.norm(mean_row)),
                   mean_row_ratio=shared_ratio,
                   residual_norm_median=float(np.median(residual_norm)),
                   mean_row_over_residual_median=float(np.linalg.norm(mean_row) / np.median(residual_norm)),
                   final_col_norm=lib.percentile_summary(np.linalg.norm(W, axis=0)),
                   final_col_weak=[int(k) for k in np.nonzero(np.linalg.norm(W, axis=0) < 0.1 * np.median(np.linalg.norm(W, axis=0)))[0]],
                   weakest_channel=int(np.argmin(row_norm)), strongest_channel=int(np.argmax(row_norm)),
                   underpromo_vs_queen_median=float(row_norm[64:73].mean() / queen_median),
                   knight_vs_queen_median=float(row_norm[56:64].mean() / queen_median),
                   queen_promo_vs_queen_median=float(row_norm[73:76].mean() / queen_median))
    weakest_family = min((f for f in families if f not in ("queen-style all",)),
                         key=lambda f: family_rows[f]["row_norm_mean"])
    summary["weakest_family"] = weakest_family
    summary["weakest_family_vs_queen_median"] = family_rows[weakest_family]["row_norm_vs_queen_median"]
    if pre is not None:
        level = conv["static_logit_level"]
        summary.update(static_shared_level=float(level.mean()), static_level_min=float(level.min()),
                       static_level_max=float(level.max()), static_level_std=float(level.std()),
                       bf16_spacing_at_shared_level=float(lib.bf16_spacing(level.mean())),
                       shared_row_rounding_noise=float(ROUNDING_REL_STD * math.sqrt(float((mean_row ** 2) @ pre["act_second"]))),
                       residual_rounding_noise_median=float(np.median(ROUNDING_REL_STD * np.sqrt((residual ** 2) @ pre["act_second"]))),
                       residual_logit_std_median=float(np.median(conv["residual_logit_std_estimate"])),
                       shared_from_bias=float(b.mean()), shared_from_mean_row=float(mean_row @ pre["act_mean"]))
        dead_or_off = pre["dead"] | pre["mostly_off"]
        weak_cols = np.linalg.norm(W, axis=0) < 0.1 * np.median(np.linalg.norm(W, axis=0))
        summary["weak_col_and_dead_or_off"] = int((weak_cols & dead_or_off).sum())
    # mirror symmetry of biases and row norms
    asym = []
    for channel in range(76):
        mirror = lib.mirror_channel(channel)
        if mirror > channel:
            asym.append(dict(channel=channel, mirror=mirror, label=lib.channel_label(channel),
                             mirror_label=lib.channel_label(mirror),
                             bias_diff=float(b[channel] - b[mirror]),
                             row_norm_ratio=float(row_norm[channel] / row_norm[mirror]),
                             level_diff=float(conv["static_logit_level"][channel] - conv["static_logit_level"][mirror]) if pre is not None else None))
    result["mirror_pairs"] = asym
    if "row_rel_change" in conv:
        summary.update(row_rel_change=lib.percentile_summary(conv["row_rel_change"]),
                       row_cos_ref=lib.percentile_summary(conv["row_cos_ref"]),
                       bias_delta_abs=lib.percentile_summary(np.abs(conv["bias_delta"])))
    if "row_velocity_norm" in conv:
        summary.update(row_velocity_norm=lib.percentile_summary(conv["row_velocity_norm"]),
                       zero_velocity_rows=[int(c) for c in np.nonzero(conv["row_velocity_norm"] == 0)[0]],
                       row_velocity_cos_w=lib.percentile_summary(np.nan_to_num(conv["row_velocity_cos_w"])),
                       bias_velocity_abs=lib.percentile_summary(np.abs(conv["bias_velocity"])),
                       zero_velocity_bias=[int(c) for c in np.nonzero(conv["bias_velocity"] == 0)[0]])
    result["conv_summary"] = summary
    return result
