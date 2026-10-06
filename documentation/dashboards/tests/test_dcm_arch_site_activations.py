"""Tests for the format-v9 site activations in scripts/dcm_arch.py and the scripts that rely on them:
the legacy resolution, the v9 gate, the retired top-level key, both mismatch directions, the
`require_relu` guard, units.py's architecture check, and the keyword-only `sites` / `md` of the
forward passes in fwd16.py and bf16-head-offset/scripts/fwd.py.

Run: python3 -m unittest discover -s documentation/dashboards/tests
Every test works on synthetic architectures; nothing reads a model file.
"""
import importlib
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import dcm_arch  # noqa: E402

FMA_SCRIPTS = os.path.join(REPO, "experiments", "20261001-se-fc1-leaky", "full-model-analysis", "scripts")
FP16_SCRIPTS = os.path.join(REPO, "documentation", "research", "fp16-feasibility", "scripts")
BF16_SCRIPTS = os.path.join(REPO, "documentation", "research", "bf16-head-offset", "scripts")


def import_from(folder, name):
    """Imports `name` from `folder` without leaving a __pycache__ in the repository."""
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, folder)
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(folder)
        sys.dont_write_bytecode = previous


def group(**fields):
    base = dict(count=1, channels=16, conv1_kernel_size=3, conv2_kernel_size=3, se_style="none",
                se_reduction_ratio=4, use_rezero=False, rezero_alpha_init=1.0, rezero_alpha_cap=1.0,
                activation_function="relu", activation_style="pre", skip_merge="clean_add",
                dropout_multiplier=1, se_beta_init="glorot", se_activation="relu",
                se_gamma_bias_init=0, branch_output_init="standard", skip_projection_init="he")
    base.update(fields)
    return base


def architecture(groups=None, sites=None, **fields):
    """A v9 block-groups architecture: pre-activation, intermediate_conv, no feature skip, ReLU
    at every existing site."""
    base = dict(input_encoding="basic30", block_groups=groups or [group()], stem_conv_kernel_size=3,
                policy_head_style="intermediate_conv", policy_pre_conv_channels=8,
                value_head_style="wdl_softmax", value_head_conv_channels=4, value_head_hidden_units=8,
                compute_data_type="float32", feature_skip_source="none", feature_skip_fusion="concat_direct",
                feature_skip_to_policy_head=False, feature_skip_to_value_head=False,
                feature_skip_to_final_block=False,
                stem_activation="does_not_apply", tower_end_activation="relu",
                feature_skip_activation="does_not_apply", policy_head_activation="relu",
                value_head_conv_activation="relu", value_head_fc1_hidden_activation="relu")
    base.update(sites or {})
    base.update(fields)
    return base


def legacy(arch, activation_function):
    """The pre-v9 form: the six site keys removed, a top-level activation_function stated."""
    out = {k: v for k, v in arch.items() if k not in dcm_arch.SITE_ACTIVATION_KEYS}
    if activation_function is not None:
        out["activation_function"] = activation_function
    return out


def uniform(activation_function, style="pre", **fields):
    base = dict(input_encoding="basic30", channels=16, num_blocks=2, stem_conv_kernel_size=3,
                activation_function=activation_function, block_activation_style=style,
                block_skip_merge="clean_add", block_use_rezero=False, rezero_alpha_init=0.5,
                block_conv1_kernel_size=3, block_conv2_kernel_size=3, block_se_style="none",
                block_se_reduction_ratio=4, policy_head_style="intermediate_conv", policy_pre_conv_channels=16,
                value_head_style="wdl_softmax", value_head_conv_channels=4, value_head_hidden_units=16,
                compute_data_type="float32")
    base.update(fields)
    return base


def md(arch, version="9"):
    out = {"architecture": json.dumps(arch)}
    if version is not None:
        out["dcm_format_version"] = version
    return out


class SiteActivationResolutionTests(unittest.TestCase):
    def test_v9_values_are_used(self):
        arch = architecture(sites=dict(value_head_fc1_hidden_activation="leaky_relu"))
        values = dcm_arch.site_activations_md(md(arch))
        self.assertEqual(values["value_head_fc1_hidden_activation"], "leaky_relu")
        self.assertEqual(values["stem_activation"], "does_not_apply")

    def test_legacy_files_resolve_existing_sites_and_absent_ones(self):
        for version in ("8", "3", None):
            values = dcm_arch.site_activations_md(md(legacy(architecture(), "gelu"), version))
            self.assertEqual(values, dict(stem_activation="does_not_apply", tower_end_activation="gelu",
                                          feature_skip_activation="does_not_apply", policy_head_activation="gelu",
                                          value_head_conv_activation="gelu",
                                          value_head_fc1_hidden_activation="gelu"), version)

    def test_legacy_post_activation_simple_conv(self):
        arch = architecture(groups=[group(activation_style="post", skip_merge="activation_gated")],
                            policy_head_style="simple_conv")
        values = dcm_arch.site_activations_md(md(legacy(arch, "relu"), "3"))
        self.assertEqual(values["stem_activation"], "relu")
        self.assertEqual(values["tower_end_activation"], "does_not_apply")
        self.assertEqual(values["policy_head_activation"], "does_not_apply")

    def test_uniform_tower_resolves_at_any_stated_version(self):
        for version in ("3", "8", "9", None):
            values = dcm_arch.site_activations_md(md(uniform("silu"), version))
            self.assertEqual(values["tower_end_activation"], "silu", version)
            self.assertEqual(values["stem_activation"], "does_not_apply", version)

    def test_uniform_tower_norm_arch_resolves_the_cap_at_v8_and_v9(self):
        for version in ("8", "9"):
            normalized = dcm_arch.norm_arch_md(md(uniform("relu"), version))
            self.assertEqual(normalized["block_groups"][0]["rezero_alpha_cap"],
                             0.5 * dcm_arch.REZERO_TANH_CEILING_MULTIPLE)
        cap_missing = architecture(groups=[{k: v for k, v in group().items() if k != "rezero_alpha_cap"}])
        with self.assertRaises(dcm_arch.ArchitectureError):
            dcm_arch.norm_arch_md(md(cap_missing, "6"))

    def test_v9_missing_key_raises(self):
        for key in dcm_arch.SITE_ACTIVATION_KEYS:
            arch = architecture()
            del arch[key]
            with self.assertRaisesRegex(dcm_arch.ArchitectureError, key):
                dcm_arch.site_activations_md(md(arch))

    def test_v9_retired_activation_function_raises(self):
        with self.assertRaisesRegex(dcm_arch.ArchitectureError, "retired"):
            dcm_arch.site_activations_md(md(architecture(activation_function="relu")))

    def test_legacy_without_activation_function_raises_naming_the_keys(self):
        with self.assertRaisesRegex(dcm_arch.ArchitectureError, "value_head_fc1_hidden_activation"):
            dcm_arch.site_activations_md(md(legacy(architecture(), None), "5"))

    def test_a_stated_key_wins_in_a_legacy_file(self):
        arch = legacy(architecture(), "gelu")
        arch["policy_head_activation"] = "leaky_relu"
        values = dcm_arch.site_activations_md(md(arch, "5"))
        self.assertEqual(values["policy_head_activation"], "leaky_relu")
        self.assertEqual(values["value_head_conv_activation"], "gelu")

    def test_both_mismatch_directions_raise(self):
        with self.assertRaisesRegex(dcm_arch.ArchitectureError, "stem_activation"):
            dcm_arch.site_activations_md(md(architecture(sites=dict(stem_activation="relu"))))
        with self.assertRaisesRegex(dcm_arch.ArchitectureError, "value_head_conv_activation"):
            dcm_arch.site_activations_md(md(architecture(sites=dict(value_head_conv_activation="does_not_apply"))))

    def test_does_not_apply_in_a_group_raises(self):
        for field in ("activation_function", "se_activation"):
            with self.assertRaisesRegex(dcm_arch.ArchitectureError, field):
                dcm_arch.site_activations_md(md(architecture(groups=[group(**{field: "does_not_apply"})])))

    def test_unknown_token_raises(self):
        with self.assertRaisesRegex(dcm_arch.ArchitectureError, "tower_end_activation"):
            dcm_arch.site_activations_md(md(architecture(sites=dict(tower_end_activation="swish"))))

    def test_compress_fusion_is_a_site(self):
        arch = architecture(feature_skip_source="stem_output", feature_skip_fusion="compress_conv_bn_relu",
                            feature_skip_to_policy_head=True, sites=dict(feature_skip_activation="relu"))
        self.assertEqual(dcm_arch.site_activations_md(md(arch))["feature_skip_activation"], "relu")


class RequireReluTests(unittest.TestCase):
    ALL = ("tower_end_activation", "policy_head_activation", "value_head_conv_activation",
           "value_head_fc1_hidden_activation")

    def test_passes_on_an_all_relu_file(self):
        dcm_arch.require_relu(md(architecture()), "t", self.ALL, block_main_path=True)

    def test_raises_naming_a_non_relu_site(self):
        with self.assertRaisesRegex(dcm_arch.ArchitectureError, "t: value_head_fc1_hidden_activation"):
            dcm_arch.require_relu(md(architecture(sites=dict(value_head_fc1_hidden_activation="leaky_relu"))),
                                  "t", self.ALL)

    def test_raises_for_a_named_site_the_file_lacks(self):
        with self.assertRaisesRegex(dcm_arch.ArchitectureError, "stem_activation"):
            dcm_arch.require_relu(md(architecture()), "t", ("stem_activation",))

    def test_checks_the_main_path_only_when_asked(self):
        leaky_groups = md(architecture(groups=[group(activation_function="leaky_relu", se_activation="leaky_relu")]))
        dcm_arch.require_relu(leaky_groups, "t", self.ALL)
        with self.assertRaisesRegex(dcm_arch.ArchitectureError, "block_groups\\[0\\]"):
            dcm_arch.require_relu(leaky_groups, "t", self.ALL, block_main_path=True)


class Checkpoint:
    """A stand-in for fma_lib.Checkpoint: file, raw metadata and the normalized architecture."""

    def __init__(self, metadata):
        self.file = "fixture.safetensors"
        self.metadata = metadata
        self.architecture = dcm_arch.norm_arch_md(metadata)


class UnitsArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units = import_from(FMA_SCRIPTS, "units")

    def test_accepts_relu_files(self):
        self.units.require_supported_architecture(Checkpoint(md(architecture())))
        self.units.require_supported_architecture(Checkpoint(md(legacy(architecture(), "relu"), "5")))

    def test_rejects_each_non_relu_head_site(self):
        for sites in (dict(value_head_fc1_hidden_activation="leaky_relu"), dict(tower_end_activation="gelu"),
                      dict(policy_head_activation="silu"), dict(value_head_conv_activation="leaky_relu")):
            with self.assertRaisesRegex(ValueError, next(iter(sites))):
                self.units.require_supported_architecture(Checkpoint(md(architecture(sites=sites))))

    def test_rejects_simple_conv(self):
        arch = architecture(policy_head_style="simple_conv", sites=dict(policy_head_activation="does_not_apply"))
        with self.assertRaisesRegex(ValueError, "policy_head_activation"):
            self.units.require_supported_architecture(Checkpoint(md(arch)))

    def test_rejects_a_feature_skip_before_any_activation_check(self):
        arch = architecture(feature_skip_source="stem_output", feature_skip_to_value_head=True,
                            sites=dict(value_head_fc1_hidden_activation="leaky_relu"))
        with self.assertRaisesRegex(ValueError, "feature skip"):
            self.units.require_supported_architecture(Checkpoint(md(arch)))


class ForwardGuardTests(unittest.TestCase):
    def test_fwd_forward_requires_md_and_refuses_a_leaky_value_fc1(self):
        fwd = import_from(BF16_SCRIPTS, "fwd")
        arch = dcm_arch.norm_arch_md(md(architecture()))
        with self.assertRaises(TypeError):
            fwd.forward({}, arch, np.zeros((30, 8, 8)))
        leaky = md(architecture(sites=dict(value_head_fc1_hidden_activation="leaky_relu")))
        with self.assertRaisesRegex(dcm_arch.ArchitectureError, "value_head_fc1_hidden_activation"):
            fwd.forward({}, arch, np.zeros((30, 8, 8)), md=leaky)

    def test_fwd16_forward_requires_sites_and_refuses_unmodelled_files(self):
        fwd16 = import_from(FP16_SCRIPTS, "fwd16")
        relu = architecture()
        arch = dcm_arch.norm_arch_md(md(relu))
        x = np.zeros((1, 30, 8, 8))
        with self.assertRaises(TypeError):
            fwd16.forward({}, arch, x)
        with self.assertRaisesRegex(AssertionError, "feature skip"):
            fwd16.forward({}, dict(arch, feature_skip_source="stem_output"), x,
                          sites=dcm_arch.site_activations_md(md(relu)))
        leaky = architecture(sites=dict(value_head_fc1_hidden_activation="leaky_relu"))
        with self.assertRaisesRegex(AssertionError, "value FC1"):
            fwd16.forward({}, dcm_arch.norm_arch_md(md(leaky)), x, sites=dcm_arch.site_activations_md(md(leaky)))
        # A pre-activation file states the stem as does_not_apply and passes the checks
        # (it then fails only for want of tensors); one stating a ReLU stem is refused first.
        with self.assertRaises(KeyError):
            fwd16.forward({}, arch, x, sites=dcm_arch.site_activations_md(md(relu)))
        with self.assertRaisesRegex(dcm_arch.ArchitectureError, "stem_activation"):
            dcm_arch.site_activations_md(md(architecture(sites=dict(stem_activation="relu"))))


if __name__ == "__main__":
    unittest.main()
