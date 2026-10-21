"""Tests for scripts/dcm_arch.py — a model file's architecture as the app reads it
(`--describe-model`) — and the scripts' guards built on it, on real model files the
app writes.

The architecture rules themselves (format-version gates, legacy resolutions, refusals)
are the app's and are tested in Swift (ArchitectureActivationSiteTests,
PolicyTailArchitectureTests, RezeroAlphaCapTests, SEBetaInitTests,
ArchitectureFormatVersionTextTests). These tests check that the Python side reads the
app's answer faithfully and that each script's guard refuses what it should.

Run: python3 -m unittest discover -s documentation/dashboards/tests
Needs a built DrewsChessMachine binary ($DCM_BINARY, else the newest DerivedData build)
and numpy; without the binary every test here skips. Building the fixtures (six model
files) takes a few seconds.
"""
import importlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import dcm_arch  # noqa: E402
import dcm_lineage  # noqa: E402

FMA_SCRIPTS = os.path.join(REPO, "experiments", "20261001-se-fc1-leaky", "full-model-analysis", "scripts")
FP16_SCRIPTS = os.path.join(REPO, "documentation", "research", "fp16-feasibility", "scripts")
BF16_SCRIPTS = os.path.join(REPO, "documentation", "research", "bf16-head-offset", "scripts")
HEAD_SITES = ("tower_end_activation", "policy_head_activation", "value_head_conv_activation",
              "value_head_fc1_hidden_activation")


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


class Fixtures:
    """Real model files, written once per run by the app's own commands:
    - relu: `--new-model` v4_4block_3x3_fp32 (pre-activation, ReLU everywhere,
      intermediate_conv, SE scale_and_bias, ReZero);
    - leaky_fc1: relu with `--set-value-head-fc1-hidden-activation leaky_relu`;
    - silu_main: relu with the block main path SiLU and every site ReLU;
    - post_simple_conv: `--new-model` v3_8block_3x3 (post-activation, simple_conv policy,
      no ReZero);
    - simple_conv: relu's own description (`--describe-model`) with a simple_conv policy,
      written as a preset file and minted by `--new-model` (pre-activation, so only the
      policy site differs from relu);
    - fusion: `--new-model` v4_5block_7x7_fusion (feature skip into both heads)."""
    folder = None
    paths = {}
    skip_reason = None

    @classmethod
    def build(cls):
        if cls.folder or cls.skip_reason:
            return
        try:
            binary = dcm_arch.app_binary()
        except dcm_arch.ArchitectureError as error:
            cls.skip_reason = str(error)
            return
        cls.folder = tempfile.mkdtemp(prefix="dcm-arch-tests-")

        def run(*args):
            result = subprocess.run([binary, *args], stdin=subprocess.DEVNULL, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"{' '.join(args[:2])} failed ({result.returncode}): {result.stderr.strip()}")

        def new_model(name, preset):
            path = os.path.join(cls.folder, f"{name}.safetensors")
            run("--new-model", "--architecture", preset, "--out-model", path, "--init-seed", "1")
            cls.paths[name] = path

        def derive(name, source, *operations):
            path = os.path.join(cls.folder, f"{name}.safetensors")
            run("--derive-model", "--from", cls.paths[source], *operations, "--out", path)
            cls.paths[name] = path

        new_model("relu", "v4_4block_3x3_fp32")
        derive("leaky_fc1", "relu", "--set-value-head-fc1-hidden-activation", "leaky_relu")
        derive("silu_main", "relu", "--set-activation", "silu", "--set-tower-end-activation", "relu",
               "--set-policy-head-activation", "relu", "--set-value-head-conv-activation", "relu",
               "--set-value-head-fc1-hidden-activation", "relu")
        new_model("post_simple_conv", "v3_8block_3x3")
        simple = dcm_arch.architecture(cls.paths["relu"])
        simple.update(policy_head_style="simple_conv", policy_head_activation=dcm_arch.DOES_NOT_APPLY)
        preset = os.path.join(cls.folder, "simple_conv-preset.json")
        with open(preset, "w") as handle:
            json.dump({"label": "simple_conv", "format_version": dcm_arch.CURRENT_FORMAT_VERSION,
                       "architecture": simple}, handle)
        new_model("simple_conv", preset)
        new_model("fusion", "v4_5block_7x7_fusion")

    @classmethod
    def path(cls, name):
        cls.build()
        if cls.skip_reason:
            raise unittest.SkipTest(cls.skip_reason)
        return cls.paths[name]


def tearDownModule():
    if Fixtures.folder:
        shutil.rmtree(Fixtures.folder)


class DescribeTests(unittest.TestCase):
    """Protects: dcm_arch returns the app's own description of a file, and passes the
    app's refusals on as ArchitectureError. Fails if: the wrapper reads the header
    itself, drops or reshapes the app's answer, or lets a refusal or an unrunnable
    binary escape as another exception type."""

    def test_reads_the_apps_description_of_the_file(self):
        path = Fixtures.path("relu")
        description = dcm_arch.describe(path)
        self.assertEqual(description["model_id"], dcm_arch.read_metadata(path)["model_id"])
        self.assertEqual(description["format_version"], dcm_arch.CURRENT_FORMAT_VERSION)
        self.assertEqual(dcm_arch.architecture(path), description["architecture"])
        self.assertEqual(dcm_arch.architecture(path)["block_groups"][0]["activation_function"], "relu")

    def test_a_caller_editing_a_result_changes_no_later_answer(self):
        path = Fixtures.path("relu")
        first = dcm_arch.architecture(path)
        first["policy_head_style"] = "edited"
        first["block_groups"][0]["channels"] = -1
        again = dcm_arch.architecture(path)
        self.assertEqual(again["policy_head_style"], "intermediate_conv")
        self.assertGreater(again["block_groups"][0]["channels"], 0)

    def test_site_activations_are_the_architectures_site_keys(self):
        sites = dcm_arch.site_activations(Fixtures.path("post_simple_conv"))
        self.assertEqual(set(sites), {"stem_activation", "tower_end_activation", "feature_skip_activation",
                                      "policy_head_activation", "value_head_conv_activation",
                                      "value_head_fc1_hidden_activation"})
        # Post-activation, simple_conv: a stem activation, no tower end, no policy pre-block.
        self.assertEqual(sites["stem_activation"], "relu")
        self.assertEqual(sites["tower_end_activation"], dcm_arch.DOES_NOT_APPLY)
        self.assertEqual(sites["policy_head_activation"], dcm_arch.DOES_NOT_APPLY)

    def test_a_file_the_app_refuses_is_an_architecture_error_with_its_reason(self):
        source = Fixtures.path("relu")
        header, data_start = dcm_arch.read_header(source)
        header["__metadata__"]["dcm_format_version"] = str(dcm_arch.CURRENT_FORMAT_VERSION + 1)
        header["__metadata__"].pop("content_sha256", None)
        encoded = json.dumps(header).encode()
        newer = os.path.join(Fixtures.folder, "newer-format.safetensors")
        with open(source, "rb") as reader, open(newer, "wb") as writer:
            reader.seek(data_start)
            writer.write(struct.pack("<Q", len(encoded)) + encoded + reader.read())
        with self.assertRaisesRegex(dcm_arch.ArchitectureError, re.escape(newer)):
            dcm_arch.describe(newer)

    def test_a_binary_that_cannot_run_is_an_architecture_error(self):
        path = Fixtures.path("relu")
        previous = os.environ.get("DCM_BINARY")
        os.environ["DCM_BINARY"] = os.path.join(Fixtures.folder, "no-such-binary")
        try:
            # The cache is per binary, so this call runs the missing one.
            with self.assertRaisesRegex(dcm_arch.ArchitectureError, "cannot run"):
                dcm_arch.describe(path)
        finally:
            if previous is None:
                del os.environ["DCM_BINARY"]
            else:
                os.environ["DCM_BINARY"] = previous


class RezeroBlockTests(unittest.TestCase):
    """Protects: the per-block ReZero settings and the forward bound C*tanh(raw/C).
    Fails if: groups stop expanding by `count`, the cap is taken from anything but
    the app's resolved architecture, or the bound's formula changes."""

    def test_blocks_follow_the_groups_with_the_resolved_cap(self):
        path = Fixtures.path("relu")
        group = dcm_arch.architecture(path)["block_groups"][0]
        blocks = dcm_arch.rezero_blocks(path)
        self.assertEqual([b.index for b in blocks], list(range(group["count"])))
        self.assertTrue(all(b.use_rezero and b.alpha_cap == float(group["rezero_alpha_cap"]) for b in blocks))
        cap = blocks[0].alpha_cap
        self.assertEqual(blocks[0].effective(0.0), 0.0)
        self.assertAlmostEqual(blocks[0].effective(0.3), cap * np.tanh(0.3 / cap))

    def test_a_block_without_rezero_has_no_effective_alpha(self):
        blocks = dcm_arch.rezero_blocks(Fixtures.path("post_simple_conv"))
        self.assertFalse(blocks[0].use_rezero)
        with self.assertRaises(dcm_arch.ArchitectureError):
            blocks[0].effective(0.1)


class RequireReluTests(unittest.TestCase):
    """Protects: the guard of scripts that model ReLU without reading the site keys.
    Fails if: a non-ReLU site, a site the file lacks, or (when asked) a non-ReLU main
    path passes."""

    def test_passes_on_an_all_relu_file(self):
        dcm_arch.require_relu(Fixtures.path("relu"), "t", HEAD_SITES, block_main_path=True)

    def test_raises_naming_a_non_relu_site(self):
        with self.assertRaisesRegex(dcm_arch.ArchitectureError, "t: value_head_fc1_hidden_activation"):
            dcm_arch.require_relu(Fixtures.path("leaky_fc1"), "t", HEAD_SITES)

    def test_raises_for_a_named_site_the_file_lacks(self):
        with self.assertRaisesRegex(dcm_arch.ArchitectureError, "stem_activation"):
            dcm_arch.require_relu(Fixtures.path("relu"), "t", ("stem_activation",))

    def test_checks_the_main_path_only_when_asked(self):
        silu_main = Fixtures.path("silu_main")
        dcm_arch.require_relu(silu_main, "t", HEAD_SITES)
        with self.assertRaisesRegex(dcm_arch.ArchitectureError, "block_groups\\[0\\]"):
            dcm_arch.require_relu(silu_main, "t", HEAD_SITES, block_main_path=True)


class UnitsArchitectureTests(unittest.TestCase):
    """Protects: units.py analyses only the topology it models. Fails if: it accepts a
    non-ReLU head site, a simple_conv policy (no pre-block), or a feature skip."""

    @classmethod
    def setUpClass(cls):
        cls.units = import_from(FMA_SCRIPTS, "units")
        cls.fma_lib = import_from(FMA_SCRIPTS, "fma_lib")

    def checkpoint(self, name):
        return self.fma_lib.Checkpoint(Fixtures.path(name))

    def test_accepts_a_relu_file(self):
        self.units.require_supported_architecture(self.checkpoint("relu"))

    def test_rejects_a_non_relu_head_site(self):
        with self.assertRaisesRegex(ValueError, "value_head_fc1_hidden_activation"):
            self.units.require_supported_architecture(self.checkpoint("leaky_fc1"))

    def test_rejects_simple_conv(self):
        with self.assertRaisesRegex(ValueError, "policy_head_activation"):
            self.units.require_supported_architecture(self.checkpoint("simple_conv"))

    def test_rejects_a_feature_skip(self):
        with self.assertRaisesRegex(ValueError, "feature skip"):
            self.units.require_supported_architecture(self.checkpoint("fusion"))


class ForwardGuardTests(unittest.TestCase):
    """Protects: the research forward passes run only on a file they model, with the
    architecture of that same file. Fails if: `path` stops being required, a
    non-ReLU site or a feature skip passes, or an architecture from another file is
    accepted."""

    def test_fwd_forward_requires_path_and_refuses_a_leaky_value_fc1(self):
        fwd = import_from(BF16_SCRIPTS, "fwd")
        relu = Fixtures.path("relu")
        with self.assertRaises(TypeError):
            fwd.forward({}, dcm_arch.architecture(relu), np.zeros((30, 8, 8)))
        leaky = Fixtures.path("leaky_fc1")
        with self.assertRaisesRegex(dcm_arch.ArchitectureError, "value_head_fc1_hidden_activation"):
            fwd.forward({}, dcm_arch.architecture(leaky), np.zeros((30, 8, 8)), path=leaky)

    def test_fwd_forward_refuses_an_architecture_from_another_file(self):
        fwd = import_from(BF16_SCRIPTS, "fwd")
        with self.assertRaisesRegex(dcm_arch.ArchitectureError, "different models"):
            fwd.forward({}, dcm_arch.architecture(Fixtures.path("post_simple_conv")), np.zeros((30, 8, 8)),
                        path=Fixtures.path("relu"))

    def test_fwd16_forward_requires_path_and_refuses_unmodelled_files(self):
        fwd16 = import_from(FP16_SCRIPTS, "fwd16")
        relu = Fixtures.path("relu")
        arch = dcm_arch.architecture(relu)
        x = np.zeros((1, 30, 8, 8))
        with self.assertRaises(TypeError):
            fwd16.forward({}, arch, x)
        fusion = Fixtures.path("fusion")
        with self.assertRaisesRegex(dcm_arch.ArchitectureError, "feature skip"):
            fwd16.forward({}, dcm_arch.architecture(fusion), x, path=fusion)
        leaky = Fixtures.path("leaky_fc1")
        with self.assertRaisesRegex(dcm_arch.ArchitectureError, "value_head_fc1_hidden_activation"):
            fwd16.forward({}, dcm_arch.architecture(leaky), x, path=leaky)
        with self.assertRaisesRegex(dcm_arch.ArchitectureError, "different models"):
            fwd16.forward({}, arch, x, path=leaky)
        # A file the guards accept fails only for want of tensors.
        with self.assertRaises(KeyError):
            fwd16.forward({}, arch, x, path=relu)


class FormatVersionTextTests(unittest.TestCase):
    """Protects: the format-version parsing `dcm_lineage` uses for a file's step
    reading, which must read a version as the app does. Fails if: it accepts text
    Swift's Int(String) refuses (whitespace, underscores, decimals, non-ASCII digits),
    a non-positive version, or a version newer than the app's."""

    def test_unknown_format_versions_are_refused(self):
        for version in (str(dcm_arch.CURRENT_FORMAT_VERSION + 1), "0", "-1", "nine"):
            with self.assertRaises(dcm_arch.ArchitectureError, msg=version):
                dcm_arch.checked_format_version(version)
        self.assertEqual(dcm_arch.checked_format_version(str(dcm_arch.CURRENT_FORMAT_VERSION)),
                         dcm_arch.CURRENT_FORMAT_VERSION)
        self.assertIsNone(dcm_arch.checked_format_version(None))

    def test_format_version_text_is_what_swift_int_parses(self):
        for value in ("1_0", " 9", "9\n", "9.0", "", "+", "-0", "٩", True, 9.0):
            with self.assertRaises(dcm_arch.ArchitectureError, msg=repr(value)):
                dcm_arch.checked_format_version(value)
        for value, expected in (("+9", 9), ("09", 9), (9, 9)):
            self.assertEqual(dcm_arch.checked_format_version(value), expected)

    # Each Python constant that hand-copies an ArchitectureFormat constant, by its
    # Swift name. A version that moves in Swift but not here would make dcm_lineage
    # read a file's step differently from the app.
    SWIFT_FORMAT_CONSTANTS = (
        ("currentVersion", dcm_arch, "CURRENT_FORMAT_VERSION"),
        ("lineageRequiredFromVersion", dcm_lineage, "LINEAGE_REQUIRED_FROM_VERSION"),
        ("unversionedLegacyVersion", dcm_lineage, "UNVERSIONED_LEGACY_VERSION"),
    )

    def test_format_versions_match_the_swift_constants(self):
        """Both sides are parsed, never written here as numbers, so a version bump
        made in both places keeps this passing and one made in only one fails it."""
        with open(os.path.join(REPO, "DrewsChessMachine", "DrewsChessMachine", "Network",
                               "ArchitectureFormat.swift"), encoding="utf-8") as swift_file:
            source = swift_file.read()
        body = re.search(r"^enum ArchitectureFormat\b[^{]*\{(.*?)^\}", source, re.S | re.M).group(1)
        swift = {}
        for name, value in re.findall(r"^\s*static let (\w+) = (\d+)\s*$", body, re.M):
            self.assertNotIn(name, swift, f"ArchitectureFormat.{name} is declared twice")
            swift[name] = int(value)
        for swift_name, module, python_name in self.SWIFT_FORMAT_CONSTANTS:
            with self.subTest(swift_name=swift_name, python_name=f"{module.__name__}.{python_name}"):
                self.assertIn(swift_name, swift, f"ArchitectureFormat.{swift_name} not found as an integer literal")
                self.assertEqual(getattr(module, python_name), swift[swift_name])


if __name__ == "__main__":
    unittest.main()
