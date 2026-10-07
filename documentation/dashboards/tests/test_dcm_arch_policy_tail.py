"""Tests for the format-v12 policy tail in scripts/dcm_arch.py (`policy_tail_precision`,
`recorded_policy_tail`), the mirror of the app's ArchitectureFormat.decodePolicyTailPrecision
and SafetensorsModelIO.recordedPolicyTailPrecision (POLICY_TAIL_ARCHITECTURE_PLAN.md PT-D2,
PT-D3 rule 1).

Run: python3 -m unittest discover -s documentation/dashboards/tests
Every test works on synthetic headers; nothing reads a model file.
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import dcm_arch  # noqa: E402


def header(version="11", dtype="bfloat16", stated=None, flat=None, configured=None):
    arch = {"block_groups": [{"count": 1}], "compute_data_type": dtype}
    if stated is not None:
        arch["policy_tail_precision"] = stated
    md = {"dcm_format_version": version, "architecture": json.dumps(arch)}
    if flat is not None:
        md["trainer_policy_tail_precision"] = flat
    if configured is not None:
        md["dcm_lineage"] = json.dumps({"configuration": {"policy_tail_precision": configured}})
    return md


class PolicyTailTests(unittest.TestCase):

    def test_a_stated_value_is_read_at_any_version(self):
        for version in ("8", "11", "12"):
            with self.subTest(version=version):
                self.assertEqual(dcm_arch.policy_tail_precision(header(version, stated="fp32_from_pre_bn")),
                                 ("fp32_from_pre_bn", "stated"))

    def test_v12_requires_the_field(self):
        with self.assertRaises(dcm_arch.ArchitectureError):
            dcm_arch.policy_tail_precision(header("12"))

    def test_a_pre_v12_file_reads_its_flat_key_then_its_lineage(self):
        self.assertEqual(dcm_arch.policy_tail_precision(header(flat="fp32_from_pre_bn")),
                         ("fp32_from_pre_bn", "recorded"))
        self.assertEqual(dcm_arch.policy_tail_precision(header(configured="fp32_from_pre_bn")),
                         ("fp32_from_pre_bn", "recorded"))

    def test_a_pre_v12_file_recording_nothing_is_mixed(self):
        self.assertEqual(dcm_arch.policy_tail_precision(header()), ("mixed_final_projection", "not_recorded"))

    def test_fp32_is_does_not_apply_whatever_is_recorded(self):
        self.assertEqual(dcm_arch.policy_tail_precision(header(dtype="float32", flat="fp32_from_pre_bn")),
                         ("does_not_apply", "fp32"))
        self.assertEqual(dcm_arch.policy_tail_precision(header(dtype="float32")), ("does_not_apply", "fp32"))

    def test_a_flat_key_disagreeing_with_the_lineage_is_refused(self):
        with self.assertRaises(dcm_arch.ArchitectureError):
            dcm_arch.policy_tail_precision(header(flat="fp32_from_pre_bn", configured="mixed_final_projection"))

    def test_a_malformed_recorded_value_is_refused(self):
        with self.assertRaises(dcm_arch.ArchitectureError):
            dcm_arch.policy_tail_precision(header(flat="fp16"))


if __name__ == "__main__":
    unittest.main()
