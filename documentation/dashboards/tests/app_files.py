"""Real model files for the Python tests, made by the app.

The Python readers (dcm_arch, dcm_lineage and everything built on them) read model
files only through the app (`--describe-model`), so their tests need files the app
reads. One tiny model (one 8-channel block) is minted per test process with
`--new-model`; every test file is a copy of it whose `__metadata__` is rewritten —
the same thing the Swift tests do to make a header case (`rewritten(...)`). The
tensors are untouched, so the app checks the header exactly as it checks a file it
wrote, and refuses what it would refuse.

A copy can be written at an older format version: its architecture JSON is put in
that version's form (the fields that version did not have are dropped and the
per-site activations folded back into `activation_function`), which the app then
resolves the way it resolves any older file.

Needs a built DrewsChessMachine binary ($DCM_BINARY, else this repo's newest
DerivedData build); without one, `base_file()` raises unittest.SkipTest.
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import dcm_arch  # noqa: E402

# The lineage keys a minted file carries: the record and its write-only flat mirrors.
# A copy starts without them, so each test states the lineage it means.
LINEAGE_KEYS = ("dcm_lineage", "lineage_run_id", "lineage_segment_index", "lineage_segment_local_step",
                "cum_trainer_step", "cum_games_fed", "cum_train_step_sec", "git_dirty", "derivation_history",
                "parent_model_id")

# Per-site activations (format v9) and the group fields each older format lacked.
SITE_ACTIVATIONS = ("stem_activation", "tower_end_activation", "feature_skip_activation",
                    "policy_head_activation", "value_head_conv_activation", "value_head_fc1_hidden_activation")
HEAD_INIT_FIELDS = ("policy_head_final_init", "value_head_final_init", "value_head_draw_prior")
GROUP_INIT_FIELDS = ("se_gamma_bias_init", "branch_output_init", "skip_projection_init")


class _Base:
    folder = None
    path = None
    error = None


def base_file():
    """The minted tiny model's path, minting it on first use."""
    if _Base.path:
        return _Base.path
    if _Base.error is not None:
        raise _Base.error
    try:
        binary = dcm_arch.app_binary()
    except dcm_arch.ArchitectureError as error:
        _Base.error = unittest.SkipTest(str(error))
        raise _Base.error from None
    folder = tempfile.mkdtemp(prefix="dcm-app-files-")

    def run(*args):
        result = subprocess.run([binary, *args], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"{' '.join(args[:2])} failed ({result.returncode}): {result.stderr.strip()}")

    try:
        # The smallest architecture a built-in preset starts from: its own form, as
        # the app describes it, cut to one 8-channel block with small heads.
        seed = os.path.join(folder, "seed.safetensors")
        run("--new-model", "--architecture", "v4_4block_3x3_fp32", "--out-model", seed, "--init-seed", "1")
        description = dcm_arch.describe(seed)
        architecture = description["architecture"]
        architecture["block_groups"] = [dict(architecture["block_groups"][0], count=1, channels=8)]
        architecture.update(policy_pre_conv_channels=8, value_head_conv_channels=2, value_head_hidden_units=8)
        preset = os.path.join(folder, "tiny-preset.json")
        with open(preset, "w") as handle:
            json.dump({"label": "tiny", "format_version": description["format_version"],
                       "architecture": architecture}, handle)
        path = os.path.join(folder, "tiny.safetensors")
        run("--new-model", "--architecture", preset, "--out-model", path, "--init-seed", "1")
    except Exception as error:
        shutil.rmtree(folder)
        _Base.error = RuntimeError(f"the tiny test model could not be minted: {error}")
        raise _Base.error from None
    _Base.folder, _Base.path = folder, path
    return path


def remove_base():
    """Removes the minted model's folder (call from a module's tearDownModule)."""
    if _Base.folder:
        shutil.rmtree(_Base.folder)
        _Base.folder = _Base.path = None


def read_file(path):
    """(header dict, tensor data bytes) of a .safetensors file."""
    with open(path, "rb") as handle:
        raw = handle.read()
    (length,) = struct.unpack("<Q", raw[:8])
    return json.loads(raw[8:8 + length]), raw[8 + length:]


def write_file(path, header, data):
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)) + encoded + data)


def architecture_at(architecture, format_version):
    """`architecture` (the current form) in the form a file of `format_version`
    (None: no version, the unversioned legacy form) holds it."""
    arch = json.loads(json.dumps(architecture))
    version = 0 if format_version is None else int(format_version)
    if version < 12:
        arch.pop("policy_tail_precision")
    if version < 9:
        activations = {arch.pop(site) for site in SITE_ACTIVATIONS} - {dcm_arch.DOES_NOT_APPLY}
        if len(activations) != 1:
            raise ValueError(f"no single activation_function for format v{version}: {sorted(activations)}")
        arch["activation_function"] = activations.pop()
    if version < 8:
        for field in HEAD_INIT_FIELDS:
            arch.pop(field)
        for group in arch["block_groups"]:
            for field in GROUP_INIT_FIELDS:
                group.pop(field)
    if version < 6:
        for group in arch["block_groups"]:
            group.pop("rezero_alpha_cap")
    if version < 5:
        for group in arch["block_groups"]:
            group.pop("se_activation")
    if version < 4:
        for group in arch["block_groups"]:
            group.pop("se_beta_init")
    return arch


def write_model_file(path, metadata, format_version="12", remove=(), keep_lineage=False):
    """A copy of the tiny model at `path`, its lineage keys dropped (kept, the
    minted file's own record, with `keep_lineage`), `metadata` (string values) set
    over the rest, `remove` keys removed, and its format version `format_version`
    (None: no version key) with the architecture in that version's form."""
    header, data = read_file(base_file())
    stored = header["__metadata__"]
    for key in () if keep_lineage else LINEAGE_KEYS:
        stored.pop(key, None)
    architecture = json.loads(stored["architecture"])
    stored["architecture"] = json.dumps(architecture_at(architecture, format_version), sort_keys=True)
    if format_version is None:
        stored.pop("dcm_format_version")
    else:
        stored["dcm_format_version"] = str(format_version)
    stored.update(metadata)
    for key in remove:
        stored.pop(key, None)
    write_file(path, header, data)
    return path


def app_record(record):
    """A test's hand-built lineage record with the two fields every record the app
    writes carries and the tests' records leave out: an empty derivation history
    and a behavior fingerprint (any recipe-3 hash; the readers never compare it)."""
    record = json.loads(json.dumps(record))
    record.setdefault("derivation_history", [])
    record["rng"].setdefault("behavior_fingerprint", {"recipe": 3, "sha256": "0" * 64})
    return record


def write_header(path, metadata):
    """A model file whose header states `metadata`, given the way a test writes a
    header: string values, `dcm_format_version` absent for an unversioned file, and
    `dcm_lineage` a record's JSON (passed through `app_record` when it is a record).
    The minted file's `creator` is dropped unless `metadata` states one, so the
    header says nothing about its writer that the test did not."""
    stored = dict(metadata)
    version = stored.pop("dcm_format_version", None)
    if "dcm_lineage" in stored:
        record = json.loads(stored["dcm_lineage"])
        if isinstance(record, dict) and isinstance(record.get("rng"), dict):
            stored["dcm_lineage"] = json.dumps(app_record(record), sort_keys=True)
    return write_model_file(path, stored, version, remove=() if "creator" in stored else ("creator",))


def write_lineage_file(path, record, model_id, training_step=None, creator=None, format_version="12",
                       metadata=None):
    """A model file carrying `record` (passed through `app_record`), stating
    `model_id` and, when given, `training_step` and `creator`."""
    stored = {"model_id": model_id, "dcm_lineage": json.dumps(app_record(record), sort_keys=True)}
    if training_step is not None:
        stored["training_step"] = str(training_step)
    if creator is not None:
        stored["creator"] = creator
    stored.update(metadata or {})
    return write_model_file(path, stored, format_version)
