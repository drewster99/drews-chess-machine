"""A model file's architecture and ReZero parameters, as the app reads them.

The app is the one implementation of the architecture rules: format-version
gates, legacy resolutions (a pre-v6 ReZero cap, pre-v9 site activations, a pre-v10
SE-less `se_activation`, a pre-v12 policy tail), and every refusal. This module
holds none of them. `describe(path)` runs the app's `--describe-model` on the file
and returns what the loader resolved; every other architecture reader here reads
that. A file the app refuses is an ArchitectureError naming the app's reason.

The app binary is $DCM_BINARY, else the newest build of this repository's project
in DerivedData (`app_binary`). Descriptions are header-only and batched (one app
launch reads many files) and cached per file version and binary.

Also here, because it is not an architecture rule: the safetensors header reader
(`read_header`, `read_metadata`), shared by the Python tooling to read raw header
values and tensors.

Import from anywhere in the repository with

    sys.path.insert(0, os.path.join(<repo root>, "scripts"))
    import dcm_arch

No import-time side effects; standard library only.
"""
import copy
import glob
import json
import math
import os
import plistlib
import struct
import subprocess

# A safetensors header larger than this is not a DCM model header (theirs are
# a few kilobytes); refusing it keeps a damaged length prefix from being read
# as a request for gigabytes.
MAX_HEADER_BYTES = 100 * 1024 * 1024
# The value a site activation holds when the topology lacks the site.
DOES_NOT_APPLY = 'does_not_apply'
# Where Xcode leaves built binaries.
DERIVED_DATA = os.path.expanduser("~/Library/Developer/Xcode/DerivedData")


class ArchitectureError(ValueError):
    """A model file whose architecture cannot be read: the app refuses it, the app
    cannot be run, or a script's guard refuses what it holds."""


# This repository's Xcode project: DerivedData folders are matched to it by the
# `WorkspacePath` Xcode records in each folder's info.plist, so a build of another
# checkout or worktree is never picked.
PROJECT_PATH = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                             "DrewsChessMachine", "DrewsChessMachine.xcodeproj"))
# A describe call reads headers only (kilobytes per file); a hung app is
# reported, not waited on. Generous per file for a cold disk.
DESCRIBE_TIMEOUT_SECONDS = 60
DESCRIBE_TIMEOUT_PER_FILE_SECONDS = 2
# Files per app launch: one launch reads many headers.
DESCRIBE_BATCH = 500


# The DerivedData binary this process found (`app_binary`): found once, since a
# scan of DerivedData reads every project's info.plist; `describe_files` keys its
# cache on the binary's identity, so a rebuilt binary is never answered from an
# older one's results.
_FOUND_BINARY = None


def app_binary():
    """The DrewsChessMachine binary that describes files: $DCM_BINARY (refused when
    set but empty), else the newest built one (Debug or Release) of this
    repository's project in DerivedData, found once per process."""
    global _FOUND_BINARY
    if "DCM_BINARY" in os.environ:
        binary = os.environ["DCM_BINARY"]
        if not binary:
            raise ArchitectureError("DCM_BINARY is set but empty")
        return binary
    if _FOUND_BINARY is None:
        _FOUND_BINARY = _newest_derived_data_binary()
    return _FOUND_BINARY


def _newest_derived_data_binary():
    candidates = []
    for folder in glob.glob(os.path.join(DERIVED_DATA, "DrewsChessMachine-*")):
        try:
            with open(os.path.join(folder, "info.plist"), "rb") as handle:
                workspace = plistlib.load(handle).get("WorkspacePath")
        except (OSError, plistlib.InvalidFileException):
            continue
        if workspace is None or os.path.realpath(workspace) != PROJECT_PATH:
            continue
        candidates += [p for p in glob.glob(os.path.join(folder, "Build", "Products", "*", "DrewsChessMachine.app",
                                                         "Contents", "MacOS", "DrewsChessMachine"))
                       if os.access(p, os.X_OK)]
    if not candidates:
        raise ArchitectureError(f"no built DrewsChessMachine binary of {PROJECT_PATH} under {DERIVED_DATA}; "
                                f"build it or set DCM_BINARY")
    return max(candidates, key=os.path.getmtime)


_DESCRIPTIONS = {}


def _identity(path):
    """What a cached description is valid for: the file's identity and version."""
    stat = os.stat(path)
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _dependency_identity(real):
    """The version of the other file a description depends on, or None. A
    session's `champion.safetensors` is described with its sibling
    `trainer.safetensors`'s policy tail when it records none itself (the app's
    session-champion rule), so its description changes with that file too."""
    folder, name = os.path.split(real)
    if name != "champion.safetensors" or not folder.endswith(".dcmsession"):
        return None
    trainer = os.path.join(folder, "trainer.safetensors")
    try:
        return _identity(trainer)
    except FileNotFoundError:
        return "absent"


def describe_files(paths):
    """The app's description of each model file in `paths`, as
    {path: description or ArchitectureError}: one app launch per
    DESCRIBE_BATCH files (`--describe-model`, header-only). A description holds
    `architecture` (resolved, current form), `expanded_blocks`, `format_version`,
    `legacy_resolutions`, `weight_plan`, `model_id`, `creator`, `lineage`,
    `step_reading` and `test_set_results` (see `ModelDescription` in the app).
    Cached per file version (and, for a session champion, its trainer file's) and
    binary, refusals included; each result is the caller's own copy."""
    if not paths:
        return {}
    binary = app_binary()
    try:
        binary_identity = _identity(binary)
    except OSError as error:
        raise ArchitectureError(f"cannot run {binary}: {error}") from None
    results, pending = {}, []
    for path in paths:
        real = os.path.realpath(path)
        try:
            key = (real, _identity(real), _dependency_identity(real), binary, binary_identity)
        except OSError as error:
            results[path] = ArchitectureError(f"{path}: {error}")
            continue
        if key in _DESCRIPTIONS:
            results[path] = _result(path, _DESCRIPTIONS[key])
        else:
            pending.append((path, real, key))
    for start in range(0, len(pending), DESCRIBE_BATCH):
        batch = pending[start:start + DESCRIBE_BATCH]
        for (path, real, key), outcome in zip(batch, _run_describe(binary, [real for _, real, _ in batch])):
            # Cached only when the file (and what it depends on) is still the
            # version stat'ed before the read.
            try:
                unchanged = (_identity(real), _dependency_identity(real)) == (key[1], key[2])
            except OSError:
                unchanged = False
            if unchanged:
                _DESCRIPTIONS[key] = outcome
            results[path] = _result(path, outcome)
    return results


def _result(path, outcome):
    """The caller's own copy of a cached outcome, a refusal naming `path`."""
    if isinstance(outcome, ArchitectureError):
        return ArchitectureError(f"{path}: {outcome}")
    return copy.deepcopy(outcome)


def _run_describe(binary, reals):
    """One `--describe-model` launch: a description or an ArchitectureError per
    file, in order."""
    timeout = DESCRIBE_TIMEOUT_SECONDS + DESCRIBE_TIMEOUT_PER_FILE_SECONDS * len(reals)
    try:
        result = subprocess.run([binary, "--describe-model", *reals], stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ArchitectureError(f"{binary} --describe-model did not finish in {timeout} s") from None
    except OSError as error:
        raise ArchitectureError(f"cannot run {binary}: {error}") from None
    if result.returncode not in (0, 1):
        reason = result.stderr.strip() or f"exit status {result.returncode}"
        raise ArchitectureError(f"--describe-model failed: {reason}")
    lines = result.stdout.splitlines()
    if len(lines) != len(reals):
        raise ArchitectureError(f"--describe-model printed {len(lines)} lines for {len(reals)} files")
    outcomes = []
    for real, line in zip(reals, lines):
        try:
            entry = json.loads(line)
        except ValueError as error:
            raise ArchitectureError(f"--describe-model printed a line that is not JSON ({error})") from None
        if entry.get("path") != real:
            raise ArchitectureError(f"--describe-model answered for {entry.get('path')!r}, expected {real!r}")
        if "error" in entry:
            outcomes.append(ArchitectureError(entry["error"]))
        else:
            outcomes.append(entry["description"])
    return outcomes


def describe(path):
    """The app's description of one model file (`describe_files`); raises
    ArchitectureError when the app refuses it."""
    outcome = describe_files([path])[path]
    if isinstance(outcome, ArchitectureError):
        raise outcome
    return outcome


def architecture(path):
    """The resolved architecture of the model file at `path` (block-groups form, every
    field the format defines present, legacy values resolved)."""
    return describe(path)["architecture"]


def site_activations(path):
    """The architecture-level site activations of the file at `path`, by key
    (`stem_activation`, `tower_end_activation`, ...): a function, or DOES_NOT_APPLY
    where the topology lacks the site."""
    arch = architecture(path)
    return {key: value for key, value in arch.items() if key.endswith("_activation")}


def require_relu(path, source, sites, block_main_path=False):
    """The guard of a script that models ReLU at `sites` without reading them:
    raises ArchitectureError naming `source` unless every named site of the file
    at `path` is exactly 'relu'. A site the file lacks is DOES_NOT_APPLY and raises
    too — the caller models it as a present ReLU. With block_main_path=True every
    block group's `activation_function` must be 'relu' as well."""
    values = site_activations(path)
    for key in sites:
        if key not in values:
            raise ArchitectureError(f"{source}: {key!r} is not a site activation key")
        if values[key] != 'relu':
            raise ArchitectureError(f"{source}: {key} is {values[key]!r}; this script models a ReLU there "
                                    f"and cannot analyse this file")
    if block_main_path:
        for i, group in enumerate(architecture(path)['block_groups']):
            if group['activation_function'] != 'relu':
                raise ArchitectureError(f"{source}: block_groups[{i}].activation_function is "
                                        f"{group['activation_function']!r}; this script models a ReLU main path")


def require_architecture_of(path, arch, source):
    """The pairing guard of a forward pass that takes an architecture and the file
    it came from as separate arguments: raises ArchitectureError naming `source`
    unless `arch` is `architecture(path)`, so the two cannot come from different
    files."""
    if arch != architecture(path):
        raise ArchitectureError(f"{source}: arch is not the architecture of {path}; the architecture and "
                                f"the file come from different models")


class RezeroBlock:
    """One residual block's ReZero settings, in build order."""

    def __init__(self, index, group_index, use_rezero, alpha_init, alpha_cap):
        self.index = index
        self.group_index = group_index
        self.use_rezero = use_rezero
        self.alpha_init = alpha_init
        self.alpha_cap = alpha_cap

    def effective(self, raw_alpha):
        """The forward pass's effective alpha, C * tanh(raw / C); refuses a block without ReZero."""
        if not self.use_rezero:
            raise ArchitectureError(f"block {self.index} has no ReZero; there is no effective alpha")
        return rezero_effective(raw_alpha, self.alpha_cap)

    def effective_derivative(self, raw_alpha):
        """d(effective)/d(raw) = sech^2(raw / C): how much of the raw gradient passes the bound."""
        if not self.use_rezero:
            raise ArchitectureError(f"block {self.index} has no ReZero")
        _require_valid_cap(self.alpha_cap)
        return 1.0 / math.cosh(raw_alpha / self.alpha_cap) ** 2


def rezero_blocks(path):
    """Every residual block's ReZero settings for the file at `path`, groups expanded by
    `count`, with the cap the app resolved (a pre-v6 file's legacy cap included)."""
    blocks = []
    for group_index, group in enumerate(architecture(path)['block_groups']):
        use_rezero = bool(group['use_rezero'])
        alpha_init = float(group['rezero_alpha_init'])
        alpha_cap = float(group['rezero_alpha_cap'])
        if use_rezero:
            _require_valid_cap(alpha_cap)
        for _ in range(int(group['count'])):
            blocks.append(RezeroBlock(len(blocks), group_index, use_rezero, alpha_init, alpha_cap))
    return blocks


def rezero_effective(raw_alpha, cap):
    """C * tanh(raw / C), the forward pass's bound (NetworkArchitecture: cap > 0 and finite)."""
    _require_valid_cap(cap)
    return cap * math.tanh(raw_alpha / cap)


def _require_valid_cap(cap):
    if not (math.isfinite(cap) and cap > 0):
        raise ArchitectureError(f"ReZero cap {cap!r} is not finite and positive")


def read_header(path):
    """The whole safetensors header of a .safetensors file — the tensor index
    and its `__metadata__` dict (string values, as stored) — and the byte offset
    at which tensor data starts (`data_offsets` are relative to it), reading
    only the header.

    The one header reader of the Python tooling (`read_metadata` and
    `dcm_lineage.read_metadata` are built on it). A damaged file is an
    ArchitectureError: a length prefix of zero or above MAX_HEADER_BYTES, a
    short read, a header that is not a JSON object, or one without a
    `__metadata__` object. Without the bound, a damaged prefix is a request to
    read up to 2^64 bytes, which fails as OverflowError or MemoryError — neither
    of which a caller catching ValueError expects."""
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ArchitectureError(f"{path}: shorter than a safetensors header length")
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length == 0 or header_length > MAX_HEADER_BYTES:
            raise ArchitectureError(f"{path}: safetensors header length {header_length} is not a model header")
        raw = handle.read(header_length)
        if len(raw) != header_length:
            raise ArchitectureError(f"{path}: header truncated ({len(raw)} of {header_length} bytes)")
    try:
        header = json.loads(raw)
    except ValueError as error:
        raise ArchitectureError(f"{path}: safetensors header is not JSON ({error})") from None
    if not isinstance(header, dict):
        raise ArchitectureError(f"{path}: safetensors header is not a JSON object")
    if not isinstance(header.get("__metadata__"), dict):
        raise ArchitectureError(f"{path}: no __metadata__ object in the safetensors header")
    return header, 8 + header_length


def read_metadata(path):
    """The `__metadata__` dict of a .safetensors file (string values, as
    stored), reading only the header: `read_header`'s, with its refusals."""
    return read_header(path)[0]["__metadata__"]
