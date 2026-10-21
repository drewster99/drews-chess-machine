"""A model file's architecture and ReZero parameters, as the app reads them.

The app is the one implementation of the architecture rules: format-version
gates, legacy resolutions (a pre-v6 ReZero cap, pre-v9 site activations, a pre-v10
SE-less `se_activation`, a pre-v12 policy tail), and every refusal. This module
holds none of them. `describe(path)` runs the app's `--describe-model` on the file
and returns what the loader resolved; every other architecture reader here reads
that. A file the app refuses is an ArchitectureError naming the app's reason.

The app binary is $DCM_BINARY, else the newest built DrewsChessMachine in
DerivedData (`app_binary`). Each file is described once per process (cached by
path, size and modification time).

Also here, because they are not architecture rules: the safetensors header reader
(`read_header`, `read_metadata`), shared by the Python tooling to read tensors, and
the format-version text parsing `dcm_lineage` uses for a file's step reading.

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
import re
import struct
import subprocess

# A safetensors header larger than this is not a DCM model header (theirs are
# a few kilobytes); refusing it keeps a damaged length prefix from being read
# as a request for gigabytes.
MAX_HEADER_BYTES = 100 * 1024 * 1024
# The text Swift's `Int(String)` parses: an optional sign, then one or more ASCII
# digits, nothing else (no whitespace, underscores or non-ASCII digits, all of
# which Python's int() accepts). Used with fullmatch: `$` would admit a trailing
# newline. Shared with `dcm_lineage`'s integer header values.
SWIFT_INT_TEXT = re.compile(r'[+-]?[0-9]+')
# ArchitectureFormat.currentVersion: the newest format `dcm_lineage`'s step
# reading understands. A newer (or non-positive) version is refused rather than
# read under rules that may not hold for it.
CURRENT_FORMAT_VERSION = 12
# The value a site activation holds when the topology lacks the site.
DOES_NOT_APPLY = 'does_not_apply'
# Where Xcode leaves built binaries.
DERIVED_DATA = os.path.expanduser("~/Library/Developer/Xcode/DerivedData")


class ArchitectureError(ValueError):
    """A model file whose architecture cannot be read: the app refuses it, the app
    cannot be run, or a script's guard refuses what it holds."""


def app_binary():
    """The DrewsChessMachine binary that describes files: $DCM_BINARY, else the newest
    built one (Debug or Release) in DerivedData, the rule of
    scripts/dcm_launcher_common.sh `dcm_resolve_binary latest`."""
    if os.environ.get("DCM_BINARY"):
        return os.environ["DCM_BINARY"]
    pattern = os.path.join(DERIVED_DATA, "DrewsChessMachine-*", "Build", "Products", "*",
                           "DrewsChessMachine.app", "Contents", "MacOS", "DrewsChessMachine")
    candidates = [p for p in glob.glob(pattern) if os.access(p, os.X_OK)]
    if not candidates:
        raise ArchitectureError(f"no built DrewsChessMachine binary under {DERIVED_DATA}; set DCM_BINARY")
    return max(candidates, key=os.path.getmtime)


_DESCRIPTIONS = {}


def describe(path):
    """The app's `--describe-model` JSON for the model file at `path`: `architecture`
    (resolved, current form), `expanded_blocks`, `format_version`,
    `legacy_resolutions`, `weight_plan`, `model_id`, ... Cached per file version; each
    call returns its own copy, so a caller editing it changes no later answer."""
    real = os.path.realpath(path)
    try:
        stat = os.stat(real)
    except OSError as error:
        raise ArchitectureError(f"{path}: {error}") from None
    binary = app_binary()
    key = (real, stat.st_size, stat.st_mtime_ns, binary)
    if key not in _DESCRIPTIONS:
        try:
            result = subprocess.run([binary, "--describe-model", real], stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True)
        except OSError as error:
            raise ArchitectureError(f"{path}: cannot run {binary}: {error}") from None
        if result.returncode != 0:
            reason = result.stderr.strip() or f"exit status {result.returncode}"
            raise ArchitectureError(f"{path}: {reason}")
        try:
            _DESCRIPTIONS[key] = json.loads(result.stdout)
        except ValueError as error:
            raise ArchitectureError(f"{path}: --describe-model printed no JSON ({error})") from None
    return copy.deepcopy(_DESCRIPTIONS[key])


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


def parsed_format_version(format_version):
    """A stated format version as a positive int, parsed as the app parses it
    (ArchitectureFormat.safetensorsFormatVersion): text must be what Swift's
    `Int(String)` accepts (`SWIFT_INT_TEXT`); an int (not a bool) is taken as is;
    anything else, or a value of 0 or below, raises. No upper bound: that is
    `checked_format_version`'s."""
    if isinstance(format_version, str):
        if not SWIFT_INT_TEXT.fullmatch(format_version):
            raise ArchitectureError(f"format version {format_version!r} is not an integer")
        version = int(format_version)
    elif isinstance(format_version, int) and not isinstance(format_version, bool):
        version = format_version
    else:
        raise ArchitectureError(f"format version {format_version!r} is not an integer")
    if version <= 0:
        raise ArchitectureError(f"format version {format_version!r} is not a positive integer")
    return version


def checked_format_version(format_version):
    """The carrier's format version as an int, or None for an unversioned (legacy)
    carrier. Refuses what `parsed_format_version` refuses, and a version newer than
    CURRENT_FORMAT_VERSION."""
    if format_version is None:
        return None
    version = parsed_format_version(format_version)
    if version > CURRENT_FORMAT_VERSION:
        raise ArchitectureError(
            f"format v{version} is newer than this module understands (newest: v{CURRENT_FORMAT_VERSION})")
    return version


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
