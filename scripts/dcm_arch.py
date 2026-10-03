"""A model file's architecture and ReZero parameters, read the way the app reads them.

The single Python source for turning a DCM `.safetensors` header into the
architecture the app would build from it. Every analysis script that needs a
block group's fields — above all the ReZero cap — gets them here, from the
checkpoint's own `__metadata__`, so no script carries a cap (or any other
architecture value) of its own.

Version gating mirrors `Network/ArchitectureFormat.swift` and
`NetworkArchitecture.swift`: a field added in format version N is required in
files of version >= N (missing it raises, as the Swift loader refuses), and a
file older than N resolves the field to what the engine did before the field
existed. For the ReZero cap that legacy value is
`rezero_alpha_init * REZERO_TANH_CEILING_MULTIPLE` — the C the forward pass
`C * tanh(alpha / C)` used before `rezero_alpha_cap` was stored.

Import from anywhere in the repository with

    sys.path.insert(0, os.path.join(<repo root>, "scripts"))
    import dcm_arch

This module has no import-time side effects and needs only the standard
library.
"""
import json
import math
import struct

# First DCM file format (safetensors `dcm_format_version`) whose block groups
# must state `se_beta_init`; older files predate the field and mean 'glorot'.
SE_BETA_INIT_REQUIRED_FROM_VERSION = 4
# First format whose block groups must state `se_activation` (the SE FC1
# activation); older files predate the field, and their FC1 used the group's
# own `activation_function`.
SE_ACTIVATION_REQUIRED_FROM_VERSION = 5
# First format whose block groups must state `rezero_alpha_cap` (the asymptote C
# of the forward ReZero bound C*tanh(alpha/C)); older files predate the field,
# and their C was `rezero_alpha_init * REZERO_TANH_CEILING_MULTIPLE`.
REZERO_ALPHA_CAP_REQUIRED_FROM_VERSION = 6
# NetworkArchitecture.rezeroTanhCeilingMultiple: the frozen legacy cap rule.
REZERO_TANH_CEILING_MULTIPLE = 1.0


class ArchitectureError(ValueError):
    """A header whose architecture the app would refuse to load."""


def norm_arch(s, format_version=None):
    """Normalize a DCM architecture JSON (string or dict) to the block-groups form.

    format_version: the carrier's `dcm_format_version` (string or int). Files of
    version >= SE_BETA_INIT_REQUIRED_FROM_VERSION must carry `se_beta_init` on
    every block group, files of version >= SE_ACTIVATION_REQUIRED_FROM_VERSION
    `se_activation`, and files of version >= REZERO_ALPHA_CAP_REQUIRED_FROM_VERSION
    `rezero_alpha_cap` (a missing one raises, mirroring the Swift loader). Older or
    unversioned files resolve a missing `se_beta_init` to 'glorot', a missing
    `se_activation` to the group's `activation_function`, and a missing
    `rezero_alpha_cap` to `rezero_alpha_init * REZERO_TANH_CEILING_MULTIPLE`. Prefer
    `norm_arch_md(md)`, which reads the version from the file's metadata.
    """
    a = json.loads(s) if isinstance(s, str) else dict(s)
    if 'block_groups' not in a:
        a['block_groups'] = [dict(
            count=a['num_blocks'], channels=a['channels'],
            conv1_kernel_size=a['block_conv1_kernel_size'],
            conv2_kernel_size=a['block_conv2_kernel_size'],
            se_style=a['block_se_style'], se_reduction_ratio=a['block_se_reduction_ratio'],
            use_rezero=a['block_use_rezero'], rezero_alpha_init=a['rezero_alpha_init'],
            activation_function=a['activation_function'],
            activation_style=a['block_activation_style'], skip_merge=a['block_skip_merge'],
            dropout_multiplier=1, se_beta_init='glorot', se_activation=a['activation_function'])]
    version = None if format_version is None else int(format_version)
    strict_beta = version is not None and version >= SE_BETA_INIT_REQUIRED_FROM_VERSION
    strict_se_act = version is not None and version >= SE_ACTIVATION_REQUIRED_FROM_VERSION
    strict_cap = version is not None and version >= REZERO_ALPHA_CAP_REQUIRED_FROM_VERSION
    for i, g in enumerate(a['block_groups']):
        if 'se_beta_init' not in g:
            if strict_beta:
                raise ArchitectureError(
                    f"block_groups[{i}].se_beta_init missing in a format v{format_version} architecture")
            g['se_beta_init'] = 'glorot'
        if 'se_activation' not in g:
            if strict_se_act:
                raise ArchitectureError(
                    f"block_groups[{i}].se_activation missing in a format v{format_version} architecture")
            g['se_activation'] = g['activation_function']
        if 'rezero_alpha_cap' not in g:
            if strict_cap:
                raise ArchitectureError(
                    f"block_groups[{i}].rezero_alpha_cap missing in a format v{format_version} architecture")
            g['rezero_alpha_cap'] = g['rezero_alpha_init'] * REZERO_TANH_CEILING_MULTIPLE
    a.setdefault('feature_skip_source', 'none')
    return a


def norm_arch_md(md):
    """norm_arch for a safetensors __metadata__ dict, gated on its dcm_format_version."""
    if 'architecture' not in md:
        raise ArchitectureError("safetensors __metadata__ has no 'architecture'")
    return norm_arch(md['architecture'], md.get('dcm_format_version'))


def read_metadata(path):
    """The `__metadata__` dict of a .safetensors file (string values, as stored)."""
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ArchitectureError(f"{path}: shorter than a safetensors header length")
        header_length = struct.unpack("<Q", prefix)[0]
        raw = handle.read(header_length)
        if len(raw) != header_length:
            raise ArchitectureError(f"{path}: header truncated ({len(raw)} of {header_length} bytes)")
    header = json.loads(raw)
    if "__metadata__" not in header:
        raise ArchitectureError(f"{path}: no __metadata__ in the safetensors header")
    return header["__metadata__"]


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


def rezero_blocks(metadata):
    """Every residual block's ReZero settings for a file's metadata, groups expanded by `count`."""
    architecture = norm_arch_md(metadata)
    blocks = []
    for group_index, group in enumerate(architecture['block_groups']):
        use_rezero = bool(group['use_rezero'])
        alpha_init = float(group['rezero_alpha_init'])
        alpha_cap = float(group['rezero_alpha_cap'])
        if use_rezero:
            _require_valid_cap(alpha_cap)
        for _ in range(int(group['count'])):
            blocks.append(RezeroBlock(len(blocks), group_index, use_rezero, alpha_init, alpha_cap))
    return blocks


def rezero_blocks_of_file(path):
    return rezero_blocks(read_metadata(path))


def rezero_effective(raw_alpha, cap):
    """C * tanh(raw / C) for the cap C the app uses (NetworkArchitecture: cap > 0 and finite)."""
    _require_valid_cap(cap)
    return cap * math.tanh(raw_alpha / cap)


def _require_valid_cap(cap):
    if not (math.isfinite(cap) and cap > 0):
        raise ArchitectureError(f"ReZero cap {cap!r} is not finite and positive; the app refuses such a file")
