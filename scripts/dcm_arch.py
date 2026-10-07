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

The six architecture-level site activations of format v9 (`stem_activation`,
`tower_end_activation`, `feature_skip_activation`, `policy_head_activation`,
`value_head_conv_activation`, `value_head_fc1_hidden_activation`) are resolved
by `site_activations` / `site_activations_md`, not by `norm_arch`: only scripts
that model the heads need them. A site the topology lacks holds
'does_not_apply' (never an identity activation), and only such a site may hold
it. `require_relu` is the one-line guard of a script that models ReLU at sites
it never reads a key for.

Import from anywhere in the repository with

    sys.path.insert(0, os.path.join(<repo root>, "scripts"))
    import dcm_arch

This module has no import-time side effects and needs only the standard
library.
"""
import copy
import json
import math
import re
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
# First format whose architecture must state the six site activations below and
# must not state the top-level `activation_function` they replace; older files
# resolve each existing site from that `activation_function` and each absent one
# to DOES_NOT_APPLY (ArchitectureFormat.siteActivationsRequiredFromVersion).
SITE_ACTIVATIONS_REQUIRED_FROM_VERSION = 9
# First format whose SE-less block groups must state se_activation =
# 'does_not_apply' (OD-13); older files (v9 included) wrote the group's own
# activation there, never applied, and resolve it to 'does_not_apply'
# (ArchitectureFormat.seLessSEActivationDoesNotApplyFromVersion).
SE_LESS_SE_ACTIVATION_DOES_NOT_APPLY_FROM_VERSION = 10
# NetworkArchitecture's site keys, in graph build order (ArchitectureActivationSite).
SITE_ACTIVATION_KEYS = (
    'stem_activation', 'tower_end_activation', 'feature_skip_activation',
    'policy_head_activation', 'value_head_conv_activation', 'value_head_fc1_hidden_activation')
# The value a site field holds when the topology lacks the site. Not a function.
DOES_NOT_APPLY = 'does_not_apply'
# ActivationFunction.functions: every token that names a function.
ACTIVATION_FUNCTIONS = ('relu', 'silu', 'gelu', 'leaky_relu')
# The format the app gives a carrier with no version marker (legacy).
UNVERSIONED_LEGACY_VERSION = 3
# ArchitectureFormat.currentVersion: the newest format this module understands.
# A newer (or non-positive) version is refused, as the app's requireSupported
# refuses it, rather than read under rules that may not hold for it. v11 adds no
# architecture field: from it a header's `training_step` is the trainer step
# (`dcm_lineage.step_reading` reads a file's step under either rule), and a copy
# of this module that predates it refuses a v11 file instead of misreading it.
CURRENT_FORMAT_VERSION = 11
# A safetensors header larger than this is not a DCM model header (theirs are
# a few kilobytes); refusing it keeps a damaged length prefix from being read
# as a request for gigabytes.
MAX_HEADER_BYTES = 100 * 1024 * 1024
# The text Swift's `Int(String)` parses: an optional sign, then one or more ASCII
# digits, nothing else (no whitespace, underscores or non-ASCII digits, all of
# which Python's int() accepts). Used with fullmatch: `$` would admit a trailing
# newline. Shared with `dcm_lineage`'s integer header values.
SWIFT_INT_TEXT = re.compile(r'[+-]?[0-9]+')
# The keys of the uniform-tower (legacy) form, every one required by the app's
# decoder (NetworkArchitecture.init(from:) without `block_groups`).
_UNIFORM_TOWER_KEYS = (
    'num_blocks', 'channels', 'block_conv1_kernel_size', 'block_conv2_kernel_size', 'block_se_style',
    'block_se_reduction_ratio', 'block_use_rezero', 'rezero_alpha_init', 'activation_function',
    'block_activation_style', 'block_skip_merge')
# The block-group keys norm_arch and rezero_blocks read, each required by the
# app's BlockGroup decoder at every format version (`activation_style`, also
# required there, is checked by site_exists, its one reader here).
_BLOCK_GROUP_KEYS_READ = ('count', 'se_style', 'use_rezero', 'rezero_alpha_init', 'activation_function')


class ArchitectureError(ValueError):
    """A header whose architecture the app would refuse to load."""


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
    """The carrier's format version as an int, or None for an unversioned
    (legacy) carrier. Refuses what the app refuses
    (ArchitectureFormat.requireSupported / safetensorsFormatVersion): a value
    `parsed_format_version` refuses, or one newer than CURRENT_FORMAT_VERSION."""
    if format_version is None:
        return None
    version = parsed_format_version(format_version)
    if version > CURRENT_FORMAT_VERSION:
        raise ArchitectureError(
            f"format v{version} is newer than this module understands (newest: v{CURRENT_FORMAT_VERSION})")
    return version


def _refuse_retired_activation_function(a, uniform, version):
    """A v9+ block-groups architecture must not state the top-level
    `activation_function` the six site keys replaced (FormatError.retiredField);
    the uniform-tower form is legacy by construction and exempt."""
    if (not uniform and version is not None and version >= SITE_ACTIVATIONS_REQUIRED_FROM_VERSION
            and 'activation_function' in a):
        raise ArchitectureError(
            f"format v{version} architecture states the retired top-level 'activation_function'; "
            f"it is replaced by {', '.join(SITE_ACTIVATION_KEYS)}")


def _require_keys(mapping, keys, prefix):
    """Raises ArchitectureError naming the first of `keys` missing from `mapping`,
    as `<prefix><key> is missing` (the app's decoder refuses a missing required key)."""
    for key in keys:
        if key not in mapping:
            raise ArchitectureError(f"{prefix}{key} is missing")


def _require_group_token(i, field, value, allow_does_not_apply):
    if value in ACTIVATION_FUNCTIONS or (allow_does_not_apply and value == DOES_NOT_APPLY):
        return
    allowed = ', '.join(ACTIVATION_FUNCTIONS + ((DOES_NOT_APPLY,) if allow_does_not_apply else ()))
    raise ArchitectureError(f"block_groups[{i}].{field} is {value!r}, which is not one of {allowed}")


def norm_arch(s, format_version=None):
    """Normalize a DCM architecture JSON (string or dict) to the block-groups form.

    Never changes its argument: a dict is deep-copied first.

    format_version: the carrier's `dcm_format_version` (string or int; None for an
    unversioned carrier, which is legacy). A version that is not a positive integer,
    or is newer than CURRENT_FORMAT_VERSION, raises (`checked_format_version`). Files
    of version >= SE_BETA_INIT_REQUIRED_FROM_VERSION must carry `se_beta_init` on
    every block group, files of version >= SE_ACTIVATION_REQUIRED_FROM_VERSION
    `se_activation`, and files of version >= REZERO_ALPHA_CAP_REQUIRED_FROM_VERSION
    `rezero_alpha_cap` (a missing one raises, mirroring the Swift loader). Older or
    unversioned files resolve a missing `se_beta_init` to 'glorot', a missing
    `se_activation` to the group's `activation_function` on a group with an SE block
    and to 'does_not_apply' on one without, and a missing `rezero_alpha_cap` to
    `rezero_alpha_init * REZERO_TANH_CEILING_MULTIPLE`.

    The block-group activation rules are checked as the app checks them: each
    group's `activation_function` is a function; `se_activation` is
    'does_not_apply' exactly when the group has no SE block (OD-13) — a file older
    than v10 whose SE-less group states the group's own activation resolves it to
    'does_not_apply', any other disagreement raises; and a v9+ block-groups
    architecture must not state the retired top-level `activation_function`. The six
    site activations themselves are resolved by `site_activations`. Prefer
    `norm_arch_md(md)`, which reads the version from the file's metadata.
    """
    a = json.loads(s) if isinstance(s, str) else copy.deepcopy(s)
    version = checked_format_version(format_version)
    uniform = 'block_groups' not in a
    _refuse_retired_activation_function(a, uniform, version)
    if uniform:
        _require_keys(a, _UNIFORM_TOWER_KEYS, '')
        _require_group_token(0, 'activation_function', a['activation_function'], allow_does_not_apply=False)
        a['block_groups'] = [dict(
            count=a['num_blocks'], channels=a['channels'],
            conv1_kernel_size=a['block_conv1_kernel_size'],
            conv2_kernel_size=a['block_conv2_kernel_size'],
            se_style=a['block_se_style'], se_reduction_ratio=a['block_se_reduction_ratio'],
            use_rezero=a['block_use_rezero'], rezero_alpha_init=a['rezero_alpha_init'],
            activation_function=a['activation_function'],
            activation_style=a['block_activation_style'], skip_merge=a['block_skip_merge'],
            dropout_multiplier=1, se_beta_init='glorot',
            # An SE-less group has no FC1 (OD-13).
            se_activation=(DOES_NOT_APPLY if a['block_se_style'] == 'none' else a['activation_function']),
            # The uniform-tower form is legacy by construction whatever version
            # its carrier states, so its cap is always the legacy derivation
            # (NetworkArchitecture's uniform-tower expansion).
            rezero_alpha_cap=a['rezero_alpha_init'] * REZERO_TANH_CEILING_MULTIPLE)]
    strict_beta = version is not None and version >= SE_BETA_INIT_REQUIRED_FROM_VERSION
    strict_se_act = version is not None and version >= SE_ACTIVATION_REQUIRED_FROM_VERSION
    strict_cap = version is not None and version >= REZERO_ALPHA_CAP_REQUIRED_FROM_VERSION
    legacy_se_less = uniform or version is None or version < SE_LESS_SE_ACTIVATION_DOES_NOT_APPLY_FROM_VERSION
    groups = a['block_groups']
    if not isinstance(groups, list) or not groups:
        raise ArchitectureError("block_groups must contain at least one group")
    for i, g in enumerate(groups):
        _require_keys(g, _BLOCK_GROUP_KEYS_READ, f"block_groups[{i}].")
        has_se = g['se_style'] != 'none'
        _require_group_token(i, 'activation_function', g['activation_function'], allow_does_not_apply=False)
        if 'se_beta_init' not in g:
            if strict_beta:
                raise ArchitectureError(
                    f"block_groups[{i}].se_beta_init missing in a format v{format_version} architecture")
            g['se_beta_init'] = 'glorot'
        if 'se_activation' not in g:
            if strict_se_act:
                raise ArchitectureError(
                    f"block_groups[{i}].se_activation missing in a format v{format_version} architecture")
            g['se_activation'] = g['activation_function'] if has_se else DOES_NOT_APPLY
        _require_group_token(i, 'se_activation', g['se_activation'], allow_does_not_apply=True)
        # OD-13: an SE-less group's se_activation is 'does_not_apply'. Before v10
        # it had to equal the group's activation and was never applied, so such
        # a file's value resolves to it; any other disagreement raises, as the
        # app's decoder refuses it (BlockGroup.decodedSEActivation).
        if has_se and g['se_activation'] == DOES_NOT_APPLY:
            raise ArchitectureError(
                f"block_groups[{i}].se_activation is 'does_not_apply', but the group has an SE block: choose one of "
                f"{', '.join(ACTIVATION_FUNCTIONS)}")
        if not has_se and g['se_activation'] != DOES_NOT_APPLY:
            if not (legacy_se_less and g['se_activation'] == g['activation_function']):
                raise ArchitectureError(
                    f"block_groups[{i}].se_activation is {g['se_activation']!r}, but the group has no SE block: "
                    f"it must be 'does_not_apply'")
            g['se_activation'] = DOES_NOT_APPLY
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


_SITE_ABSENT_REASON = {
    'stem_activation': 'the first block group is pre-activation, so the stem has no activation',
    'tower_end_activation': 'the last block group is post-activation, so the tower has no tower-end activation',
    'feature_skip_activation': 'no compress fusion node is built',
    'policy_head_activation': 'the policy head is simple_conv, so it has no pre-block',
}


def site_exists(norm, key):
    """Whether a normalized architecture (`norm_arch` output) has the site of
    `key` — the rule of NetworkArchitecture.hasActivationSite."""
    groups = norm['block_groups']
    if key == 'stem_activation':
        _require_keys(groups[0], ('activation_style',), 'block_groups[0].')
        return groups[0]['activation_style'] == 'post'
    if key == 'tower_end_activation':
        _require_keys(groups[-1], ('activation_style',), f"block_groups[{len(groups) - 1}].")
        return groups[-1]['activation_style'] == 'pre'
    if key == 'feature_skip_activation':
        return (norm.get('feature_skip_source', 'none') != 'none'
                and norm.get('feature_skip_fusion') == 'compress_conv_bn_relu'
                and bool(norm.get('feature_skip_to_policy_head') or norm.get('feature_skip_to_value_head')))
    if key == 'policy_head_activation':
        _require_keys(norm, ('policy_head_style',), '')
        style = norm['policy_head_style']
        if style not in ('simple_conv', 'intermediate_conv', 'fc_bottleneck'):
            raise ArchitectureError(f"unknown policy_head_style {style!r}")
        return style != 'simple_conv'
    if key in ('value_head_conv_activation', 'value_head_fc1_hidden_activation'):
        return True
    raise ArchitectureError(f"{key!r} is not a site activation key (one of {', '.join(SITE_ACTIVATION_KEYS)})")


def _require_token(key, value):
    if value != DOES_NOT_APPLY and value not in ACTIVATION_FUNCTIONS:
        raise ArchitectureError(f"{key} is {value!r}, which is not one of {', '.join(ACTIVATION_FUNCTIONS)} "
                                f"or {DOES_NOT_APPLY!r}")


def site_activations(arch, format_version=None):
    """The six site activations of a DCM architecture JSON (string or dict, as
    stored — before `norm_arch`, which erases the uniform-tower form), under the
    app's rules (NetworkArchitecture.init(from:format:)):

    - a stated key wins, at any version;
    - a file older than SITE_ACTIVATIONS_REQUIRED_FROM_VERSION (None = unversioned
      = legacy), or in the uniform-tower form, resolves an unstated site from its
      top-level `activation_function` where the topology has the site and to
      DOES_NOT_APPLY where it does not; with no `activation_function` either it
      raises, naming every unresolved key;
    - a v9+ block-groups file missing a key, or stating the retired top-level
      `activation_function`, raises.

    Then both directions are checked — a site the topology has must hold a
    function and one it lacks must hold DOES_NOT_APPLY — and `norm_arch` checks the
    same rule for each block group: `activation_function` is a function, and
    `se_activation` is DOES_NOT_APPLY exactly when the group has no SE block
    (OD-13). An explicit JSON null in a site key counts as absent (the app's
    decodeIfPresent). A version that is not a positive integer or is newer than
    CURRENT_FORMAT_VERSION raises. Never changes its argument. Every
    ArchitectureError names the key and the site."""
    raw = json.loads(arch) if isinstance(arch, str) else copy.deepcopy(arch)
    checked = checked_format_version(format_version)
    version = UNVERSIONED_LEGACY_VERSION if checked is None else checked
    uniform = 'block_groups' not in raw
    legacy_allowed = uniform or version < SITE_ACTIVATIONS_REQUIRED_FROM_VERSION
    _refuse_retired_activation_function(raw, uniform, version)
    tower = raw.get('activation_function')
    if tower is not None:
        _require_token('activation_function', tower)
    if uniform:
        if tower is None:
            raise ArchitectureError("a uniform-tower architecture must state 'activation_function'")
        if tower == DOES_NOT_APPLY:
            raise ArchitectureError("activation_function is 'does_not_apply', but that site always exists")
    values = {}
    resolved = []
    for key in SITE_ACTIVATION_KEYS:
        # An explicit JSON null is absent, as Swift's decodeIfPresent reads it.
        if raw.get(key) is not None:
            _require_token(key, raw[key])
            values[key] = raw[key]
        elif legacy_allowed:
            resolved.append(key)
        else:
            raise ArchitectureError(f"site activation '{key}' is missing in a format v{version} architecture")
    if resolved and tower is None:
        raise ArchitectureError(
            f"format v{version} architecture states neither {', '.join(resolved)} nor the top-level "
            f"'activation_function' they resolve from")
    norm = norm_arch(raw, None if uniform else format_version)
    for key in resolved:
        values[key] = tower if site_exists(norm, key) else DOES_NOT_APPLY
    # The block-group activation rules (OD-13 included) are checked by norm_arch.
    for key in SITE_ACTIVATION_KEYS:
        exists = site_exists(norm, key)
        if exists and values[key] == DOES_NOT_APPLY:
            raise ArchitectureError(f"{key} is 'does_not_apply', but the model has that site: choose one of "
                                    f"{', '.join(ACTIVATION_FUNCTIONS)}")
        if not exists and values[key] != DOES_NOT_APPLY:
            raise ArchitectureError(f"{key} is {values[key]!r}, but {_SITE_ABSENT_REASON[key]}: "
                                    f"it must be 'does_not_apply'")
    return values


def site_activations_md(md):
    """site_activations for a safetensors __metadata__ dict, gated on its dcm_format_version."""
    if 'architecture' not in md:
        raise ArchitectureError("safetensors __metadata__ has no 'architecture'")
    return site_activations(md['architecture'], md.get('dcm_format_version'))


def require_relu(md, source, sites, block_main_path=False):
    """The guard of a script that models ReLU at `sites` without reading a key:
    raises ArchitectureError naming `source` and the site unless every named site
    of the file (a safetensors __metadata__ dict) is exactly 'relu'. A site the
    file lacks is 'does_not_apply' and raises too — the caller models it as a
    present ReLU. With block_main_path=True every block group's
    `activation_function` must be 'relu' as well."""
    values = site_activations_md(md)
    for key in sites:
        if key not in SITE_ACTIVATION_KEYS:
            raise ArchitectureError(f"{source}: {key!r} is not a site activation key")
        if values[key] != 'relu':
            raise ArchitectureError(f"{source}: {key} is {values[key]!r}; this script models a ReLU there "
                                    f"and cannot analyse this file")
    if block_main_path:
        for i, group in enumerate(norm_arch_md(md)['block_groups']):
            if group['activation_function'] != 'relu':
                raise ArchitectureError(f"{source}: block_groups[{i}].activation_function is "
                                        f"{group['activation_function']!r}; this script models a ReLU main path")


def require_architecture_of(md, arch, source):
    """The pairing guard of a forward pass that takes a normalized architecture
    and the file metadata its site activations come from as separate arguments:
    raises ArchitectureError naming `source` unless `arch` equals
    `norm_arch_md(md)`, so the two cannot come from different files."""
    if arch != norm_arch_md(md):
        raise ArchitectureError(f"{source}: arch is not norm_arch_md(md); the architecture and the metadata "
                                f"come from different files")


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
