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
# First format whose architecture must state the six site activations below and
# must not state the top-level `activation_function` they replace; older files
# resolve each existing site from that `activation_function` and each absent one
# to DOES_NOT_APPLY (ArchitectureFormat.siteActivationsRequiredFromVersion).
SITE_ACTIVATIONS_REQUIRED_FROM_VERSION = 9
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
# A safetensors header larger than this is not a DCM model header (theirs are
# a few kilobytes); refusing it keeps a damaged length prefix from being read
# as a request for gigabytes.
MAX_HEADER_BYTES = 100 * 1024 * 1024


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
            dropout_multiplier=1, se_beta_init='glorot', se_activation=a['activation_function'],
            # The uniform-tower form is legacy by construction whatever version
            # its carrier states, so its cap is always the legacy derivation
            # (NetworkArchitecture's uniform-tower expansion).
            rezero_alpha_cap=a['rezero_alpha_init'] * REZERO_TANH_CEILING_MULTIPLE)]
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
        return groups[0]['activation_style'] == 'post'
    if key == 'tower_end_activation':
        return groups[-1]['activation_style'] == 'pre'
    if key == 'feature_skip_activation':
        return (norm.get('feature_skip_source', 'none') != 'none'
                and norm.get('feature_skip_fusion') == 'compress_conv_bn_relu'
                and bool(norm.get('feature_skip_to_policy_head') or norm.get('feature_skip_to_value_head')))
    if key == 'policy_head_activation':
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
    function and one it lacks must hold DOES_NOT_APPLY — and a DOES_NOT_APPLY in
    any group's `activation_function` / `se_activation` raises. Every
    ArchitectureError names the key and the site."""
    raw = json.loads(arch) if isinstance(arch, str) else dict(arch)
    version = UNVERSIONED_LEGACY_VERSION if format_version is None else int(format_version)
    uniform = 'block_groups' not in raw
    legacy_allowed = uniform or version < SITE_ACTIVATIONS_REQUIRED_FROM_VERSION
    if not uniform and not legacy_allowed and 'activation_function' in raw:
        raise ArchitectureError(
            f"format v{version} architecture states the retired top-level 'activation_function'; "
            f"it is replaced by {', '.join(SITE_ACTIVATION_KEYS)}")
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
        if key in raw:
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
    for i, group in enumerate(norm['block_groups']):
        for field in ('activation_function', 'se_activation'):
            if group[field] == DOES_NOT_APPLY:
                raise ArchitectureError(
                    f"block_groups[{i}].{field} is 'does_not_apply', but that site exists whenever its group does")
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


def read_metadata(path):
    """The `__metadata__` dict of a .safetensors file (string values, as stored),
    reading only the header.

    The one header reader of the Python tooling (`dcm_lineage.read_metadata`
    delegates here). A damaged file is an ArchitectureError: a length prefix of
    zero or above MAX_HEADER_BYTES, a short read, a header that is not a JSON
    object, or one without a `__metadata__` object. Without the bound, a damaged
    prefix is a request to read up to 2^64 bytes, which fails as OverflowError or
    MemoryError — neither of which a caller catching ValueError expects."""
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
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        raise ArchitectureError(f"{path}: no __metadata__ object in the safetensors header")
    return metadata


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
