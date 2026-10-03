import json

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
# NetworkArchitecture.rezeroTanhCeilingMultiple: the legacy cap rule.
REZERO_TANH_CEILING_MULTIPLE = 1.0

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
    a=json.loads(s) if isinstance(s,str) else dict(s)
    if 'block_groups' not in a:
        a['block_groups']=[dict(count=a['num_blocks'],channels=a['channels'],conv1_kernel_size=a['block_conv1_kernel_size'],conv2_kernel_size=a['block_conv2_kernel_size'],se_style=a['block_se_style'],se_reduction_ratio=a['block_se_reduction_ratio'],use_rezero=a['block_use_rezero'],rezero_alpha_init=a['rezero_alpha_init'],activation_function=a['activation_function'],activation_style=a['block_activation_style'],skip_merge=a['block_skip_merge'],dropout_multiplier=1,se_beta_init='glorot',se_activation=a['activation_function'])]
    version = None if format_version is None else int(format_version)
    strict_beta = version is not None and version >= SE_BETA_INIT_REQUIRED_FROM_VERSION
    strict_se_act = version is not None and version >= SE_ACTIVATION_REQUIRED_FROM_VERSION
    strict_cap = version is not None and version >= REZERO_ALPHA_CAP_REQUIRED_FROM_VERSION
    for i, g in enumerate(a['block_groups']):
        if 'se_beta_init' not in g:
            if strict_beta:
                raise KeyError(f"block_groups[{i}].se_beta_init missing in a format v{format_version} architecture")
            g['se_beta_init'] = 'glorot'
        if 'se_activation' not in g:
            if strict_se_act:
                raise KeyError(f"block_groups[{i}].se_activation missing in a format v{format_version} architecture")
            g['se_activation'] = g['activation_function']
        if 'rezero_alpha_cap' not in g:
            if strict_cap:
                raise KeyError(f"block_groups[{i}].rezero_alpha_cap missing in a format v{format_version} architecture")
            g['rezero_alpha_cap'] = g['rezero_alpha_init'] * REZERO_TANH_CEILING_MULTIPLE
    a.setdefault('feature_skip_source','none')
    return a

def norm_arch_md(md):
    """norm_arch for a safetensors __metadata__ dict, gated on its dcm_format_version."""
    return norm_arch(md['architecture'], md.get('dcm_format_version'))
