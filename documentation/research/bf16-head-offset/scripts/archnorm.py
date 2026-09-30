import json

# First DCM file format (safetensors `dcm_format_version`) whose block groups
# must state `se_beta_init`; older files predate the field and mean 'glorot'.
SE_BETA_INIT_REQUIRED_FROM_VERSION = 4

def norm_arch(s, format_version=None):
    """Normalize a DCM architecture JSON (string or dict) to the block-groups form.

    format_version: the carrier's `dcm_format_version` (string or int). Files of
    version >= SE_BETA_INIT_REQUIRED_FROM_VERSION must carry `se_beta_init` on
    every block group (a missing one raises, mirroring the Swift loader); older
    or unversioned files resolve a missing value to 'glorot'. Prefer
    `norm_arch_md(md)`, which reads the version from the file's metadata.
    """
    a=json.loads(s) if isinstance(s,str) else dict(s)
    if 'block_groups' not in a:
        a['block_groups']=[dict(count=a['num_blocks'],channels=a['channels'],conv1_kernel_size=a['block_conv1_kernel_size'],conv2_kernel_size=a['block_conv2_kernel_size'],se_style=a['block_se_style'],se_reduction_ratio=a['block_se_reduction_ratio'],use_rezero=a['block_use_rezero'],rezero_alpha_init=a['rezero_alpha_init'],activation_function=a['activation_function'],activation_style=a['block_activation_style'],skip_merge=a['block_skip_merge'],dropout_multiplier=1,se_beta_init='glorot')]
    strict = format_version is not None and int(format_version) >= SE_BETA_INIT_REQUIRED_FROM_VERSION
    for i, g in enumerate(a['block_groups']):
        if 'se_beta_init' not in g:
            if strict:
                raise KeyError(f"block_groups[{i}].se_beta_init missing in a format v{format_version} architecture")
            g['se_beta_init'] = 'glorot'
    a.setdefault('feature_skip_source','none')
    return a

def norm_arch_md(md):
    """norm_arch for a safetensors __metadata__ dict, gated on its dcm_format_version."""
    return norm_arch(md['architecture'], md.get('dcm_format_version'))
