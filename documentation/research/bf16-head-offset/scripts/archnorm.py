import json
def norm_arch(s):
    a=json.loads(s) if isinstance(s,str) else dict(s)
    if 'block_groups' not in a:
        a['block_groups']=[dict(count=a['num_blocks'],channels=a['channels'],conv1_kernel_size=a['block_conv1_kernel_size'],conv2_kernel_size=a['block_conv2_kernel_size'],se_style=a['block_se_style'],se_reduction_ratio=a['block_se_reduction_ratio'],use_rezero=a['block_use_rezero'],rezero_alpha_init=a['rezero_alpha_init'],activation_function=a['activation_function'],activation_style=a['block_activation_style'],skip_merge=a['block_skip_merge'],dropout_multiplier=1)]
    a.setdefault('feature_skip_source','none')
    return a
