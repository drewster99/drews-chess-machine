"""The research scripts' import name for the architecture reader.

The version-gated architecture rules live in `scripts/dcm_arch.py` (the one
Python source for them, shared with the experiment and dashboard tooling); this
module re-exports them so `from archnorm import norm_arch` keeps working here.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "scripts"))

from dcm_arch import (  # noqa: E402
    REZERO_ALPHA_CAP_REQUIRED_FROM_VERSION,
    REZERO_TANH_CEILING_MULTIPLE,
    SE_ACTIVATION_REQUIRED_FROM_VERSION,
    SE_BETA_INIT_REQUIRED_FROM_VERSION,
    ArchitectureError,
    norm_arch,
    norm_arch_md,
)
