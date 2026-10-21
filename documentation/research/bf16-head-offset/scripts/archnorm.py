"""The research scripts' import name for the architecture reader.

A model file's architecture comes from the app (`scripts/dcm_arch.py` runs
`--describe-model` on the file), the one implementation of the format-version
rules; this module re-exports the readers under the name these scripts import.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "scripts"))

from dcm_arch import (  # noqa: E402
    ArchitectureError,
    architecture,
    require_architecture_of,
    require_relu,
    site_activations,
)

__all__ = ["ArchitectureError", "architecture", "require_architecture_of", "require_relu", "site_activations"]
