"""
Dual-Space Entropy-Complexity Framework for Unsupervised Multivariate
Concept Drift Detection.

An extension of QCV-HxC (Barbosa Costa et al., IEEE Access 2026) from univariate
to multivariate streams, adding a second (autoencoder latent) representation
space and a dual-space score fusion with an adaptive baseline threshold.

Public API
----------
    DualSpaceDriftDetector : end-to-end detector (fit_detect).
    DualSpaceConfig        : configuration dataclass.
    data_loader            : synthetic generators + real-CSV loader.
"""

from .config import DualSpaceConfig
from .detector import DualSpaceDriftDetector, DualSpaceResult
from . import data_loader

__all__ = [
    "DualSpaceConfig",
    "DualSpaceDriftDetector",
    "DualSpaceResult",
    "data_loader",
]

__version__ = "1.0.0"
