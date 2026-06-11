"""ZATRON — Zero-Access Transformed Retrieval Over Noise.

Privacy-preserving semantic search via multi-channel modular arithmetic.
Search sensitive documents by meaning without exposing content.
"""
from .core import (
    ModularBarcodeSystem,
    SecurityAuditor,
    AccessPatternGuard,
    select_primes,
    auto_config,
)

__version__ = "0.1.0"
__all__ = [
    "ModularBarcodeSystem",
    "SecurityAuditor",
    "AccessPatternGuard",
    "select_primes",
    "auto_config",
]
