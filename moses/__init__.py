"""
MOSES — Molecular Sets benchmarking platform.

Community-patched fork (molsets 0.3.2.post1):
  - NumPy 2.x compatible
  - Python 3.10 / 3.11 / 3.12 compatible
  - pomegranate removed (replaced with scipy-based scaffold diversity)
  - pkg_resources removed (replaced with importlib.metadata)
  - Training data not bundled; use download_data() to fetch separately
"""

# ── version ───────────────────────────────────────────────────────────────────
try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("molsets")
    except PackageNotFoundError:
        __version__ = "0.3.2.post1"
except ImportError:
    __version__ = "0.3.2.post1"

# ── public API ────────────────────────────────────────────────────────────────
from moses.metrics import get_all_metrics
from moses.metrics.metrics import (
    fraction_valid,
    fraction_unique,
    novelty,
    internal_diversity,
    SNN,
    scaffold_similarity,
    FCD,
    filters,
)
from moses.datasets import get_dataset, download_data

__all__ = [
    "__version__",
    "get_all_metrics",
    "fraction_valid",
    "fraction_unique",
    "novelty",
    "internal_diversity",
    "SNN",
    "scaffold_similarity",
    "FCD",
    "filters",
    "get_dataset",
    "download_data",
]
