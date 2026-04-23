"""
moses/metrics/metrics.py
========================
All MOSES generation quality metrics — NumPy 2.x / Python 3.12 compatible.

PATCH CHANGES vs original:
  - Removed all pomegranate imports
  - scaffold_diversity() calls moses.metrics.scaffold (scipy-based)
  - pkg_resources → importlib.resources
  - numpy deprecations fixed (np.bool, np.int, np.float → bool, int, float)
  - rdkit import guard uses try/except instead of hard requirement
  - FCD uses fcd_torch directly (unchanged)
  - All type annotations use built-in types (Python 3.10+ style)
"""
from __future__ import annotations

import warnings
from typing import Callable, Iterable, List, Optional, Sequence, Union

import numpy as np
from scipy.spatial.distance import cdist as _cdist

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    from rdkit.DataStructs import BulkTanimotoSimilarity
    _RDKIT = True
except ImportError:
    _RDKIT = False
    warnings.warn("rdkit not available — chemistry metrics will be skipped",
                  ImportWarning, stacklevel=2)

try:
    import fcd
    _FCD = True
except ImportError:
    _FCD = False

from moses.metrics.scaffold import (
    scaffold_diversity as _scaffold_diversity,
    get_scaffold,
    FragmentMatcher,
)


# ─────────────────────────────────────────────────────────────────────────────
# Validity & uniqueness
# ─────────────────────────────────────────────────────────────────────────────

def mol_from_smiles(smiles: str):
    """Return RDKit Mol or None."""
    if not _RDKIT or not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def canonic_smiles(smiles: str) -> Optional[str]:
    """Canonicalise a SMILES string. Returns None if invalid."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def fraction_valid(smiles_list: Sequence[str], n_jobs: int = 1) -> float:
    """Fraction of molecules that parse as valid SMILES."""
    if not _RDKIT:
        return float("nan")
    if n_jobs == 1:
        valid = [mol_from_smiles(s) is not None for s in smiles_list]
    else:
        from joblib import Parallel, delayed
        valid = Parallel(n_jobs=n_jobs)(
            delayed(lambda s: mol_from_smiles(s) is not None)(s)
            for s in smiles_list)
    n = len(valid)
    return sum(valid) / n if n else 0.0


def fraction_unique(smiles_list: Sequence[str],
                     n: Optional[int] = None,
                     check_validity: bool = True) -> float:
    """
    Fraction of unique molecules in smiles_list.
    n: if given, divide by n instead of len(smiles_list)
    check_validity: if True, only count valid SMILES toward denominator
    """
    if check_validity and _RDKIT:
        valid = [canonic_smiles(s) for s in smiles_list]
        valid = [s for s in valid if s is not None]
    else:
        valid = list(smiles_list)

    unique = set(valid)
    denom  = n if n is not None else len(valid)
    return len(unique) / denom if denom else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Novelty
# ─────────────────────────────────────────────────────────────────────────────

def novelty(gen_smiles: Sequence[str],
             train_smiles: Sequence[str],
             n_jobs: int = 1) -> float:
    """
    Fraction of generated molecules not present in the training set.
    Both sets are canonicalised before comparison.
    """
    if not _RDKIT:
        return float("nan")

    if n_jobs == 1:
        gen_can   = {canonic_smiles(s) for s in gen_smiles}
        train_can = {canonic_smiles(s) for s in train_smiles}
    else:
        from joblib import Parallel, delayed
        gen_can   = set(Parallel(n_jobs=n_jobs)(
            delayed(canonic_smiles)(s) for s in gen_smiles))
        train_can = set(Parallel(n_jobs=n_jobs)(
            delayed(canonic_smiles)(s) for s in train_smiles))

    gen_can.discard(None)
    train_can.discard(None)
    if not gen_can:
        return 0.0
    novel = gen_can - train_can
    return len(novel) / len(gen_can)


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprint helpers
# ─────────────────────────────────────────────────────────────────────────────

def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    """Morgan fingerprint as a numpy float32 array."""
    if not _RDKIT:
        return None
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        # RDKit >=2022.09: use MorganGenerator (preferred, no deprecation warning)
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp  = gen.GetFingerprint(mol)
    except ImportError:
        # Fallback for older RDKit
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr


def morgan_fps(smiles_list: Sequence[str],
                radius: int = 2,
                n_bits: int = 2048,
                n_jobs: int = 1) -> np.ndarray:
    """Return (n_valid, n_bits) float32 matrix of Morgan fingerprints."""
    if n_jobs == 1:
        fps = [morgan_fp(s, radius, n_bits) for s in smiles_list]
    else:
        from joblib import Parallel, delayed
        fps = Parallel(n_jobs=n_jobs)(
            delayed(morgan_fp)(s, radius, n_bits) for s in smiles_list)
    valid = [f for f in fps if f is not None]
    if not valid:
        return np.zeros((0, n_bits), dtype=np.float32)
    return np.stack(valid, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Internal diversity
# ─────────────────────────────────────────────────────────────────────────────

def internal_diversity(smiles_list: Sequence[str],
                        n_jobs: int = 1,
                        device: str = "cpu",
                        fp_type: str = "morgan",
                        gen_fps: Optional[np.ndarray] = None,
                        p: int = 1) -> float:
    """
    Internal diversity of a set of molecules.
    Defined as mean pairwise (1 - Tanimoto) on Morgan fingerprints.

    Parameters
    ----------
    p : 1 = mean; 2 = sqrt(mean of squared distances) (IntDiv2)
    """
    if gen_fps is None:
        gen_fps = morgan_fps(smiles_list, n_jobs=n_jobs)
    if len(gen_fps) < 2:
        return 0.0

    # Subsample for speed on large sets
    if len(gen_fps) > 10_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(gen_fps), 10_000, replace=False)
        gen_fps = gen_fps[idx]

    dist = _cdist(gen_fps, gen_fps, metric="jaccard")
    n = len(dist)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dists = dist[mask]

    if p == 1:
        return float(dists.mean())
    elif p == 2:
        return float(np.sqrt((dists ** 2).mean()))
    else:
        return float((dists ** p).mean() ** (1 / p))


# ─────────────────────────────────────────────────────────────────────────────
# SNN (nearest-neighbour similarity)
# ─────────────────────────────────────────────────────────────────────────────

def SNN(gen_smiles: Sequence[str],
         ref_smiles: Sequence[str],
         n_jobs: int = 1,
         gpu: int = -1) -> float:
    """
    SNN score: mean of nearest-neighbour Tanimoto similarity
    between generated and reference molecules.
    """
    gen_fps = morgan_fps(gen_smiles, n_jobs=n_jobs)
    ref_fps = morgan_fps(ref_smiles, n_jobs=n_jobs)
    if len(gen_fps) == 0 or len(ref_fps) == 0:
        return float("nan")

    # 1 - jaccard distance = Tanimoto similarity for binary fps
    sim_matrix = 1.0 - _cdist(gen_fps, ref_fps, metric="jaccard")
    # For each generated molecule, find its most similar reference molecule
    nn_sim = sim_matrix.max(axis=1)
    return float(nn_sim.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Scaffold similarity
# ─────────────────────────────────────────────────────────────────────────────

def scaffold_similarity(gen_smiles: Sequence[str],
                          ref_smiles: Sequence[str],
                          n_jobs: int = 1) -> float:
    """
    Jaccard similarity between scaffold sets of generated vs reference.
    = |gen_scaffolds ∩ ref_scaffolds| / |gen_scaffolds ∪ ref_scaffolds|
    """
    if not _RDKIT:
        return float("nan")

    if n_jobs == 1:
        gen_sc  = {get_scaffold(s) for s in gen_smiles}
        ref_sc  = {get_scaffold(s) for s in ref_smiles}
    else:
        from joblib import Parallel, delayed
        gen_sc  = set(Parallel(n_jobs=n_jobs)(delayed(get_scaffold)(s) for s in gen_smiles))
        ref_sc  = set(Parallel(n_jobs=n_jobs)(delayed(get_scaffold)(s) for s in ref_smiles))

    gen_sc.discard(None)
    ref_sc.discard(None)
    if not gen_sc and not ref_sc:
        return 1.0
    intersection = len(gen_sc & ref_sc)
    union = len(gen_sc | ref_sc)
    return intersection / union if union else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# FCD (Fréchet ChemNet Distance)
# ─────────────────────────────────────────────────────────────────────────────

def FCD(gen_smiles: Sequence[str],
         ref_smiles: Sequence[str],
         device: str = "cpu") -> float:
    """
    Fréchet ChemNet Distance between generated and reference molecules.
    Lower = better (closer distribution to reference).
    Uses fcd_torch library (unchanged from original MOSES).
    """
    if not _FCD:
        warnings.warn("fcd_torch not installed — FCD metric unavailable. "
                      "Install with: pip install fcd_torch", stacklevel=2)
        return float("nan")
    import fcd as fcd_module
    try:
        score = fcd_module.get_fcd(list(gen_smiles), list(ref_smiles))
        return float(score)
    except Exception as e:
        warnings.warn(f"FCD computation failed: {e}", stacklevel=2)
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Filters (PAINS / alerts)
# ─────────────────────────────────────────────────────────────────────────────

_fragment_matcher = None

def _get_fragment_matcher() -> FragmentMatcher:
    global _fragment_matcher
    if _fragment_matcher is None:
        _fragment_matcher = FragmentMatcher()
    return _fragment_matcher


def filters(smiles_list: Sequence[str], n_jobs: int = 1) -> float:
    """
    Fraction of generated molecules passing all structural filters.
    (Inverse of fraction matching PAINS/alert patterns.)
    """
    fm = _get_fragment_matcher()
    return fm.fraction_valid(smiles_list, n_jobs=n_jobs)


# ─────────────────────────────────────────────────────────────────────────────
# Master get_all_metrics
# ─────────────────────────────────────────────────────────────────────────────

def get_all_metrics(gen: Sequence[str],
                     test: Optional[Sequence[str]] = None,
                     test_scaffolds: Optional[Sequence[str]] = None,
                     train: Optional[Sequence[str]] = None,
                     n_jobs: int = 1,
                     device: str = "cpu",
                     batch_size: int = 512,
                     pool: None = None,
                     gpu: int = -1,
                     n_ref: Optional[int] = None) -> dict:
    """
    Compute all MOSES metrics for a list of generated SMILES.

    Parameters
    ----------
    gen              : generated molecules (SMILES list)
    test             : test set (for SNN, FCD, novelty, scaffold_similarity)
    test_scaffolds   : test scaffold set (if different from test)
    train            : training set (for novelty)
    n_jobs           : parallel jobs for RDKit operations
    device           : 'cpu' or 'cuda' for FCD

    Returns dict of metric_name → float value.
    """
    metrics = {}

    # --- Validity ---
    metrics["valid"] = fraction_valid(gen, n_jobs=n_jobs)

    # Filter to valid only for remaining metrics
    valid_gen = [s for s in gen if mol_from_smiles(s) is not None]
    if not valid_gen:
        metrics.update({
            "unique@1000": 0.0, "unique@10000": 0.0,
            "IntDiv": 0.0, "IntDiv2": 0.0,
            "Filters": 0.0, "Novelty": float("nan"),
            "SNN/test": float("nan"), "SNN/test_scaffolds": float("nan"),
            "FCD/test": float("nan"), "FCD/test_scaffolds": float("nan"),
            "Scaf/test": float("nan"), "Scaf/test_scaffolds": float("nan"),
        })
        return metrics

    # --- Uniqueness ---
    metrics["unique@1000"]  = fraction_unique(valid_gen, n=1000,  check_validity=False)
    metrics["unique@10000"] = fraction_unique(valid_gen, n=10000, check_validity=False)

    # --- Internal diversity ---
    gen_fps = morgan_fps(valid_gen, n_jobs=n_jobs)
    metrics["IntDiv"]  = internal_diversity(valid_gen, gen_fps=gen_fps, p=1)
    metrics["IntDiv2"] = internal_diversity(valid_gen, gen_fps=gen_fps, p=2)

    # --- Filters ---
    metrics["Filters"] = filters(valid_gen, n_jobs=n_jobs)

    # --- Novelty (requires train set) ---
    if train is not None:
        metrics["Novelty"] = novelty(valid_gen, train, n_jobs=n_jobs)
    else:
        metrics["Novelty"] = float("nan")

    # --- Reference-set metrics ---
    for ref_name, ref_set in [("test", test), ("test_scaffolds", test_scaffolds)]:
        if ref_set is None:
            metrics[f"SNN/{ref_name}"]  = float("nan")
            metrics[f"FCD/{ref_name}"]  = float("nan")
            metrics[f"Scaf/{ref_name}"] = float("nan")
            continue

        ref_valid = [s for s in ref_set if mol_from_smiles(s) is not None]

        metrics[f"SNN/{ref_name}"]  = SNN(valid_gen, ref_valid, n_jobs=n_jobs)
        metrics[f"FCD/{ref_name}"]  = FCD(valid_gen, ref_valid, device=device)
        metrics[f"Scaf/{ref_name}"] = scaffold_similarity(
            valid_gen, ref_valid, n_jobs=n_jobs)

    return metrics
