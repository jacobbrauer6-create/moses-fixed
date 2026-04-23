"""
moses/datasets.py
=================
Dataset loading with on-demand download.

PATCH: Original MOSES stored training data in git-lfs (data/*.smi.gz).
The repo has exceeded its LFS bandwidth quota, making pip install fail.
This version downloads data files on first use from a mirror, with a
clear local cache in ~/.cache/moses/. No data is bundled in the package.

Mirror URLs are provided in order of preference:
  1. Zenodo archive (permanent DOI)
  2. Original GitHub release assets
  3. Fallback: user must provide own dataset
"""
from __future__ import annotations

import gzip
import hashlib
import os
from pathlib import Path
from typing import List, Optional
from urllib.request import urlretrieve
from urllib.error import URLError

# ── Known dataset files ───────────────────────────────────────────────────────
_CACHE_DIR = Path(os.environ.get("MOSES_CACHE", Path.home() / ".cache" / "moses"))

_DATASETS = {
    "train": {
        "filename": "train.csv.gz",
        # Zenodo archive for MOSES dataset
        # DOI: 10.5281/zenodo.7558701 (community mirror)
        # If this URL changes, update here; the md5 checksum is the truth
        "urls": [
            "https://zenodo.org/records/7558701/files/train.csv.gz",
            "https://github.com/molecularsets/moses/releases/download/v1.0/train.csv.gz",
        ],
        "md5":  None,   # set after first download verification
        "n_molecules": 1_584_663,
    },
    "test": {
        "filename": "test.csv.gz",
        "urls": [
            "https://zenodo.org/records/7558701/files/test.csv.gz",
            "https://github.com/molecularsets/moses/releases/download/v1.0/test.csv.gz",
        ],
        "md5":  None,
        "n_molecules": 176_074,
    },
    "test_scaffolds": {
        "filename": "test_scaffolds.csv.gz",
        "urls": [
            "https://zenodo.org/records/7558701/files/test_scaffolds.csv.gz",
            "https://github.com/molecularsets/moses/releases/download/v1.0/test_scaffolds.csv.gz",
        ],
        "md5":  None,
        "n_molecules": 176_074,
    },
}


def _cache_path(filename: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / filename


def _download(url: str, dest: Path, desc: str = "") -> bool:
    """Download url to dest. Returns True on success."""
    try:
        print(f"  Downloading {desc or url} → {dest}")
        urlretrieve(url, str(dest))
        return True
    except (URLError, OSError) as e:
        print(f"  Failed ({e})")
        return False


def download_data(dataset: str = "all", force: bool = False) -> None:
    """
    Download MOSES dataset files to ~/.cache/moses/.

    Parameters
    ----------
    dataset : "all" | "train" | "test" | "test_scaffolds"
    force   : re-download even if file exists
    """
    targets = list(_DATASETS.keys()) if dataset == "all" else [dataset]
    for name in targets:
        info = _DATASETS.get(name)
        if not info:
            print(f"Unknown dataset: {name}. Available: {list(_DATASETS.keys())}")
            continue
        dest = _cache_path(info["filename"])
        if dest.exists() and not force:
            print(f"  {name}: already cached at {dest}")
            continue
        print(f"Downloading MOSES {name} dataset (~{info['n_molecules']:,} molecules)...")
        ok = False
        for url in info["urls"]:
            if _download(url, dest, name):
                ok = True
                break
        if not ok:
            print(f"  Could not download {name}. Please download manually and place at {dest}")
            print(f"  Tried: {info['urls']}")


def get_dataset(name: str = "test",
                 auto_download: bool = True) -> List[str]:
    """
    Load a MOSES dataset as a list of SMILES strings.

    Parameters
    ----------
    name           : "train" | "test" | "test_scaffolds"
    auto_download  : download if not cached (default True)

    Returns list of SMILES strings.
    """
    info = _DATASETS.get(name)
    if not info:
        raise ValueError(f"Unknown dataset: {name!r}. "
                          f"Available: {list(_DATASETS.keys())}")

    dest = _cache_path(info["filename"])
    if not dest.exists():
        if not auto_download:
            raise FileNotFoundError(
                f"Dataset {name!r} not found at {dest}. "
                f"Run moses.download_data('{name}') to fetch it.")
        download_data(name)

    if not dest.exists():
        raise FileNotFoundError(
            f"Dataset {name!r} could not be downloaded. "
            f"Please obtain it manually and place at {dest}")

    return _load_smiles_gz(dest)


def _load_smiles_gz(path: Path) -> List[str]:
    """Load SMILES from a gzipped CSV (first column = SMILES, optional header)."""
    smiles = []
    with gzip.open(str(path), "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Skip header if it looks like 'SMILES' or 'smiles'
            if i == 0 and line.upper().startswith("SMILES"):
                continue
            # CSV: take first column only
            smiles.append(line.split(",")[0])
    return smiles
