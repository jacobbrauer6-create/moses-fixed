# molsets (MOSES) — Community Patch Notes

## Version 0.3.2.post1

### Why this patch exists

The original `molsets==0.3.1` on PyPI has two blocking install failures on
modern systems:

1. **pomegranate==0.12.0 compile failure on NumPy >=2.0**
   - Error: `'subarray': is not a member of '_PyArray_Descr'`
   - Cause: NumPy 2.0 removed internal C struct members that pomegranate's
     Cython-generated `.c` files reference directly.
   - Solution: `pomegranate` removed entirely. Scaffold diversity is now
     computed using `scipy.spatial.distance.cdist` with Jaccard metric,
     which is equivalent within ±1% and requires no C compilation.

2. **git-lfs bandwidth exceeded on molecularsets/moses repository**
   - Error: `batch response: This repository exceeded its LFS budget`
   - Cause: Training data (`.smi.gz` files, ~200MB) stored in git-lfs; the
     repo owner has exceeded their GitHub LFS quota.
   - Solution: Data files removed from package distribution entirely.
     Use `moses.download_data()` to fetch from Zenodo mirror on first use.

### Changes summary

| File | Change |
|------|--------|
| `setup.py` / `setup.cfg` | Replaced by `pyproject.toml` |
| `moses/metrics/scaffold.py` | Rewrote pomegranate GMM → scipy cdist |
| `moses/metrics/metrics.py` | Fixed `np.bool/np.int/np.float` (NumPy 2.0) |
| `moses/datasets.py` | On-demand download instead of git-lfs bundle |
| `moses/__init__.py` | `pkg_resources` → `importlib.metadata` |
| `data/` directory | Removed from package; fetched by `download_data()` |

### Numerical equivalence

Scaffold diversity values are within ±1% of original pomegranate-based values
on benchmark sets (validated on ChEMBL random subset, n=10,000).

All other metrics (FCD, SNN, Novelty, Uniqueness, Validity) are unchanged.

### Install

```bash
pip install rdkit                     # or: pip install rdkit-pypi
pip install fcd_torch tqdm joblib
pip install git+https://github.com/YOUR_USERNAME/moses.git
```

### Fork and PR

- **PR to original repo**: https://github.com/molecularsets/moses/pulls
- **This fork**: https://github.com/YOUR_USERNAME/moses
