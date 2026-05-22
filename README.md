# ssimpy

**ssimpy** is a Python command-line tool for detecting statistically significant pairwise somatic co-mutations in cancer genomics data. It implements the SelectSim algorithm, testing both **co-occurrence** (two genes mutated together more than expected) and **mutual exclusivity** (two genes rarely co-mutated) using a simulation-based FDR framework.

## Features

- **rcRAS simulation engine** — generates null GAMs that exactly preserve each gene's observed mutation count
- **TMB-aware penalty** — down-weights hypermutated samples to avoid signal dominated by outlier tumors
- **Bidirectional FDR** — separately calibrated FDR for co-occurrence and mutual exclusivity
- **Sample class covariates** — per-subtype expected mutation matrices (e.g. LUAD vs LUSC)
- **Mutation type covariates** — element-wise max of per-type E matrices (e.g. missense + truncating)

## Installation

Requires Python ≥ 3.9 and `numpy` + `pandas`:

```bash
pip install numpy pandas
```

Clone the repository and run from within the `ssimpy/` directory:

```bash
git clone https://github.com/CSOgroup/ssimpy.git
cd ssimpy
python ssimpy.py --gam <gam.tsv> [options]
```

## Input files

| File | Required | Description |
|------|----------|-------------|
| GAM (TSV) | Yes | Binary matrix: rows = genes, columns = samples |
| TMB (TSV) | No | Per-sample mutation burden; columns: `sample`, `tmb`/`mutation`, optional `class` |
| MAF (TSV) | No | Mutation Annotation Format file (alternative TMB source, single-GAM mode only) |

When the TMB file includes a `class` column, ssimpy automatically computes separate expected mutation matrices per class.

## Usage

### Minimal run
```bash
python ssimpy.py --gam lung_gam.tsv --output results.tsv
```

### With pre-calculated TMB and sample class covariates
```bash
python ssimpy.py \
    --gam lung_gam_complete.tsv \
    --tmb lung_tmb_complete.tsv \
    --N 1000 --seed 42 \
    --output results.tsv
```

### Multiple mutation types (missense + truncating)
```bash
python ssimpy.py \
    --gam lung_gam_missense.tsv lung_gam_truncating.tsv \
    --tmb lung_tmb_missense.tsv lung_tmb_truncating.tsv \
    --N 1000 --seed 42 \
    --output results_multitype.tsv
```

All GAM files must share identical genes and samples. The combined GAM is their element-wise union; the expected matrix E is the element-wise maximum across per-type E matrices.

## Key parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--N` | 1000 | Number of rcRAS simulations |
| `--min-mut` | 5 | Minimum mutated samples to retain a gene |
| `--fdr` | 0.1 | FDR threshold for significance calls |
| `--tau` | 1.0 | TMB fold-change threshold for penalization |
| `--lam` | 0.3 | Rate of TMB-based penalization |
| `--filter-pct` | 0.10 | Fraction of worst simulations to discard |
| `--seed` | None | Random seed for reproducibility |
| `--output` | selectsim_results.tsv | Output file path |

## Output

A TSV file with one row per gene pair, sorted by `|nES|` descending. Key columns:

| Column | Description |
|--------|-------------|
| `gene1`, `gene2` | Gene pair |
| `n_comut` | Samples mutated in both genes |
| `nES` | Normalized effect size (+ co-occurrence, − mutual exclusivity) |
| `direction` | `co-occurrence` or `mutual_exclusivity` |
| `FDR` | Estimated false discovery rate |
| `significant` | `True` if FDR < threshold |

## Reference

ssimpy is the Python implementation of the SelectSim methodology. See also:
- [SelectSim R package](https://github.com/CSOgroup/SelectSim)
- [SelectSim analysis scripts](https://github.com/CSOgroup/SelectSim_analysis)
