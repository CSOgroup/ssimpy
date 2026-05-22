import numpy as np
import pandas as pd


def load_gam(path: str):
    """Load binary GAM from a TSV file (rows=genes, columns=samples).

    Returns:
        gam         : np.ndarray [n_g x n_t], int8
        gene_names  : list of str
        sample_names: list of str
    """
    df = pd.read_csv(path, sep='\t', index_col=0)
    gene_names = list(df.index)
    sample_names = list(df.columns)
    gam = df.values.astype(np.int8)
    return gam, gene_names, sample_names


_SILENT_CLASSES = {'silent', 'synonymous'}


def load_maf(path: str, sample_names: list) -> np.ndarray:
    """Count non-silent mutations per sample from a MAF file.

    Required columns (case-insensitive):
        Tumor_Sample_Barcode    — identifies the sample for each mutation
        Variant_Classification  — mutation type; rows classified as
                                  'Silent' or 'Synonymous' are excluded

    Samples listed in sample_names but absent from the MAF receive TMB = 0.

    Returns:
        tmb: np.ndarray [n_t], float — non-silent mutation counts per sample,
             aligned to sample_names order.
    """
    maf = pd.read_csv(path, sep='\t', comment='#', low_memory=False)

    # Normalise column names to lower-case for robust matching
    maf.columns = maf.columns.str.lower()

    for col in ('tumor_sample_barcode', 'variant_classification'):
        if col not in maf.columns:
            raise ValueError(
                f"MAF file is missing required column '{col}'. "
                f"Columns found: {list(maf.columns)}"
            )

    # Keep only non-silent / non-synonymous mutations
    non_silent = ~maf['variant_classification'].str.lower().isin(_SILENT_CLASSES)
    maf = maf[non_silent]

    counts = maf['tumor_sample_barcode'].value_counts()
    return np.array([counts.get(s, 0) for s in sample_names], dtype=float)


def load_tmb(path: str, sample_names: list) -> tuple:
    """Load pre-calculated TMB values from a TSV file.

    Required columns (case-insensitive):
        sample  — sample identifier
        tmb     — pre-calculated TMB value

    Optional column:
        class   — sample subgroup label; when present, the E matrix and
                  simulations are computed separately per class and then
                  concatenated (Step 1-bis of the SelectSim algorithm).

    Samples listed in sample_names but absent from the file receive TMB = 0
    and, if classes are used, are assigned to class 'unknown'.

    Returns:
        tmb    : np.ndarray [n_t], float — TMB values aligned to sample_names
        classes: np.ndarray [n_t], str   — class labels, or None if no 'class' column
    """
    df = pd.read_csv(path, sep='\t')
    df.columns = df.columns.str.lower()

    if 'sample' not in df.columns:
        raise ValueError(
            f"TMB file is missing required column 'sample'. "
            f"Columns found: {list(df.columns)}"
        )

    tmb_col = next((c for c in ('tmb', 'mutation') if c in df.columns), None)
    if tmb_col is None:
        raise ValueError(
            f"TMB file must contain a 'tmb' or 'mutation' column. "
            f"Columns found: {list(df.columns)}"
        )

    tmb_map = df.set_index('sample')[tmb_col].to_dict()
    tmb = np.array([tmb_map.get(s, 0) for s in sample_names], dtype=float)

    if 'class' in df.columns:
        class_map = df.set_index('sample')['class'].to_dict()
        classes = np.array([str(class_map.get(s, 'unknown')) for s in sample_names])
    else:
        classes = None

    return tmb, classes


def tmb_from_gam(gam: np.ndarray) -> np.ndarray:
    """Estimate TMB as column sums of the GAM (fallback when no external TMB is given)."""
    return gam.sum(axis=0).astype(float)


def save_results(df: pd.DataFrame, path: str) -> None:
    """Write results DataFrame to a TSV file."""
    df.to_csv(path, sep='\t', index=False)
