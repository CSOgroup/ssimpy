import numpy as np
import pandas as pd

from comutation import compute_weighted_comutation

_SIN45 = np.sin(np.pi / 4)   # = sqrt(2)/2, the Euclidean distance scaling factor

# Default cumulative-frequency bins for stratified FDR estimation
DEFAULT_FREQ_INTERVALS = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 1.01)]


# ---------------------------------------------------------------------------
# Effect-size computation
# ---------------------------------------------------------------------------

def _compute_wCO_list(sims: list, p: np.ndarray) -> list:
    """Compute weighted co-mutation matrix for each simulated GAM."""
    return [compute_weighted_comutation(S, p) for S in sims]


def compute_wES(wCO: np.ndarray, wCO_exp: np.ndarray) -> np.ndarray:
    """wES = (wCO - wCO_exp) * sin(pi/4).

    The scalar sin(pi/4) converts the difference along both axes into the
    Euclidean distance from the diagonal wCO = wCO_exp.
    """
    return (wCO - wCO_exp) * _SIN45


def compute_nES(
    wES_obs: np.ndarray,
    wES_sims: list,
) -> tuple:
    """Compute normalized effect sizes for the observed data and each simulation.

    The normalization subtracts the mean absolute null effect size so that nES
    values are comparable across gene pairs with different mutation frequencies.

        wES_exp_abs = mean_i( |wES_sim_i| )
        nES = sign(wES) * max(|wES| - wES_exp_abs, 0)

    Args:
        wES_obs : [n_g x n_g] observed weighted effect size matrix
        wES_sims: list of N [n_g x n_g] simulated weighted effect size matrices

    Returns:
        nES_obs : [n_g x n_g] normalized observed effect sizes
        nES_sims: list of N [n_g x n_g] normalized simulated effect sizes
    """
    wES_arr = np.stack(wES_sims, axis=0)               # [N x n_g x n_g]
    wES_exp_abs = np.mean(np.abs(wES_arr), axis=0)     # [n_g x n_g] mean null magnitude

    def _normalize(wES):
        return np.sign(wES) * np.maximum(np.abs(wES) - wES_exp_abs, 0.0)

    nES_obs  = _normalize(wES_obs)
    nES_sims = [_normalize(w) for w in wES_sims]
    return nES_obs, nES_sims


# ---------------------------------------------------------------------------
# Frequency categories and upper-triangle index helpers
# ---------------------------------------------------------------------------

def _upper_triangle_indices(n_g: int):
    return np.triu_indices(n_g, k=1)


def assign_freq_categories(
    gene_freq: np.ndarray,
    n_g: int,
    intervals: list = None,
) -> tuple:
    """Assign each gene pair (upper triangle) to a cumulative-frequency category.

    Args:
        gene_freq: [n_g] mutation frequency per gene
        n_g      : number of genes
        intervals: list of (lo, hi) half-open intervals (lo, hi]; default is
                   [(0,0.02),(0.02,0.05),(0.05,0.10),(0.10,1.01)]

    Returns:
        i_idx     : row indices of upper-triangle pairs
        j_idx     : col indices of upper-triangle pairs
        categories: [n_pairs] int, category index for each pair (-1 if unassigned)
    """
    if intervals is None:
        intervals = DEFAULT_FREQ_INTERVALS
    i_idx, j_idx = _upper_triangle_indices(n_g)
    cum_freq = gene_freq[i_idx] + gene_freq[j_idx]
    categories = np.full(len(i_idx), -1, dtype=int)
    for cat_idx, (lo, hi) in enumerate(intervals):
        mask = (cum_freq > lo) & (cum_freq <= hi)
        categories[mask] = cat_idx
    return i_idx, j_idx, categories


# ---------------------------------------------------------------------------
# FDR estimation
# ---------------------------------------------------------------------------

def _fdr_one_direction(
    obs: np.ndarray,
    sims_cat: np.ndarray,
) -> np.ndarray:
    """Compute per-pair FDR for the positive tail (obs > 0) within one category.

    For each pair with observed nES = x > 0:
        TP(x) = #{observed pairs with nES > x}   (across all pairs in category)
        FP(x) = mean over simulations of #{pairs with nES_sim > x}
        FDR(x) = FP(x) / (TP(x) + FP(x))

    Pairs with nES <= 0 receive FDR = 1.0.
    FDR is monotone-corrected to be non-decreasing as threshold decreases.

    Args:
        obs     : [n_pairs_cat] nES values (signs already oriented for this direction)
        sims_cat: [N_sims x n_pairs_cat] simulated nES values (same orientation)

    Returns:
        fdr: [n_pairs_cat] FDR values in [0, 1]
    """
    n_pairs = len(obs)
    fdr = np.ones(n_pairs)

    if not np.any(obs > 0):
        return fdr

    order = np.argsort(-obs)
    sorted_obs = obs[order]

    sims_sorted_asc = np.sort(sims_cat, axis=1)   # [N_sims x n_pairs_cat]
    N_sims = sims_cat.shape[0]

    n_pos = int(np.searchsorted(-sorted_obs, 0, side='left'))
    if n_pos == 0:
        return fdr

    thresholds = sorted_obs[:n_pos]   # [n_pos] descending, all > 0

    sim_counts = np.empty((N_sims, n_pos), dtype=np.float64)
    for n in range(N_sims):
        insertions = np.searchsorted(sims_sorted_asc[n], thresholds, side='right')
        sim_counts[n] = n_pairs - insertions

    mean_fp      = sim_counts.mean(axis=0)
    discoveries  = np.arange(1, n_pos + 1, dtype=np.float64)  # #{observed pairs > alpha}
    raw_fdr      = np.ones(n_pairs)
    raw_fdr[:n_pos] = np.minimum(mean_fp / discoveries, 1.0)

    # Monotone: FDR non-decreasing as threshold decreases (cumulative max)
    raw_fdr[:n_pos] = np.maximum.accumulate(raw_fdr[:n_pos])

    fdr[order] = raw_fdr
    return fdr


def _fdr_one_category(
    obs: np.ndarray,
    sims_cat: np.ndarray,
) -> np.ndarray:
    """Compute per-pair FDR using absolute nES values (both tails pooled).

    Matches R's estimateFDR2: co-occurrence and mutual exclusivity compete
    against the same null magnitude distribution. A pair's FDR reflects how
    often any simulation produces an effect at least as large in either
    direction, so weak ME pairs are not privileged by having a sparser
    one-sided null.

    Pairs with nES = 0 receive FDR = 1.0.

    Args:
        obs     : [n_pairs_cat] observed nES values
        sims_cat: [N_sims x n_pairs_cat] simulated nES values

    Returns:
        fdr: [n_pairs_cat] FDR values in [0, 1]
    """
    return _fdr_one_direction(np.abs(obs), np.abs(sims_cat))


def estimate_fdr(
    nES_obs_pairs: np.ndarray,
    nES_sims_pairs: list,
    categories: np.ndarray,
    n_cats: int,
) -> np.ndarray:
    """Estimate per-pair FDR values across all frequency categories.

    Args:
        nES_obs_pairs : [n_pairs] observed nES for each upper-triangle gene pair
        nES_sims_pairs: list of N [n_pairs] arrays, one per simulation
        categories    : [n_pairs] int category index for each pair
        n_cats        : total number of frequency categories

    Returns:
        fdr_values: [n_pairs] FDR estimates (1.0 for non-positive or uncategorized pairs)
    """
    fdr_values = np.ones(len(nES_obs_pairs))
    sims_stack = np.stack(nES_sims_pairs, axis=0)   # [N_sims x n_pairs]

    for cat in range(n_cats):
        cat_mask = categories == cat
        if not np.any(cat_mask):
            continue
        obs_cat  = nES_obs_pairs[cat_mask]
        sims_cat = sims_stack[:, cat_mask]           # [N_sims x n_pairs_cat]
        fdr_values[cat_mask] = _fdr_one_category(obs_cat, sims_cat)

    return fdr_values


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_significance_analysis(
    wCO_obs: np.ndarray,
    sims: list,
    p: np.ndarray,
    gene_freq: np.ndarray,
    gene_names: list,
    gam: np.ndarray,
    target_fdr: float = 0.1,
    freq_intervals: list = None,
) -> pd.DataFrame:
    """Run the full Steps 2-3 pipeline and return a results DataFrame.

    Args:
        wCO_obs      : [n_g x n_g] observed weighted co-mutation matrix
        sims         : list of N filtered simulated [n_g x n_t] int8 GAMs
        p            : [n_t] penalty vector
        gene_freq    : [n_g] gene mutation frequencies
        gene_names   : list of n_g gene name strings
        gam          : [n_g x n_t] original binary GAM (for raw co-mutation counts)
        target_fdr   : significance FDR threshold (default 0.1)
        freq_intervals: cumulative-frequency bins; uses DEFAULT_FREQ_INTERVALS if None

    Returns:
        DataFrame with columns:
            gene1, gene2, n_mut_gene1, n_mut_gene2, n_comut,
            freq_gene1, freq_gene2, cum_freq, freq_cat,
            wCO_obs, wCO_exp, wES, wES_exp, nES, FDR, significant
        Sorted by nES descending.
    """
    if freq_intervals is None:
        freq_intervals = DEFAULT_FREQ_INTERVALS
    n_cats = len(freq_intervals)
    n_g = wCO_obs.shape[0]
    n_t = gam.shape[1]

    # --- wCO for each simulation, then compute mean (wCO_exp) ---
    wCO_sims_list = _compute_wCO_list(sims, p)
    wCO_exp = np.mean(np.stack(wCO_sims_list, axis=0), axis=0)   # [n_g x n_g]

    # --- wES ---
    wES_obs  = compute_wES(wCO_obs, wCO_exp)
    wES_sims = [compute_wES(w, wCO_exp) for w in wCO_sims_list]

    # --- nES (also returns per-sim nES for FDR; wES_exp_abs is the noise floor) ---
    wES_arr = np.stack(wES_sims, axis=0)
    wES_exp_abs = np.mean(np.abs(wES_arr), axis=0)   # [n_g x n_g] mean null magnitude
    nES_obs, nES_sims = compute_nES(wES_obs, wES_sims)

    # --- Gene pair indices, raw counts, and frequency categories ---
    i_idx, j_idx, categories = assign_freq_categories(gene_freq, n_g, freq_intervals)

    gene_counts = gam.sum(axis=1)                        # [n_g] int
    raw_comut   = (gam.astype(np.int32) @ gam.T)        # [n_g x n_g] unweighted co-mutation counts

    # --- Extract upper-triangle values ---
    nES_obs_pairs  = nES_obs[i_idx, j_idx]
    nES_sims_pairs = [m[i_idx, j_idx] for m in nES_sims]

    # --- FDR ---
    fdr_values = estimate_fdr(nES_obs_pairs, nES_sims_pairs, categories, n_cats)

    # --- Build output ---
    gene_names_arr = np.array(gene_names)
    direction = np.where(
        nES_obs_pairs > 0, 'co-occurrence',
        np.where(nES_obs_pairs < 0, 'mutual_exclusivity', 'none')
    )
    results = pd.DataFrame({
        'gene1'            : gene_names_arr[i_idx],
        'gene2'            : gene_names_arr[j_idx],
        'n_mut_gene1'      : gene_counts[i_idx],
        'n_mut_gene2'      : gene_counts[j_idx],
        'n_comut'          : raw_comut[i_idx, j_idx],
        'freq_gene1'       : gene_freq[i_idx].round(4),
        'freq_gene2'       : gene_freq[j_idx].round(4),
        'cum_freq'         : (gene_freq[i_idx] + gene_freq[j_idx]).round(4),
        'freq_cat'         : categories,
        'wCO_obs'          : wCO_obs[i_idx, j_idx].round(4),
        'wCO_exp'          : wCO_exp[i_idx, j_idx].round(4),
        'wES'              : wES_obs[i_idx, j_idx].round(4),
        'wES_exp'          : wES_exp_abs[i_idx, j_idx].round(4),
        'nES'              : nES_obs_pairs.round(4),
        'direction'        : direction,
        'FDR'              : fdr_values.round(4),
        'significant'      : fdr_values < target_fdr,
    })
    # Sort by |nES| descending so co-occurrence and mutual exclusivity are interleaved
    results['abs_nES'] = results['nES'].abs()
    results = results.sort_values('abs_nES', ascending=False).drop(columns='abs_nES')
    return results.reset_index(drop=True)
