import numpy as np


def simulate_rcras(
    E: np.ndarray,
    gene_counts: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """One rcRAS simulation: generates a binary matrix that exactly preserves
    each gene's observed mutation count (row sums).

    Algorithm:
        S_hat = E - Uniform(0,1)
        For each gene row i, keep exactly gene_counts[i] entries as 1,
        chosen as the top-gene_counts[i] values of S_hat[i].

    Args:
        E          : [n_g x n_t] expected mutation probability matrix
        gene_counts: [n_g] int, observed number of mutated samples per gene
        rng        : numpy random Generator (shared across calls for reproducibility)

    Returns:
        S: [n_g x n_t] int8 binary simulated GAM
    """
    S_hat = E - rng.uniform(size=E.shape)

    # For each row, rank values descending; keep rank < gene_counts[i] as 1.
    # Double-argsort trick: rank_pos[i,j] = rank of column j in desc order of row i.
    desc_order = np.argsort(-S_hat, axis=1)          # [n_g x n_t]
    rank_pos   = np.argsort(desc_order, axis=1)       # [n_g x n_t]
    S = (rank_pos < gene_counts[:, np.newaxis]).astype(np.int8)
    return S


def run_simulations(
    E: np.ndarray,
    N: int,
    gene_counts: np.ndarray = None,
    seed: int = None,
    class_slices: list = None,
) -> list:
    """Generate N rcRAS simulated GAMs.

    Without class covariates (class_slices=None):
        Each simulation is a single rcRAS draw over the full [n_g x n_t] matrix,
        preserving global gene mutation counts (gene_counts).

    With class covariates (class_slices provided):
        Each simulation draws one rcRAS block per class independently, preserving
        class-specific gene mutation counts, then concatenates the blocks.
        gene_counts is ignored in this case.

    Args:
        E           : [n_g x n_t] expected mutation probability matrix
                      (class-sorted when class_slices is provided)
        N           : number of simulations to generate
        gene_counts : [n_g] int, global gene mutation counts (no-class case)
        seed        : optional random seed for reproducibility
        class_slices: list of (col_start, col_end, gene_counts_k) tuples,
                      one per class (returned by compute_E_matrix_by_class)

    Returns:
        list of N np.ndarray [n_g x n_t] int8 simulated GAMs
    """
    rng = np.random.default_rng(seed)

    if class_slices is None:
        return [simulate_rcras(E, gene_counts, rng) for _ in range(N)]

    def _one_sim():
        blocks = [
            simulate_rcras(E[:, s:e], gc_k, rng)
            for s, e, gc_k in class_slices
        ]
        return np.concatenate(blocks, axis=1)

    return [_one_sim() for _ in range(N)]


def filter_simulations(
    sims: list,
    gam: np.ndarray,
    pct: float = 0.10,
) -> list:
    """Remove the worst `pct` fraction of simulations by sample-frequency deviation.

    The quality metric for each simulation is the mean absolute difference
    between its column sums (mutations per sample) and those of the observed GAM.
    The top `pct` fraction with the highest deviation are discarded.

    Args:
        sims: list of simulated [n_g x n_t] int8 matrices
        gam : observed [n_g x n_t] int8 GAM
        pct : fraction to remove (default 0.10)

    Returns:
        filtered list of simulated matrices
    """
    obs_col_sums = gam.sum(axis=0).astype(float)
    mad_scores = np.array([
        np.mean(np.abs(S.sum(axis=0).astype(float) - obs_col_sums))
        for S in sims
    ])
    n_remove = max(1, round(pct * len(sims)))
    worst = set(np.argpartition(mad_scores, -n_remove)[-n_remove:].tolist())
    return [S for i, S in enumerate(sims) if i not in worst]
