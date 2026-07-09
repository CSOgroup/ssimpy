import numpy as np


# ---------------------------------------------------------------------------
# Single-class helpers (used directly and by the multi-class function)
# ---------------------------------------------------------------------------

def compute_gene_frequencies(gam: np.ndarray) -> np.ndarray:
    """f(g_i) = (number of mutated samples for gene i) / n_t.

    Returns: np.ndarray [n_g], float
    """
    return gam.sum(axis=1) / gam.shape[1]


def compute_relative_tmb(tmb: np.ndarray) -> np.ndarray:
    """mu(t_j) = TMB(t_j) / sum_t(TMB(t)).

    Returns: np.ndarray [n_t], float
    """
    total = tmb.sum()
    if total == 0:
        raise ValueError("Total TMB is zero — check input data or MAF file.")
    return tmb / total


def compute_E_matrix(gam: np.ndarray, tmb: np.ndarray) -> np.ndarray:
    """Compute the expected mutation probability matrix E.

    E = n_t * f(g) outer-product mu,  capped element-wise at 1.

    Args:
        gam: [n_g x n_t] binary mutation matrix
        tmb: [n_t] raw TMB counts per sample

    Returns:
        E: [n_g x n_t] float, values in (0, 1]
    """
    n_t = gam.shape[1]
    f = compute_gene_frequencies(gam)      # [n_g]
    mu = compute_relative_tmb(tmb)         # [n_t]
    E = n_t * np.outer(f, mu)             # [n_g x n_t]
    np.clip(E, None, 1.0, out=E)
    return E


# ---------------------------------------------------------------------------
# Multi-class E matrix (Step 1-bis)
# ---------------------------------------------------------------------------

def compute_E_matrix_by_class(
    gam: np.ndarray,
    tmb: np.ndarray,
    classes: np.ndarray,
) -> tuple:
    """Compute E by building one block per sample class, then concatenating.

    For each class c_k, gene frequencies f_k(g) and relative TMBs mu_k are
    estimated using only the samples that belong to c_k.  The final matrix is:

        E = [ E_{c_1} | E_{c_2} | ... | E_{c_K} ]   (column-wise concatenation)

    Args:
        gam    : [n_g x n_t] binary mutation matrix (original column order)
        tmb    : [n_t] TMB values (original column order)
        classes: [n_t] class label for each sample

    Returns:
        E         : [n_g x n_t] concatenated expected mutation matrix
        col_order : [n_t] int array — original column indices in class-sorted order;
                    apply gam[:, col_order] to align the GAM with E
        class_slices: list of (col_start, col_end, gene_counts_k) tuples, one per
                    class, used by run_simulations to simulate each block independently
    """
    unique_classes = np.unique(classes)   # sorted alphabetically
    E_blocks      = []
    col_order     = []
    class_slices  = []
    col_cursor    = 0

    for c in unique_classes:
        idx = np.where(classes == c)[0]          # original column indices for class c
        col_order.append(idx)

        gam_k = gam[:, idx]
        tmb_k = tmb[idx]
        E_k   = compute_E_matrix(gam_k, tmb_k)  # [n_g x n_k]

        gene_counts_k = gam_k.sum(axis=1).astype(int)   # [n_g] for rcRAS within block
        n_k = len(idx)
        class_slices.append((col_cursor, col_cursor + n_k, gene_counts_k))

        E_blocks.append(E_k)
        col_cursor += n_k

    col_order = np.concatenate(col_order)
    E = np.concatenate(E_blocks, axis=1)
    return E, col_order, class_slices


# ---------------------------------------------------------------------------
# Multi-type E matrix (element-wise max across mutation types)
# ---------------------------------------------------------------------------

def compute_E_matrix_multitype(gams: list, tmbs: list) -> list:
    """Per-mutation-type E matrices returned as a list.

    For K mutation types, computes E_k for each k. The simulation combines
    them via independent noise draws per type (max of per-type residuals),
    which better reflects that missense and truncating processes act
    independently.

    Args:
        gams: list of K [n_g x n_t] binary mutation matrices
        tmbs: list of K [n_t] TMB arrays

    Returns:
        E_list: list of K [n_g x n_t] expected mutation matrices, one per type
    """
    return [compute_E_matrix(g, t) for g, t in zip(gams, tmbs)]


def compute_E_matrix_multitype_by_class(
    gams: list,
    tmbs: list,
    classes: np.ndarray,
) -> tuple:
    """Per-mutation-type class-stratified E matrices returned as a list.

    Calls compute_E_matrix_by_class for each (gam_k, tmb_k) pair and returns
    the list. col_order and class_slices are taken from the first type
    (identical across types because all types share the same sample set and
    class labels).

    Note: gene_counts_k inside class_slices reflects type 0 only. When the
    combined (OR) GAM is used for simulation the caller must recompute
    gene_counts_k from the reordered combined GAM before passing class_slices
    to run_simulations.

    Args:
        gams   : list of K [n_g x n_t] binary mutation matrices (unsorted)
        tmbs   : list of K [n_t] TMB arrays
        classes: [n_t] sample class labels

    Returns:
        E_list     : list of K [n_g x n_t] expected mutation matrices (class-sorted)
        col_order  : [n_t] original indices in class-sorted order
        class_slices: list of (col_start, col_end, gene_counts_k) from type 0
    """
    col_order = class_slices = None
    E_list = []
    for g, t in zip(gams, tmbs):
        E_k, co, cs = compute_E_matrix_by_class(g, t, classes)
        E_list.append(E_k)
        if col_order is None:
            col_order, class_slices = co, cs
    return E_list, col_order, class_slices
