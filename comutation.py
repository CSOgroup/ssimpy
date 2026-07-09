import numpy as np


def compute_penalty_vector(
    tmb: np.ndarray,
    tau: float = 1.0,
    lam: float = 0.3,
    classes: np.ndarray = None,
) -> np.ndarray:
    """Compute the TMB-based sample penalty vector p.

    High-TMB samples are down-weighted so that co-mutations in low-TMB
    samples contribute more to the signal.

        TMB_FC = TMB / ref_TMB
        p = 1 / (1 + max(lambda * (TMB_FC - tau), 0))

    The reference TMB depends on whether class covariates are provided:
        - No classes : ref_TMB = median(TMB)  over all samples
        - With classes: ref_TMB = mean of per-class medians

    Args:
        tmb    : [n_t] raw mutation counts per sample (class-sorted if classes given)
        tau    : fold-change threshold above which penalization starts (default 1.0)
        lam    : rate of penalization (default 0.3)
        classes: [n_t] class labels aligned to tmb, or None

    Returns:
        p: [n_t] float penalty weights in (0, 1]
    """
    if classes is None:
        ref_tmb = np.median(tmb)
    else:
        class_medians = [np.median(tmb[classes == c]) for c in np.unique(classes)]
        ref_tmb = float(np.mean(class_medians))

    if ref_tmb == 0:
        raise ValueError("Reference TMB is zero — check TMB values.")
    tmb_fc = tmb / ref_tmb
    return 1.0 / (1.0 + np.maximum(lam * (tmb_fc - tau), 0.0))


def compute_weighted_comutation(
    gam: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    """Compute the weighted co-mutation incidence matrix.

        wCO = (p * GAM) x GAM^T

    where p is applied column-wise (each sample's mutations are scaled by p[j]).

    Args:
        gam: [n_g x n_t] binary mutation matrix (int8 or float)
        p  : [n_t] penalty weight vector

    Returns:
        wCO: [n_g x n_g] float weighted co-mutation matrix
    """
    weighted = gam * p                  # broadcast: [n_g x n_t] * [n_t] → [n_g x n_t]
    return np.dot(weighted, gam.T)     # [n_g x n_g]; np.dot avoids spurious BLAS FP warnings with mixed int8/float64
