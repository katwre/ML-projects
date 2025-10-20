import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment, nnls
from typing import Dict, Tuple, Optional

def nmf_deconvolve_and_evaluate(
    ref_aln: pd.DataFrame,
    mix_aln: pd.DataFrame,
    truth: pd.DataFrame,
    n_components: Optional[int] = None,
    init: str = "nndsvda",
    max_iter: int = 2000,
    random_state: int = 0,
    return_factor_matrices: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, pd.Series], Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    Run an NMF baseline for deconvolution and evaluate against ground truth.

    Parameters
    ----------
    ref_aln : pd.DataFrame
        CpGs × celltypes reference matrix (columns = cell types).
    mix_aln : pd.DataFrame
        CpGs × samples mixture matrix (columns = samples).
    truth : pd.DataFrame
        celltypes × samples matrix of true proportions. Row index must be cell types; columns are samples.
    n_components : int, optional
        Number of latent components. Defaults to len(ref_aln.columns).
    init : str
        NMF init method. For non-negative data "nndsvda" is a good default.
    max_iter : int
        Max iterations for NMF.
    random_state : int
        Random seed.
    return_factor_matrices : bool
        If True, also return (Wm, Hm) from NMF.

    Returns
    -------
    P_hat_df : pd.DataFrame
        Estimated proportions (celltypes × samples) aligned to truth's row (celltype) order and mix's columns (samples).
    metrics_scalar : dict
        Scalar metrics, currently {"mae_overall": float}.
    metrics_vectors : dict
        Vector metrics, currently:
            - "mae_per_sample": pd.Series indexed by sample.
            - "corr_alignment": pd.Series correlation of aligned rows (by celltype).
    factors : tuple or None
        (Wm, Hm) if return_factor_matrices is True, else None.
    """
    # Ensure consistent ordering
    celltypes = list(ref_aln.columns)
    samples = list(mix_aln.columns)

    # If truth has extra/missing labels, align and check
    missing_ct = [ct for ct in celltypes if ct not in truth.index]
    missing_samples = [s for s in samples if s not in truth.columns]
    if missing_ct:
        raise ValueError(f"Truth is missing cell types: {missing_ct}")
    if missing_samples:
        raise ValueError(f"Truth is missing samples: {missing_samples}")

    T_true = truth.loc[celltypes, samples].values.astype(float)  # (k × n)

    # Data for factorization
    Y = mix_aln.values.astype(float)   # (p × n)
    k = n_components if n_components is not None else len(celltypes)

    # --- NMF: Y ≈ Wm (p×k) @ Hm (k×n) ---
    nmf = NMF(n_components=k, init=init, max_iter=max_iter, random_state=random_state)
    Wm = nmf.fit_transform(Y)          # (p × k)
    Hm = nmf.components_               # (k × n)

    # Normalize columns of Hm to sum to 1 -> proportions per sample
    P_hat = Hm / (Hm.sum(axis=0, keepdims=True) + 1e-12)  # (k × n)

    # --- Align components to truth via Hungarian assignment on correlation ---
    # Build k×k correlation matrix between rows of P_hat and rows of T_true
    def _row_center(X):
        return X - X.mean(axis=1, keepdims=True)

    A = _row_center(P_hat)
    B = _row_center(T_true if T_true.shape[0] == k else T_true[:k])  # safety if k != len(celltypes)

    An = np.sqrt((A**2).sum(axis=1, keepdims=True)) + 1e-12
    Bn = np.sqrt((B**2).sum(axis=1, keepdims=True)) + 1e-12
    C = (A @ B.T) / (An @ Bn.T)   # (k × k) correlation matrix

    row_ind, col_ind = linear_sum_assignment(-C)  # maximize correlation

    # Reorder P_hat rows to match truth row order using the assignment
    # row_ind[i] in P_hat corresponds to col_ind[i] in truth
    # Create inverse permutation so final order is truth's order [0..k-1]
    inv_perm = np.argsort(col_ind)
    P_hat_aligned = P_hat[row_ind][inv_perm]      # (k × n)

    # Correlations of each aligned row with its matched truth row
    corr_alignment = pd.Series(
        C[row_ind, col_ind][inv_perm],
        index=celltypes[:k],
        name="corr(P_hat_row, truth_row)"
    )

    # Package to DataFrame aligned with truth's order and sample names
    P_hat_df = pd.DataFrame(P_hat_aligned, index=celltypes[:k], columns=samples)

    # --- Metrics (MAE) ---
    # If k < len(celltypes), restrict truth to first k to match
    T_for_mae = truth.loc[celltypes[:k], samples]
    diff = (P_hat_df - T_for_mae).abs()
    mae_overall = float(diff.values.mean())
    mae_per_sample = diff.mean(axis=0)  # Series over samples

    metrics_scalar = {"mae_overall": mae_overall}
    metrics_vectors = {
        "mae_per_sample": mae_per_sample,
        "corr_alignment": corr_alignment,
    }

    factors = (Wm, Hm) if return_factor_matrices else None
    return P_hat_df, metrics_scalar, metrics_vectors, factors




def semi_supervised_nmf_anchor(ref_aln, mix_aln, truth,
                               lam=1.0,
                               max_iter=200,
                               tol=1e-5,
                               sum_to_one=True,
                               verbose=True,
                               seed=0):
    """
    Semi-supervised NMF with anchoring of W toward R:
        minimize ||Y - W H||_F^2 + lam * ||W - R||_F^2
    with W >= 0, H >= 0.
    
    Inputs
    ------
    ref_aln : DataFrame (CpGs × celltypes)
    mix_aln : DataFrame (CpGs × samples)
    truth   : DataFrame (celltypes × samples)  [only used for evaluation]
    
    Returns
    -------
    W_df : DataFrame (CpGs × celltypes)  learned basis, anchored to ref
    H_df : DataFrame (celltypes × samples) nonnegative activations
    P_df : DataFrame (celltypes × samples) proportions (columns sum to 1 if sum_to_one=True)
    hist : dict of losses across iterations
    """

    rng = np.random.default_rng(seed)

    # matrices
    celltypes = list(ref_aln.columns)
    samples   = list(mix_aln.columns)

    R = ref_aln.loc[:, celltypes].values.astype(float)  # (p × k)
    Y = mix_aln.loc[:, samples].values.astype(float)    # (p × n)
    k = R.shape[1]
    p, n = Y.shape

    # --- initialization ---
    # Start W near R; small nonneg noise
    W = np.clip(R + rng.normal(0, 0.01, size=R.shape), 0, None)

    # Initialize H by NNLS per sample using initial W
    # Start W near the reference R (semi-supervised “warm start”), then clip negatives to 0
    H = np.zeros((k, n), dtype=float)
    for j in range(n):
        h_j, _ = nnls(W, Y[:, j])
        H[:, j] = h_j
    if sum_to_one:
        colsum = H.sum(axis=0, keepdims=True) + 1e-12
        H = H / colsum

    hist = {"obj": []}

    # Pre-allocate for speed
    I_k = np.eye(k)

    # --- alternating minimization ---
    for it in range(1, max_iter + 1):

        # Update W (anchored ridge step) and project to nonneg
        # W = (Y H^T + lam R) (H H^T + lam I)^(-1)
        # W-update (anchored ridge step)
        HHt = H @ H.T
        try:
            W_new = (Y @ H.T + lam * R) @ np.linalg.inv(HHt + lam * I_k)
        except np.linalg.LinAlgError:
            # add tiny jitter if ill-conditioned
            W_new = (Y @ H.T + lam * R) @ np.linalg.pinv(HHt + lam * I_k + 1e-8 * I_k)

        W_new = np.clip(W_new, 0, None)

        # Update H by NNLS per sample (using updated W)
        H_new = np.zeros_like(H)
        for j in range(n):
            h_j, _ = nnls(W_new, Y[:, j])
            H_new[:, j] = h_j

        if sum_to_one:
            colsum = H_new.sum(axis=0, keepdims=True) + 1e-12
            H_new = H_new / colsum

        # Compute objective
        # Compute objective & check convergence
        recon = np.linalg.norm(Y - W_new @ H_new, ord='fro')**2
        anchor = np.linalg.norm(W_new - R, ord='fro')**2
        obj = recon + lam * anchor
        hist["obj"].append(obj)

        # Check convergence (relative change) and record the total loss and its parts
        if it > 1:
            rel = abs(hist["obj"][-2] - obj) / (hist["obj"][-2] + 1e-12)
            if verbose and (it % 10 == 0 or it == 1):
                print(f"[iter {it:03d}] obj={obj:.6e}  recon={recon:.6e}  anchor={anchor:.6e}  Δrel={rel:.3e}")
            if rel < tol:
                if verbose:
                    print(f"Converged at iter {it} (Δrel={rel:.3e})")
                W, H = W_new, H_new
                break
        else:
            if verbose:
                print(f"[iter {it:03d}] obj={obj:.6e}  recon={recon:.6e}  anchor={anchor:.6e}")

        W, H = W_new, H_new

    # Wrap outputs
    W_df = pd.DataFrame(W, index=ref_aln.index, columns=celltypes)
    H_df = pd.DataFrame(H, index=celltypes, columns=samples)

    # proportions (same as H if we normalized columns)
    P_df = H_df.copy()

    # Quick evaluation if truth provided on same axes
    if truth is not None:
        truth_aln = truth.loc[celltypes, samples]
        mae_overall = (P_df - truth_aln).abs().values.mean()
        mae_per_sample = (P_df - truth_aln).abs().mean(axis=0)
        print(f"\nSemiNMF (anchored) overall MAE: {mae_overall:.4f}")
        print(mae_per_sample.head())

    return W_df, H_df, P_df, hist
