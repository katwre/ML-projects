# -------------------------
# Vizualization
# -------------------------

"""
Reusable visualization utilities for TinyDeconvTransformer experiments.
Drop this file into your project (e.g., import as `import transformer_viz_utils as viz`).

All plots use matplotlib only (no seaborn) and avoid specifying colors explicitly.
Each function returns the matplotlib Figure (and sometimes Axes) so you can save
or further customize them.
"""
from __future__ import annotations
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# Helper
# -------------------------------
def _np(x):
    """Detach/convert tensors to numpy if needed; pass through numpy arrays."""
    try:
        import torch  # local import to avoid hard dep if not used
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def get_W_numpy(model) -> np.ndarray:
    """Convenience: extract `model.W` as a NumPy array."""
    import torch  # type: ignore
    W = getattr(model, "W", None)
    if W is None:
        raise AttributeError("model has no attribute `W`")
    if isinstance(W, torch.Tensor):
        return W.detach().cpu().numpy()
    return np.asarray(W)


# -------------------------------
# 1) Learning curves
# -------------------------------
def plot_learning_curves(
    train_history: Sequence[float],
    val_history: Optional[Sequence[float]] = None,
    r_history: Optional[Sequence[float]] = None,
    mae_history: Optional[Sequence[float]] = None,
) -> Tuple[plt.Figure, list[plt.Figure]]:
    """Plot learning curves across epochs.

    Returns
    -------
    (loss_fig, other_figs)
        loss_fig: Figure for Train/Val loss
        other_figs: Figures for r and MAE (if provided)
    """
    train_history = _np(train_history)
    val_history = _np(val_history) if val_history is not None else None

    # Loss curves
    loss_fig = plt.figure(figsize=(6, 4))
    plt.plot(train_history, label="Train Loss")
    if val_history is not None:
        plt.plot(val_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curves")
    plt.legend()
    plt.tight_layout()

    other_figs: list[plt.Figure] = []

    # r (validation)
    if r_history is not None:
        r_fig = plt.figure(figsize=(6, 4))
        plt.plot(_np(r_history), label="Mean Pearson r (Validation)")
        plt.xlabel("Epoch")
        plt.ylabel("r")
        plt.title("Validation Correlation")
        plt.legend()
        plt.tight_layout()
        other_figs.append(r_fig)

    # MAE (validation)
    if mae_history is not None:
        mae_fig = plt.figure(figsize=(6, 4))
        plt.plot(_np(mae_history), label="Mean MAE (Validation)")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title("Validation MAE")
        plt.legend()
        plt.tight_layout()
        other_figs.append(mae_fig)

    return loss_fig, other_figs


# -------------------------------
# 2) Predicted vs True proportions (scatter)
# -------------------------------
def plot_pred_vs_true(
    H_true: np.ndarray,
    H_hat: np.ndarray,
    lim: Tuple[float, float] = (0.0, 1.0),
    suptitle: str = "Predicted vs True Cell-Type Proportions (Validation)",
) -> plt.Figure:
    """Scatter plots of predicted vs true proportions per cell type.

    Parameters
    ----------
    H_true, H_hat : arrays shaped [N, K]
    lim : (min, max) for both axes
    """
    H_true = _np(H_true)
    H_hat = _np(H_hat)
    assert H_true.shape == H_hat.shape, "H_true and H_hat must have same shape [N, K]"

    K = H_hat.shape[1]
    fig, axes = plt.subplots(1, K, figsize=(3.2 * K, 3.2), sharex=True, sharey=True)
    if K == 1:
        axes = [axes]  # type: ignore

    minv, maxv = lim
    for k in range(K):
        ax = axes[k]
        ax.scatter(H_true[:, k], H_hat[:, k], s=12, alpha=0.7)
        ax.plot([minv, maxv], [minv, maxv], linestyle="--")
        ax.set_title(f"Cell {k}")
        ax.set_xlabel("True")
        if k == 0:
            ax.set_ylabel("Predicted")
        ax.set_xlim(minv, maxv)
        ax.set_ylim(minv, maxv)

    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


# -------------------------------
# 3) Stacked bars for first S samples
# -------------------------------

def plot_stacked_bars(
    H_true: np.ndarray,
    H_hat: np.ndarray,
    S: int = 20,
    suptitles: Tuple[str, str] = ("True Proportions (Stacked)", "Predicted Proportions (Stacked)"),
) -> plt.Figure:
    """Stacked bar charts for first S validation samples."""
    H_true = _np(H_true)
    H_hat = _np(H_hat)
    S = int(min(S, H_true.shape[0], H_hat.shape[0]))
    K = H_true.shape[1]

    idx = np.arange(S)
    true = H_true[:S]
    pred = H_hat[:S]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].set_title(suptitles[0])
    axes[1].set_title(suptitles[1])

    bottom = np.zeros(S)
    for k in range(K):
        axes[0].bar(idx, true[:, k], bottom=bottom, width=0.9)
        bottom += true[:, k]

    bottom = np.zeros(S)
    for k in range(K):
        axes[1].bar(idx, pred[:, k], bottom=bottom, width=0.9)
        bottom += pred[:, k]

    axes[1].set_xlabel("Validation Sample")
    for ax in axes:
        ax.set_ylabel("Proportion")
        ax.set_ylim(0, 1)

    fig.tight_layout()
    return fig


# -------------------------------
# 4) Compare learned W to reference R
# -------------------------------

def plot_W_vs_R(
    R_ref: np.ndarray,
    W_hat: Optional[np.ndarray] = None,
    model: Optional[object] = None,
    transpose: bool = True,
    titles: Tuple[str, str] = ("Reference R (Cells × Regions)", "Learned W (Cells × Regions)"),
) -> plt.Figure:
    """Heatmaps comparing reference R and learned W.

    Parameters
    ----------
    R_ref : array shaped [P, K] or [K, P] (we'll try to infer orientation)
    W_hat : array shaped like R_ref; if None, will try to read from model.W
    model : model containing attribute `W` (used if W_hat is None)
    transpose : if True, display as Cells × Regions
    """
    R_ref = _np(R_ref)

    if W_hat is None:
        if model is None:
            raise ValueError("Provide either W_hat or model with attribute `W`.")
        W_hat = get_W_numpy(model)
    W_hat = _np(W_hat)

    # Try to align shapes as [Cells, Regions] for display
    def _to_cells_regions(A: np.ndarray) -> np.ndarray:
        if A.shape[0] <= A.shape[1]:
            # likely [P, K] -> transpose to [K, P]
            A = A.T
        return A

    R_disp = _to_cells_regions(R_ref)
    W_disp = _to_cells_regions(W_hat)

    fig = plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(R_disp, aspect="auto", vmin=0, vmax=1)
    plt.title(titles[0])
    plt.xlabel("Region")
    plt.ylabel("Cell")

    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(W_disp, aspect="auto", vmin=0, vmax=1)
    plt.title(titles[1])
    plt.xlabel("Region")

    fig.tight_layout()
    return fig


# -------------------------------
# 5) One-shot dashboard (optional)
# -------------------------------

def plot_all_diagnostics(
    train_history: Sequence[float],
    val_history: Optional[Sequence[float]],
    r_history: Optional[Sequence[float]],
    mae_history: Optional[Sequence[float]],
    H_true: np.ndarray,
    H_hat: np.ndarray,
    R_ref: np.ndarray,
    W_hat: Optional[np.ndarray] = None,
    model: Optional[object] = None,
) -> None:
    """Convenience wrapper to reproduce all standard plots in one call."""
    plot_learning_curves(train_history, val_history, r_history, mae_history)
    plot_pred_vs_true(H_true, H_hat)
    plot_stacked_bars(H_true, H_hat)
    plot_W_vs_R(R_ref, W_hat=W_hat, model=model)
    plt.show()
