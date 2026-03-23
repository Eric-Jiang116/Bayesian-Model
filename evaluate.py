"""
Load a checkpoint and produce all evaluation plots.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from data  import VCDataset
from model import VCNet, predict_trajectories


# ──────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────

def load_checkpoint(
    checkpoint_path: str,
    dataset: VCDataset,
    hidden: tuple = (64, 64),
    dropout: float = 0.2,
) -> tuple[VCNet, dict]:
    """
    Restore a VCNet from a checkpoint file.

    Returns
    -------
    model : VCNet  (eval mode)
    ckpt  : raw checkpoint dict
    """
    checkpt  = torch.load(checkpoint_path, map_location="cpu")
    model = VCNet(dataset.P, hidden=hidden, dropout=dropout)
    model.load_state_dict(checkpt["model_state_dict"])
    model.eval()
    return model, checkpt


# ──────────────────────────────────────────────
# Individual plots
# ──────────────────────────────────────────────

def plot_trajectories(
    dage_np: np.ndarray,
    Y_test: torch.Tensor,
    mean_test: torch.Tensor,
    low_test: torch.Tensor,
    high_test: torch.Tensor,
    n_subjects: int = 50,
):
    """Predicted vs true trajectories with MC-dropout uncertainty bands."""
    plt.figure(figsize=(10, 5))

    for s in range(min(n_subjects, Y_test.shape[1])):
        plt.plot(dage_np, Y_test[:, s].cpu(),    "k--", alpha=0.3, label="True"      if s == 0 else "")
        plt.plot(dage_np, mean_test[:, s].cpu(),                    label="Predicted" if s == 0 else "")
        plt.fill_between(
            dage_np,
            low_test[:, s].cpu(),
            high_test[:, s].cpu(),
            alpha=0.1,
        )

    plt.axvline(0.0, color="gray", linestyle=":", alpha=0.7)
    plt.axhline(1.0, color="red",  linestyle=":", label="f(0) = 1")
    plt.title("Predicted Trajectories — Test Set")
    plt.xlabel("Disease Age")
    plt.ylabel("f(v)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_varying_coefficients(
    v_np: np.ndarray,
    beta_pred_np: np.ndarray,
    beta_hat_np: np.ndarray,
    P: int,
):
    """Estimated β̂(v) vs ground-truth β(v) for each covariate."""
    fig, axes = plt.subplots(P, 1, figsize=(8, 3 * P), sharex=True)

    if P == 1:
        axes = [axes]

    for j, ax in enumerate(axes):
        ax.plot(v_np, beta_pred_np[:, j], "k--", lw=2, label="True β")
        ax.plot(v_np, beta_hat_np[:, j],         lw=2, label="Estimated β̂")
        ax.set_ylabel(f"β[{j}]")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("v")
    fig.suptitle("Varying Coefficients: β̂(v) vs β(v)", y=1.002)
    plt.tight_layout()
    plt.show()


def plot_rate_curves(
    v_np: np.ndarray,
    r_true: np.ndarray,
    rates_np: np.ndarray,
    n_subjects: int = 50,
):
    """Predicted rate-vs-value curves against ground truth."""
    plt.figure(figsize=(10, 5))

    for s in range(min(n_subjects, r_true.shape[1])):
        plt.plot(v_np, r_true[:, s],   "k--", alpha=0.2, label="True"      if s == 0 else "")
        plt.plot(v_np, rates_np[:, s],         alpha=0.4, label="Predicted" if s == 0 else "")

    plt.axvline(1.0, color="red", linestyle=":", label="f(0) = 1 anchor")
    plt.xlabel("f_pred(v)")
    plt.ylabel("Rate")
    plt.title("Rate vs Value Curve — Test Subjects")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# Main evaluation pipeline
# ──────────────────────────────────────────────

def evaluate(
    checkpoint_path: str = "model_checkpoint.pt",
    data_dir: str = "data",
    hidden: tuple = (64, 64),
    dropout: float = 0.2,
    n_plot: int = 50,
):
    """
    Full evaluation pipeline:
      1. Load dataset and checkpoint.
      2. Plot predicted trajectories with uncertainty.
      3. Plot estimated vs true varying coefficients.
      4. Plot rate-vs-value curves.
    """
    dataset = VCDataset(data_dir=data_dir)
    model, ckpt = load_checkpoint(checkpoint_path, dataset, hidden=hidden, dropout=dropout)

    dage_np      = dataset.t_grid.numpy()
    v_np         = dataset.v_grid.numpy()
    beta_pred_np = dataset.beta_pred.numpy()

    mean_test = ckpt["mean_test"]
    low_test  = ckpt["low_test"]
    high_test = ckpt["high_test"]
    X_test    = ckpt["X_test"]
    Y_test    = ckpt["Y_test"]
    idx_test  = ckpt["idx_test"]

    # ── 1. Trajectories ───────────────────────────────
    plot_trajectories(dage_np, Y_test, mean_test, low_test, high_test, n_subjects=n_plot)

    # ── 2. Varying coefficients ───────────────────────
    model.eval()
    with torch.no_grad():
        beta_hat = model(dataset.v_grid)                                 # [K, P]
        rates    = torch.exp(beta_hat @ X_test.T)               # [K, S_test]

    beta_hat_np = beta_hat.cpu().numpy()
    plot_varying_coefficients(v_np, beta_pred_np, beta_hat_np, dataset.P)

    # ── 3. Rate curves ────────────────────────────────
    r_true  = dataset.r_pred_np[:, idx_test]   # [K, S_test]
    rates_np = rates.cpu().numpy()
    plot_rate_curves(v_np, r_true, rates_np, n_subjects=n_plot)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    evaluate()