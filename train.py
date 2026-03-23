"""
Training loop, MC-dropout inference, and checkpoint saving
"""

import torch
import torch.nn as nn

from data  import VCDataset
from model import VCNet, predict_trajectories


# ──────────────────────────────────────────────
# MC Dropout helpers
# ──────────────────────────────────────────────

def _enable_dropout(module: nn.Module):
    """Switch Dropout layers back to train mode (for MC dropout)."""
    if isinstance(module, nn.Dropout):
        module.train()


def mc_dropout_predict(
    model: VCNet,
    dataset: VCDataset,
    X: torch.Tensor,
    n_samples: int = 50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run MC-dropout inference and return mean + 95 % credible interval.

    Parameters
    ----------
    model     : trained VCNet
    dataset   : VCDataset (provides t_grid, norm_v, onset_index)
    X         : [S, P] subject covariates
    n_samples : number of stochastic forward passes

    Returns
    -------
    mean : [T, S]
    lo   : [T, S]  2.5th percentile
    hi   : [T, S]  97.5th percentile
    """
    model.eval()
    model.apply(_enable_dropout)

    i0 = dataset.onset_index()
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = predict_trajectories(
                model, dataset.t_grid, X, dataset.norm_v, i0
            )
            samples.append(pred)

    stacked = torch.stack(samples, dim=0)           # [n_samples, T, S]
    mean = stacked.mean(dim=0)
    lo = torch.quantile(stacked, 0.025, dim=0)
    hi = torch.quantile(stacked, 0.975, dim=0)
    return mean, lo, hi


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train(
    dataset: VCDataset,
    epochs: int = 2000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden: tuple = (64, 64),
    dropout: float = 0.2,
    log_every: int = 100,
    checkpoint_path: str = "model_checkpoint.pt",
    mc_samples: int = 50,
) -> VCNet:
    """
    Train VCNet and save a checkpoint with test-set MC-dropout predictions.

    Parameters
    ----------
    dataset         : VCDataset
    epochs          : number of training epochs
    lr              : AdamW learning rate
    weight_decay    : AdamW weight decay
    hidden          : hidden layer sizes for VCNet
    dropout         : dropout probability
    log_every       : print interval
    checkpoint_path : where to save the .pt file
    mc_samples      : MC-dropout samples for post-training inference

    Returns
    -------
    model : trained VCNet
    """
    model     = VCNet(dataset.P, hidden=hidden, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn   = nn.MSELoss()
    i0        = dataset.onset_index()

    print(f"Onset index: {i0}  (t ≈ {float(dataset.t_grid[i0]):.3f})")

    for epoch in range(epochs + 1):
        model.train()
        optimizer.zero_grad()

        preds = predict_trajectories(model, dataset.t_grid, dataset.X_train, dataset.norm_v, i0)

        loss = loss_fn(preds, dataset.Y_train)
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0:
            model.eval()
            with torch.no_grad():
                val_preds = predict_trajectories(
                    model, dataset.t_grid, dataset.X_val, dataset.norm_v, i0
                )
                val_loss = loss_fn(val_preds, dataset.Y_val).item()
            print(
                f"Epoch {epoch:4d} | "
                f"Train Loss: {loss.item():.6f} | "
                f"Val Loss:   {val_loss:.6f}"
            )

    # ── post-training MC-dropout on test set ──────────
    mean_test, low_test, high_test = mc_dropout_predict(
        model, dataset, dataset.X_test, n_samples=mc_samples
    )

    # ── save checkpoint ───────────────────────────────
    torch.save(
        {
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "mean_test":            mean_test,
            "low_test":             low_test,
            "high_test":            high_test,
            "idx_test":             dataset.idx_test,
            "X_test":               dataset.X_test,
            "Y_test":               dataset.Y_test,
        },
        checkpoint_path,
    )
    print(f"\nCheckpoint saved to {checkpoint_path}")
    return model


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
dataset = VCDataset(data_dir="data")
train(dataset)