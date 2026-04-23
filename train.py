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
    epochs: int = 1000,
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
    # --- MANUAL WEIGHT INJECTION ---
    with torch.no_grad():
        # 1. Paste the weight matrix you printed from the terminal
        # Make sure the shape matches [P, last_hidden_dim]
        weights = torch.tensor([
            [-3.1895e-01, -1.4369e-01, -5.7668e-03,  6.1796e-02, -5.9135e-02,
            -2.0179e-01, -1.0176e-01, -1.9441e-01,  4.7679e-02, -2.8812e-01,
            8.8882e-02, -1.4775e-01, -1.7861e-01, -1.6748e-01, -7.5941e-02,
            2.5388e-02, -2.1092e-01,  2.7424e-02, -1.7874e-01,  8.0295e-02,
            -1.3224e-01, -1.2410e-01,  2.1736e-02, -1.4876e-01, -1.7377e-01,
            -1.4149e-01, -1.6698e-01, -1.3810e-01,  7.7347e-02,  3.2075e-02,
            -2.4484e-01, -3.5861e-01,  5.3425e-02, -9.8929e-02, -1.2367e-01,
            7.0653e-02,  3.4250e-02, -1.6821e-01, -1.0338e-01,  6.9577e-02,
            -2.0696e-01, -1.8878e-01, -1.5366e-01, -1.0094e-01, -7.1937e-02,
            -1.2719e-01, -1.1608e-01, -1.2266e-01, -1.3827e-01, -9.6695e-02,
            -1.6345e-01, -1.1384e-01,  8.2223e-02,  5.0584e-02, -1.2593e-01,
            -1.1919e-01,  6.4794e-03, -1.6686e-01, -7.3178e-02, -1.5286e-01,
            5.7992e-03,  9.8407e-02,  8.3025e-02, -3.0247e-01],
            [ 1.1318e-01,  6.0539e-03,  5.5673e-02,  1.4664e-01,  1.5957e-01,
            -1.7030e-02,  6.1538e-03, -1.9883e-02, -4.4838e-02, -2.3909e-02,
            1.0400e-01, -4.5314e-03, -3.7610e-03,  9.5555e-03,  1.5577e-01,
            -7.8240e-03, -3.4356e-02,  1.1631e-01, -3.3026e-03,  6.6116e-02,
            -1.7828e-02,  5.6878e-03, -5.9771e-02, -9.2559e-05,  6.9514e-03,
            1.0085e-02, -2.0233e-03, -1.7495e-02,  9.8220e-03, -1.1613e-02,
            -6.4826e-02, -5.7750e-02,  7.2965e-02, -3.4043e-02,  3.6670e-03,
            -1.7621e-03,  8.0057e-02, -2.4649e-03, -1.1527e-02, -4.2255e-02,
            -5.9830e-03, -1.4876e-03, -2.3875e-02,  3.8190e-03,  5.2224e-03,
            3.0186e-03,  8.9632e-03,  3.1768e-03,  3.9346e-03, -1.3854e-02,
            5.7486e-03,  2.0239e-03,  2.4500e-01,  7.0959e-02,  4.9709e-03,
            -1.3360e-03,  3.1749e-02, -1.1377e-02,  9.0194e-03, -9.4468e-03,
            -1.1503e-01,  4.1638e-02,  7.5600e-02, -7.9415e-02],
            [-1.0427e-03, -8.8565e-05, -9.2044e-02, -5.2378e-02, -9.2669e-04,
            -1.1035e-04, -8.8352e-05,  5.7004e-05, -6.3226e-02, -1.2938e-04,
            7.8342e-02, -1.2409e-04, -1.6472e-04, -1.5988e-04, -8.0646e-04,
            1.5989e-02,  4.9579e-05, -2.7253e-02, -7.4288e-05,  2.2600e-02,
            3.2560e-05, -1.4039e-04, -1.0370e-01, -1.4587e-04, -8.6385e-05,
            -1.8324e-04, -2.0769e-04,  1.6128e-04,  5.6537e-02,  1.0942e-01,
            2.0327e-04, -4.2181e-05, -2.9896e-02,  5.1933e-05, -1.1210e-04,
            9.7044e-02,  4.1583e-02, -5.3145e-05, -8.6016e-05,  4.6132e-02,
            -1.6626e-04, -1.4348e-04,  2.2407e-04, -1.3977e-04, -4.2621e-05,
            -5.9757e-05, -9.8406e-05, -1.4058e-04, -1.4798e-04,  4.3382e-06,
            -1.7321e-04, -4.5860e-06, -1.5055e-03, -2.9033e-02, -9.3187e-05,
            -5.8609e-05, -7.9599e-02, -9.4863e-05, -4.8742e-04, -1.0132e-04,
            -8.6419e-03, -4.1114e-02,  1.0225e-01,  1.5451e-04],
            [ 4.1923e-02,  8.6126e-04,  3.6127e-03, -5.1993e-02,  6.3191e-02,
            -9.8144e-03,  1.7942e-03, -1.1186e-02, -3.9919e-02, -1.3826e-02,
            -2.4610e-02, -4.1257e-03, -3.2495e-03,  2.5081e-03,  6.0143e-02,
            4.4630e-02, -1.8276e-02,  5.5749e-02, -3.7190e-03,  5.2462e-02,
            -9.1442e-03,  1.3642e-04, -3.2067e-02, -2.0169e-03, -2.7515e-05,
            3.3776e-03, -2.4121e-03, -7.8879e-03, -3.3600e-02,  8.4797e-02,
            -2.8249e-02, -2.8465e-02, -4.5846e-02, -1.7382e-02,  2.1199e-04,
            9.7558e-03,  1.2443e-01, -2.7846e-03, -5.9008e-03,  8.0260e-02,
            -4.7454e-03, -3.5285e-03, -1.2463e-02,  5.1851e-04,  3.0608e-03,
            1.9370e-04,  2.5944e-03, -1.1464e-03,  2.0305e-04, -5.8686e-03,
            6.0837e-04, -5.8964e-04,  1.0633e-01,  1.8544e-02,  7.9087e-04,
            -2.2181e-03,  5.9624e-02, -6.8572e-03,  7.6346e-04, -6.5392e-03,
            -8.3054e-02, -3.3207e-02, -2.1817e-02, -3.6696e-02]
        ])
        
        # Get bias
        bias = torch.tensor([-0.1990,  0.4999,  0.0035,  0.3121])

        model.net[-1].weight.copy_(weights)
        model.net[-1].bias.copy_(bias)

        print("Manual initialization complete.")

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