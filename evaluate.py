import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from data import load_data, split_and_normalize, to_tensors
from model import VCNet, rate_fn


# --- MC Dropout (on test set)---

def enable_mc_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_predict(model, v, Xtest, n_samples=1000):
    enable_mc_dropout(model)
    preds = []
    for _ in range(n_samples):
        with torch.no_grad():
            vc_pred   = model(v)
            rate_pred = rate_fn(vc_pred, Xtest)
            preds.append(rate_pred)

    preds = torch.stack(preds, dim=0)          # [n_samples, N, n_subjects]
    scalar_preds = preds.mean(dim=-1)           # [n_samples, N]
    mean_pred = scalar_preds.mean(dim=0)           # [N]
    lower_95 = torch.quantile(scalar_preds, 0.025, dim=0)
    upper_95 = torch.quantile(scalar_preds, 0.975, dim=0)
    return mean_pred, lower_95, upper_95, scalar_preds, preds

######################
#       Plots
######################

# ----- Rate vs Value Curve
def plot_rate_vs_v(v, Xtest, model):
    vmin, vmax = float(v.min()), float(v.max())
    v_grid = np.linspace(vmin, vmax, 50).astype(np.float32).reshape(-1, 1)
    
    with torch.no_grad():
        vc_hat = model(torch.tensor(v_grid))
        rate_pred = rate_fn(vc_hat, Xtest).cpu().numpy()

    plt.figure(figsize=(9, 6))
    for s in range(rate_pred.shape[1]):
        plt.plot(v_grid[:, 0], rate_pred[:, s], alpha=0.2, lw=1)
    plt.xlabel("v")
    plt.ylabel("Predicted Rate")
    plt.title("Rate vs v_pred — All Subjects")
    plt.grid(True, alpha=0.3)
    plt.show()

# ------ True vs Predicted VC Functions ------ 
def plot_vc_comparison(v, v_np, vc, model, P):
    vc_pred_np = vc.to_numpy().astype(np.float32)
    with torch.no_grad():
        vc_hat_np = model(v).cpu().numpy()

    fig, axes = plt.subplots(P, 1, sharex=True)
    for j in range(P):
        axes[j].plot(v_np, vc_pred_np[:, j], "k--", lw=2, label="beta_pred")
        axes[j].plot(v_np, vc_hat_np[:, j],  lw=2,        label="beta_hat")
        axes[j].set_ylabel(f"beta[{j}]")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend()
    axes[-1].set_xlabel("v (v_pred grid)")
    fig.suptitle("Varying coefficients: beta_hat(v) vs beta_pred(v)", y=0.995)
    plt.tight_layout()
    plt.show()

# ------ Confidence Intervals ------
def plot_ci(v, mean_pred, lower, upper):
    order = np.argsort(v[:, 0])
    v_sorted = v[order, 0]
    mean_sorted = mean_pred.cpu().numpy()[order]
    lower_sorted = lower.cpu().numpy()[order]
    upper_sorted = upper.cpu().numpy()[order]

    plt.figure(figsize=(9, 6))
    plt.fill_between(v_sorted, lower_sorted, upper_sorted, color="lightblue", alpha=0.4, label="95% CI")
    plt.plot(v_sorted, mean_sorted, color="blue", label="Mean Rate")
    plt.xlabel("v")
    plt.ylabel("rate(v)")
    plt.title("Mean Subject-Weighted Rate Function with 95% MC Dropout CI")
    plt.legend()
    plt.show()


# --- Main ---

if __name__ == "__main__":
    loss_fn = nn.MSELoss()

    v, vc, Xsub, rate = load_data()
    P = vc.shape[1]

    Xsub_train, Xsub_val, Xsub_test, rate_train, rate_val, rate_test = split_and_normalize(Xsub, rate)
    Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, v_tensor, v_np = to_tensors(
        Xsub_train, Xsub_val, Xsub_test, rate_train, rate_val, rate_test, v
    )

    # Load checkpoint
    checkpoint = torch.load("model_checkpoint.pt")
    model = VCNet(P)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Test MSE
    model.eval()
    with torch.no_grad():
        rate_test_pred = rate_fn(model(v_tensor), Xtest)
        test_mse = loss_fn(rate_test_pred, Ytest).item()
    print(f"Test MSE: {test_mse:.6f}")

    # MC Dropout
    mean_pred, lower, upper, scalar_preds, preds = mc_dropout_predict(model, v_tensor, Xtest)
    for j in range(len(lower)):
        print(f"Test point {j+1}: 95% CI = ({lower[j]:.4f}, {upper[j]:.4f})")
    model.eval()
    # Plots
    plot_rate_vs_v(v_tensor, Xtest, model)
    plot_vc_comparison(v_tensor, v_np, vc, model, P)