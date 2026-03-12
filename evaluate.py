import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint

# ---------------- DATA ----------------
X_sub_df     = pd.read_csv("data/Xsub.csv")
f_df         = pd.read_csv("data/f_pred.csv")
dage_df      = pd.read_csv("data/dage.csv")
v_df         = pd.read_csv("data/v_pred.csv")
beta_pred_df = pd.read_csv("data/beta_pred.csv")
r_pred_df    = pd.read_csv("data/r_pred.csv")

P          = X_sub_df.shape[1]
v_np       = v_df.to_numpy().reshape(-1).astype(np.float32)
beta_pred_np = beta_pred_df.to_numpy().astype(np.float32)
r_pred_np  = r_pred_df.to_numpy().astype(np.float32)
dage_np    = dage_df.to_numpy().reshape(-1).astype(np.float32)
f_np       = f_df.to_numpy().astype(np.float32)
X_all      = torch.tensor(X_sub_df.to_numpy().astype(np.float32))
v_grid     = torch.tensor(v_np, dtype=torch.float32)
t_grid     = torch.tensor(dage_np, dtype=torch.float32)
i0         = int(torch.argmin(torch.abs(t_grid - 0.0)).item())
beta = torch.tensor(beta_pred_np, dtype=torch.float32)
# ---------------- MODEL DEFINITION (must match training) ----------------
class VCNet(nn.Module):
    def __init__(self, P, hidden=(64, 64), dropout=0.2):
        super().__init__()
        layers = []
        in_dim = 1
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        last_layer = nn.Linear(in_dim, P, bias=True)
        nn.init.zeros_(last_layer.weight) # start with zero output (prevent exploding numbers)
        nn.init.zeros_(last_layer.bias)
        layers += [last_layer]
        self.net = nn.Sequential(*layers)

    def forward(self, t_norm):  # [batch,1]
        return self.net(t_norm) # [batch,P]

rate = torch.exp(beta @ X_all.T)
print(rate)
print(r_pred_np)
# ---------------- LOAD CHECKPOINT ----------------
model = VCNet(P)
checkpoint = torch.load("model_checkpoint.pt", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])

v_mean = checkpoint["v_mean"]
v_std  = checkpoint["v_std"]
mean_test = checkpoint["mean_test"]
low_test  = checkpoint["low_test"]
high_test  = checkpoint["high_test"]
idx_test = checkpoint["idx_test"]
X_test = checkpoint["X_test"]
Y_test = checkpoint["Y_test"]

def norm_v(v):
    return (v - v_mean) / v_std

# ---------------- RUN EVALUATION / PLOTS ----------------
plt.figure(figsize=(10, 5))
for s in range(50):
    plt.plot(dage_np, Y_test[:, s].cpu(), 'k--', alpha=0.3, label="True" if s==0 else "")
    plt.plot(dage_np, mean_test[:, s].cpu())
    plt.fill_between(dage_np, low_test[:, s].cpu(), high_test[:, s].cpu(), alpha=0.1)

plt.axvline(0.0, color='gray', linestyle=':', alpha=0.7)
plt.axhline(1.0, color='red', linestyle=':', label="f(0)=1")
plt.title("Subject-Based Split: Predicted Trajectories (Test Set)")
plt.xlabel("Disease Age")
plt.ylabel("f(v)")
plt.legend()
plt.show()

model.eval()
with torch.no_grad():
    v_norm = norm_v(v_grid).view(-1, 1)     # [K,1] in the SAME normalization as training
    beta_hat = model(v_norm)    # [K,P]
    rates = torch.exp(beta_hat @ X_test.T)
    
beta_hat_np = beta_hat.cpu().numpy()

# Predicted vc vs true vc 
fig, axes = plt.subplots(P, 1, sharex=True)
for j in range(P):
    axes[j].plot(v_np, beta_pred_np[:, j], "k--", lw=2, label="beta_pred")
    axes[j].plot(v_np, beta_hat_np[:, j], lw=2, label="beta_hat (mapped)")
    axes[j].set_ylabel(f"beta[{j}]")
    axes[j].grid(True, alpha=0.3)
    axes[j].legend()
axes[-1].set_xlabel("v (v_pred grid)")
fig.suptitle("Varying coefficients: beta_hat(v) vs beta_pred(v)", y=0.995)
plt.tight_layout()
plt.show()

# Rate vs value curve
r_true = r_pred_np[:, idx_test]
rates_np = rates.cpu().numpy()

plt.figure(figsize=(10, 5))
for s in range(min(50, X_test.shape[0])):
    plt.plot(v_np, r_true[:, s], 'k--', alpha=0.2, label="True" if s == 0 else "")
    plt.plot(v_np, rates_np[:, s], alpha=0.4, label="Predicted" if s == 0 else "")

plt.axvline(1.0, color='red', linestyle=':', label="f(0)=1 anchor")
plt.xlabel("fpred(v)")
plt.ylabel("Rate")
plt.title("Rate vs Value Curve — test subjects")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()