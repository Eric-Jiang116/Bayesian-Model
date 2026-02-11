"""
  - f_pred rows correspond to the same "v-grid" as dage (length 250). If dage is disease-age,
    we treat it as the integration grid variable (call it v_grid) and use it in odeint.
  - X_sub has shape [n_subjects, P] = [1000, 4].
  - f_pred has shape [n_grid, n_subjects] = [250, 1000].
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------- LOAD DATA ----------------
v_50   = pd.read_csv("data/v_pred.csv")        # (50,1) or (50,)
vc_50  = pd.read_csv("data/beta_pred.csv")     # (50,P) optional (not used for training unless you want)
X_sub_df = pd.read_csv("data/Xsub.csv")       # (n_subjects,P)
f_df   = pd.read_csv("data/f_pred.csv")        # (n_grid,n_subjects) = (250,1000)
dage_df = pd.read_csv("data/dage.csv")         # (n_grid,1) or (n_grid,)

# ---------------- BASIC SHAPES ----------------
P = X_sub_df.shape[1]
n_subjects = X_sub_df.shape[0]

# Ensure dage is a flat (n_grid,) vector
dage_np = dage_df.to_numpy().reshape(-1).astype(np.float32)
n_grid = len(dage_np)

# Convert f to numpy float32: shape [n_grid, n_subjects]
f_np = f_df.to_numpy().astype(np.float32)
assert f_np.shape[0] == n_grid, f"f_pred has {f_np.shape[0]} rows but dage has {n_grid} points"
assert f_np.shape[1] == n_subjects, f"f_pred has {f_np.shape[1]} subjects but X_sub has {n_subjects}"

# Convert X_sub: shape [n_subjects, P]
X_sub = torch.tensor(X_sub_df.to_numpy().astype(np.float32))  # [S,P]

# ---------------- TRAIN/VAL/TEST SPLIT (ALIGNED) ----------------
# Split indices along the grid axis (rows of f, entries of dage)
idx = np.arange(n_grid)
idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=SEED, shuffle=True)
idx_train, idx_val  = train_test_split(idx_train, test_size=0.25, random_state=SEED, shuffle=True)  # 0.25 of 0.8 -> 0.2

# Create torch grids and targets for each split
# IMPORTANT: odeint expects t sorted increasing (best practice). We'll sort within each split.
def make_split(split_idx):
    split_idx = np.sort(split_idx)
    t_split = torch.tensor(dage_np[split_idx], dtype=torch.float32)                 # [T_split]
    y_split = torch.tensor(f_np[split_idx, :], dtype=torch.float32)                # [T_split, S]
    return t_split, y_split, split_idx

t_train, Ytrain, idx_train_sorted = make_split(idx_train)
t_val,   Yval,   idx_val_sorted   = make_split(idx_val)
t_test,  Ytest,  idx_test_sorted  = make_split(idx_test)

# ---------------- NORMALIZE INPUT GRID (TRAIN ONLY) ----------------
# We normalize the independent variable (dage) using train split only
v_mean = float(t_train.mean().item())
v_std  = float(t_train.std().item() + 1e-8)

def norm_t(t):
    return (t - v_mean) / v_std

# ---------------- MODEL ----------------
class VCNet(nn.Module):
    def __init__(self, P, hidden=(64,64), dropout=0.2):
        super().__init__()
        layers = []
        in_dim = 1
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, P)]
        self.net = nn.Sequential(*layers)

    def forward(self, t_norm):  # t_norm: [batch, 1]
        return self.net(t_norm) # [batch, P]

model = VCNet(P)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

# Initial condition: f(v0) = 0 for all subjects
y0 = torch.zeros(n_subjects, dtype=torch.float32)  # [S]

# ---------------- ODE RHS ----------------
# y is current state f(v) but RHS does not depend on y here.
def rhs(t, y):
    """
    t: scalar tensor
    y: [S]
    returns: dy/dt = rate(t) = exp( beta(t)^T X_sub )
    """
    t_in = norm_t(t).view(1, 1)              # [1,1]
    beta = model(t_in)                       # [1,P]
    # rate per subject: [1,P] @ [P,S] -> [1,S] -> [S]
    rate = torch.exp(beta @ X_sub.T).squeeze(0)
    return rate

# Helper to integrate on a given grid
def integrate_on_grid(t_grid, step_size=0.25, method="rk4"):
    # odeint returns [T, S]
    return odeint(rhs, y0, t_grid, method=method, options={"step_size": step_size})

# ---------------- TRAIN LOOP ----------------
EPOCHS = 2000
step_size = 0.25

for epoch in range(EPOCHS):
    model.train()
    opt.zero_grad()

    pred_train = integrate_on_grid(t_train, step_size=step_size, method="rk4")  # [T_train, S]
    loss = loss_fn(pred_train, Ytrain)
    loss.backward()
    opt.step()

    if epoch % 200 == 0:
        model.eval()
        with torch.no_grad():
            pred_val = integrate_on_grid(t_val, step_size=step_size, method="rk4")
            val_loss = loss_fn(pred_val, Yval).item()
        print(f"Epoch {epoch:4d} | train {loss.item():.6f} | val {val_loss:.6f}")

# ---------------- TEST EVAL ----------------
model.eval()
with torch.no_grad():
    pred_test = integrate_on_grid(t_test, step_size=step_size, method="rk4")
    test_mse = loss_fn(pred_test, Ytest).item()
print(f"Test MSE: {test_mse:.6f}")

# ---------------- MC DROPOUT ----------------
# Keep dropout ON at test time by setting Dropout modules to train mode only
def enable_dropout(m):
    if isinstance(m, nn.Dropout):
        m.train()

def mc_dropout_predict(t_grid, n_samples=200):
    """
    Returns:
      mean_pred: [T,S]
      lower_95 : [T,S]
      upper_95 : [T,S]
      samples  : [n_samples,T,S]
    """
    model.eval()
    model.apply(enable_dropout)

    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            samples.append(integrate_on_grid(t_grid, step_size=step_size, method="rk4"))
    samples = torch.stack(samples, dim=0)  # [K,T,S]

    mean_pred = samples.mean(dim=0)  # [T,S]
    lower_95 = torch.quantile(samples, 0.025, dim=0)
    upper_95 = torch.quantile(samples, 0.975, dim=0)
    return mean_pred, lower_95, upper_95, samples

mean_test, lower_test, upper_test, samples_test = mc_dropout_predict(t_test, n_samples=200)
print("MC samples shape:", samples_test.shape)  # [K, T_test, S]

# Example: print CI for a single subject at each timepoint (subject 0)
s = 0
for i in range(len(t_test)):
    print(f"t={t_test[i].item():.3f}: 95% CI (subject {s}) = ({lower_test[i,s].item():.4f}, {upper_test[i,s].item():.4f})")

# ---------------- PLOT: trajectories for a few subjects ----------------
# Plot on the FULL grid for interpretability
t_full = torch.tensor(dage_np, dtype=torch.float32)

model.eval()
with torch.no_grad():
    pred_full = integrate_on_grid(t_full, step_size=step_size, method="rk4")  # [n_grid,S]

pred_full_np = pred_full.cpu().numpy()

plt.figure(figsize=(9,6))
for s in range(min(100, n_subjects)):  # plot first 10 subjects to avoid clutter
    plt.plot(dage_np, pred_full_np[:, s], alpha=0.8, lw=1)
plt.xlabel("Disease Age (grid)")
plt.ylabel("Integrated rate f(v)")
plt.title("Predicted integrated trajectories (first 100 subjects)")
plt.grid(True, alpha=0.3)
plt.show()

# ---------------- PLOT: rate function r(v) for a few subjects ----------------
# Rate at each grid point is exp(beta(v)^T x_s)
model.eval()
with torch.no_grad():
    t_in = norm_t(t_full).view(-1,1)     # [n_grid,1]
    beta_full = model(t_in)              # [n_grid,P]
    rate_full = torch.exp(beta_full @ X_sub.T)  # [n_grid,S]

rate_full_np = rate_full.cpu().numpy()

plt.figure(figsize=(9,6))
for s in range(min(30, n_subjects)):
    plt.plot(dage_np, rate_full_np[:, s], alpha=0.8, lw=1)
plt.xlabel("Disease Age (grid)")
plt.ylabel("Rate r(v) = exp(beta(v)^T x)")
plt.title("Predicted rate curves (first 30 subjects)")
plt.grid(True, alpha=0.3)
plt.show()
