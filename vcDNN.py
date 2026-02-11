"""
Data assumptions:
- dage.csv -> (T,) with T=250, includes (or is near) 0
- f_pred.csv -> (T, S) with S=1000 subjects
- Xsub.csv -> (S, P) subject covariates (P=4)
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
X_sub_df = pd.read_csv("data/Xsub.csv")        # (S,P)
f_df     = pd.read_csv("data/f_pred.csv")      # (T,S)
dage_df  = pd.read_csv("data/dage.csv")        # (T,1) or (T,)

# ---------------- BASIC SHAPES ----------------
X_sub_np = X_sub_df.to_numpy().astype(np.float32)              # [S,P]
f_np     = f_df.to_numpy().astype(np.float32)                  # [T,S]
dage_np  = dage_df.to_numpy().reshape(-1).astype(np.float32)   # [T]

S, P = X_sub_np.shape
T = len(dage_np)

assert f_np.shape == (T, S), f"Expected f_pred shape {(T,S)} but got {f_np.shape}"

# Torch tensors
X_sub_all = torch.tensor(X_sub_np)                 # [S,P]
Y_all     = torch.tensor(f_np)                     # [T,S]
t_full    = torch.tensor(dage_np)                  # [T]

# Optional: check nearest gridpoint to 0
i0 = torch.argmin(torch.abs(t_full - 0.0))
print("Nearest gridpoint to 0:", float(t_full[i0].item()))

# ---------------- SPLIT BY SUBJECTS ----------------
sub_idx = np.arange(S)
sub_train, sub_test = train_test_split(sub_idx, test_size=0.2, random_state=SEED, shuffle=True)
sub_train, sub_val  = train_test_split(sub_train, test_size=0.25, random_state=SEED, shuffle=True)  # 0.25 of 0.8 -> 0.2

# Slice subject matrices
X_train = X_sub_all[sub_train]          # [S_train,P]
X_val   = X_sub_all[sub_val]            # [S_val,P]
X_test  = X_sub_all[sub_test]           # [S_test,P]

Ytrain = Y_all[:, sub_train]            # [T,S_train]
Yval   = Y_all[:, sub_val]              # [T,S_val]
Ytest  = Y_all[:, sub_test]             # [T,S_test]

# ---------------- NORMALIZE INPUT GRID (TRAIN ONLY) ----------------
# We normalize the independent variable (dage) once using the full grid,
# but you can also use train-only stats; here we use full-grid stats to keep it simple.
v_mean = float(t_full.mean().item())
v_std  = float(t_full.std().item() + 1e-8)

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

    def forward(self, t_norm):        # [batch,1]
        return self.net(t_norm)       # [batch,P]

model = VCNet(P)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

# ---------------- ODE INTEGRATION HELPERS ----------------
def make_rhs(X_sub_mat):
    """
    Returns an RHS function rhs(t, y) for a given subject covariate matrix X_sub_mat [S_split,P].
    """
    # Precompute transpose once
    X_T = X_sub_mat.T  # [P,S_split]

    def rhs(t, y):
        # t scalar -> [1,1]
        t_in = norm_t(t).view(1, 1)
        beta = model(t_in)                  # [1,P]
        rate = torch.exp(beta @ X_T).squeeze(0)  # [S_split]
        return rate

    return rhs

def integrate_on_grid(t_grid, X_sub_mat, step_size=0.25, method="rk4"):
    S_split = X_sub_mat.shape[0]
    y0 = torch.zeros(S_split, dtype=torch.float32)

    rhs = make_rhs(X_sub_mat)

    g = odeint(rhs, y0, t_grid, method=method, options={"step_size": step_size})
    # g shape: [T, S_split]

    # find index of 0
    i0 = torch.argmin(torch.abs(t_grid - 0.0))

    # shift so that f(0)=1
    f = g - g[i0] + 1.0

    return f

# ---------------- TRAIN LOOP ----------------
EPOCHS = 2000
step_size = 0.25

for epoch in range(EPOCHS):
    model.train()
    opt.zero_grad()

    pred_train = integrate_on_grid(t_full, X_train, step_size=step_size, method="rk4")  # [T,S_train]
    loss = loss_fn(pred_train, Ytrain)
    loss.backward()
    opt.step()

    if epoch % 200 == 0:
        model.eval()
        with torch.no_grad():
            pred_val = integrate_on_grid(t_full, X_val, step_size=step_size, method="rk4")
            val_loss = loss_fn(pred_val, Yval).item()
        print(f"Epoch {epoch:4d} | train {loss.item():.6f} | val {val_loss:.6f}")

# ---------------- TEST EVAL ----------------
model.eval()
with torch.no_grad():
    pred_test = integrate_on_grid(t_full, X_test, step_size=step_size, method="rk4")
    test_mse = loss_fn(pred_test, Ytest).item()
print(f"Test MSE: {test_mse:.6f}")

# ---------------- MC DROPOUT (FOR TEST SUBJECTS) ----------------
def enable_dropout(m):
    if isinstance(m, nn.Dropout):
        m.train()

def mc_dropout_predict(t_grid, X_sub_mat, n_samples=200):
    """
    Returns:
      mean_pred: [T,S_split]
      lower_95 : [T,S_split]
      upper_95 : [T,S_split]
      samples  : [K,T,S_split]
    """
    model.eval()
    model.apply(enable_dropout)

    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            samples.append(integrate_on_grid(t_grid, X_sub_mat, step_size=step_size, method="rk4"))
    samples = torch.stack(samples, dim=0)  # [K,T,S_split]

    mean_pred = samples.mean(dim=0)
    lower_95 = torch.quantile(samples, 0.025, dim=0)
    upper_95 = torch.quantile(samples, 0.975, dim=0)
    return mean_pred, lower_95, upper_95, samples

mean_test, lower_test, upper_test, samples_test = mc_dropout_predict(t_full, X_test, n_samples=200)
print("MC samples shape:", samples_test.shape)  # [K,T,S_test]


# Print CI for a single test subject (index within test split)
s = 0
for i in [i0, min(i0+10, T-1), T-1]:
    print(f"t={t_full[i].item():.3f}: 95% CI (test-subject {s}) = "
          f"({lower_test[i,s].item():.4f}, {upper_test[i,s].item():.4f})")

# ---------------- PLOT: integrated trajectories for first N test subjects ----------------
pred_test_np = pred_test.cpu().numpy()

plt.figure(figsize=(9,6))
Nplot = min(50, pred_test_np.shape[1])
for s in range(Nplot):
    plt.plot(dage_np, pred_test_np[:, s], alpha=0.7, lw=1)
plt.xlabel("Disease Age (grid)")
plt.ylabel("Integrated progression f(v)")
plt.title(f"Predicted integrated trajectories (first {Nplot} TEST subjects)")
plt.grid(True, alpha=0.3)
plt.show()

# ---------------- PLOT: rate curves for first N test subjects ----------------
model.eval()
with torch.no_grad():
    t_in = norm_t(t_full).view(-1, 1)          # [T,1]
    beta_full = model(t_in)                    # [T,P]
    rate_test = torch.exp(beta_full @ X_test.T)  # [T,S_test]

rate_test_np = rate_test.cpu().numpy()

plt.figure(figsize=(9,6))
Nplot = min(30, rate_test_np.shape[1])
for s in range(Nplot):
    plt.plot(dage_np, rate_test_np[:, s], alpha=0.7, lw=1)
plt.xlabel("Disease Age (grid)")
plt.ylabel("Rate r(v) = exp(beta(v)^T x)")
plt.title(f"Predicted rate curves (first {Nplot} TEST subjects)")
plt.grid(True, alpha=0.3)
plt.show()
