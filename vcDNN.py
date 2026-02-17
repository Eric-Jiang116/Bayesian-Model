import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from sklearn.model_selection import train_test_split

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------- 1. LOAD DATA ----------------
X_sub_df = pd.read_csv("data/Xsub.csv")     # [S, P]
f_df     = pd.read_csv("data/f_pred.csv")   # [T, S]
dage_df  = pd.read_csv("data/dage.csv")     # [T]
v_df = pd.read_csv("data/v_pred.csv")       # [P]
# ---------------- 2. BASIC SHAPES & TENSORS ----------------
P = X_sub_df.shape[1]
n_subjects = X_sub_df.shape[0]

dage_np = dage_df.to_numpy().reshape(-1).astype(np.float32)
t_grid = torch.tensor(dage_np, dtype=torch.float32)  # [T]
T = len(dage_np)

f_np = f_df.to_numpy().astype(np.float32)            # [T, S]
X_all = torch.tensor(X_sub_df.to_numpy().astype(np.float32))  # [S, P]

assert f_np.shape == (T, n_subjects), f"Expected {(T,n_subjects)}, got {f_np.shape}"

# index closest to onset (0)
i0 = int(torch.argmin(torch.abs(t_grid - 0.0)).item())
print("Nearest gridpoint to 0:", float(t_grid[i0].item()), "at index", i0)

# ---------------- 3. SUBJECT-BASED SPLIT ----------------
sub_indices = np.arange(n_subjects)
idx_train, idx_test = train_test_split(sub_indices, test_size=0.2, random_state=SEED, shuffle=True)
idx_train, idx_val  = train_test_split(idx_train, test_size=0.25, random_state=SEED, shuffle=True)

def get_split_tensors(indices):
    x_split = X_all[indices, :]                                 # [S_split, P]
    y_split = torch.tensor(f_np[:, indices], dtype=torch.float32) # [T, S_split]
    return x_split, y_split

X_train, Y_train = get_split_tensors(idx_train)
X_val,   Y_val   = get_split_tensors(idx_val)
X_test,  Y_test  = get_split_tensors(idx_test)

# ---------------- 4. NORMALIZATION ----------------
v_mean = float(t_grid.mean().item())
v_std  = float(t_grid.std().item() + 1e-8)

def norm_t(t):
    return (t - v_mean) / v_std

# ---------------- 5. MODEL & ODE ----------------
class VCNet(nn.Module):
    def __init__(self, P, hidden=(64, 64), dropout=0.2):
        super().__init__()
        layers = []
        in_dim = 1
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, P)]
        self.net = nn.Sequential(*layers)

    def forward(self, t_norm):  # [batch,1]
        return self.net(t_norm) # [batch,P]

model = VCNet(P)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

def get_ode_rhs(current_X):
    X_T = current_X.T  # [P, S_split]
    def rhs(t, y):
        t_in = norm_t(t).view(1, 1)    # [1,1]
        beta = model(t_in)             # [1,P]
        rate = torch.exp(beta @ X_T).squeeze(0)  # [S_split]
        return rate
    return rhs

def anchor_f0_1(F):
    """
    F: [T, S_split]
    returns: anchored so F(t=0) == 1 (using nearest gridpoint)
    """
    return F - F[i0:i0+1, :] + 1.0

def predict_trajectories(t_grid, current_X, method="rk4", step_size=0.25, anchor=True):
    # y0 is at t_grid[0] (≈ -30). Keep it 0 (or anything), then shift to enforce f(0)=1.
    y0 = torch.zeros(current_X.shape[0], dtype=torch.float32)  # [S_split]
    F = odeint(get_ode_rhs(current_X), y0, t_grid, method=method, options={"step_size": step_size})  # [T, S_split]
    return anchor_f0_1(F) if anchor else F

# ---------------- 6. TRAINING ----------------
# IMPORTANT: if your *targets* are supposed to be anchored at f(0)=1, anchor them too.
ANCHOR_TARGETS = False
if ANCHOR_TARGETS:
    Y_train = anchor_f0_1(Y_train)
    Y_val   = anchor_f0_1(Y_val)
    Y_test  = anchor_f0_1(Y_test)

EPOCHS = 1000
for epoch in range(EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    preds = predict_trajectories(t_grid, X_train, anchor=True)  # [T, S_train]
    loss = loss_fn(preds, Y_train)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            val_preds = predict_trajectories(t_grid, X_val, anchor=True)
            val_loss = loss_fn(val_preds, Y_val).item()
        print(f"Epoch {epoch:4d} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss:.6f}")

# ---------------- 7. MC DROPOUT ----------------
def enable_dropout(m):
    if isinstance(m, nn.Dropout):
        m.train()

def mc_dropout_predict(t_grid, current_X, n_samples=50):
    # Proper MC dropout: keep model in eval, but enable dropout layers
    model.eval()
    model.apply(enable_dropout)

    all_samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            all_samples.append(predict_trajectories(t_grid, current_X, anchor=True))
    samples = torch.stack(all_samples, dim=0)  # [K, T, S_split]

    mean = samples.mean(dim=0)
    lo = torch.quantile(samples, 0.025, dim=0)
    hi = torch.quantile(samples, 0.975, dim=0)
    return mean, lo, hi

mean_test, low_test, high_test = mc_dropout_predict(t_grid, X_test)

# ---------------- 8. VISUALIZATION ----------------
plt.figure(figsize=(10, 5))
for s in range(50):
    plt.plot(dage_np, Y_test[:, s].cpu(), 'k--', alpha=0.3, label="True" if s==0 else "")
    plt.plot(dage_np, mean_test[:, s].cpu(), label=f"Test subj {idx_test[s]}")
    plt.fill_between(dage_np, low_test[:, s].cpu(), high_test[:, s].cpu(), alpha=0.1)

plt.axvline(0.0, color='gray', linestyle=':', alpha=0.7)
plt.axhline(1.0, color='red', linestyle=':', label="Anchor f(0)=1")
plt.title("Subject-Based Split: Predicted Trajectories (Test Set) — anchored at onset")
plt.xlabel("Disease Age")
plt.ylabel("f(v)")
plt.legend()
plt.show()

beta_pred_df = pd.read_csv("data/beta_pred.csv")   # [K, P]
v_pred_df    = pd.read_csv("data/v_pred.csv")      # [K, 1] or [K]

v_pred_np = v_pred_df.to_numpy().reshape(-1).astype(np.float32)      # [K]
beta_pred_np = beta_pred_df.to_numpy().astype(np.float32)            # [K, P]
K = len(v_pred_np)
assert beta_pred_np.shape == (K, P), f"Expected beta_pred {(K,P)} got {beta_pred_np.shape}"

# Ranges
d_min, d_max = float(dage_np.min()), float(dage_np.max())
v_min, v_max = float(v_pred_np.min()), float(v_pred_np.max())
print("dage range:", d_min, d_max)
print("v_pred range:", v_min, v_max)

# Inverse map: v -> dage
def v_to_dage(v):
    return d_min + (v - v_min) * (d_max - d_min) / (v_max - v_min)

dage_at_v = v_to_dage(v_pred_np).astype(np.float32)    # [K]
dage_at_v_t = torch.tensor(dage_at_v, dtype=torch.float32)

# Evaluate beta_hat at these mapped dage locations
model.eval()
with torch.no_grad():
    t_in = norm_t(dage_at_v_t).view(-1, 1)     # [K,1] in the SAME normalization as training
    beta_hat_np = model(t_in).cpu().numpy()    # [K,P]

# Plot beta_hat(v) vs beta_pred(v) on the v axis
fig, axes = plt.subplots(P, 1, figsize=(9, 2.6*P), sharex=True)
for j in range(P):
    axes[j].plot(v_pred_np, beta_pred_np[:, j], "k--", lw=2, label="beta_pred")
    axes[j].plot(v_pred_np, beta_hat_np[:, j], lw=2, label="beta_hat (mapped)")
    axes[j].set_ylabel(f"beta[{j}]")
    axes[j].grid(True, alpha=0.3)
    axes[j].legend()
axes[-1].set_xlabel("v (v_pred grid)")
fig.suptitle("Varying coefficients: beta_hat(v) vs beta_pred(v)\n(using affine v↔dage mapping)", y=0.995)
plt.tight_layout()
plt.show()


