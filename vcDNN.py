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
X_sub_df = pd.read_csv("data/Xsub.csv")     # [1000, 4]
f_df     = pd.read_csv("data/f_pred.csv")   # [250, 1000]
dage_df  = pd.read_csv("data/dage.csv")     # [250, 1]
v_df = pd.read_csv("data/v_pred.csv")       # [50, 1]
beta_pred_df = pd.read_csv("data/beta_pred.csv")   # [50, 4]
# ---------------- 2. BASIC SHAPES & TENSORS ----------------
P = X_sub_df.shape[1]
n_subjects = X_sub_df.shape[0]

v_np = v_df.to_numpy().reshape(-1).astype(np.float32)
v_grid = torch.tensor(v_np, dtype=torch.float32)
beta_pred_np = beta_pred_df.to_numpy().astype(np.float32) 

dage_np = dage_df.to_numpy().reshape(-1).astype(np.float32)
t_grid = torch.tensor(dage_np, dtype=torch.float32)  # [T]
T = len(dage_np)

f_np = f_df.to_numpy().astype(np.float32)            # [T, S]
X_all = torch.tensor(X_sub_df.to_numpy().astype(np.float32))  # [S, P]

assert f_np.shape == (T, n_subjects), f"Expected {(T,n_subjects)}, got {f_np.shape}"
K = len(v_np)
assert beta_pred_np.shape == (K, P), f"Expected beta_pred {(K,P)} got {beta_pred_np.shape}"

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
v_mean = float(v_np.mean().item())
v_std  = float(v_np.std().item() + 1e-8)

def norm_v(v):
    return (v - v_mean) / v_std

# ---------------- 5. MODEL & ODE ----------------
class VCNet(nn.Module):
    def __init__(self, P, hidden=(64, 64), dropout=0.2):
        super().__init__()
        layers = []
        in_dim = 1
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        last_layer = nn.Linear(in_dim, P)
        nn.init.zeros_(last_layer.weight) # start with zero output
        nn.init.zeros_(last_layer.bias)
        layers += [last_layer]
        self.net = nn.Sequential(*layers)

    def forward(self, t_norm):  # [batch,1]
        return self.net(t_norm) # [batch,P]

model = VCNet(P)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

def get_ode_rhs(current_X):
    X_T = current_X.T  # [P, S_split]
    def rhs(t, y):
        y_norm = norm_v(y).view(-1, 1)    # [S_split,1]
        beta = model(y_norm)             # [S_split,P]
        exponent = (beta * current_X).sum(dim=1)     # [S]
        rate = torch.exp(exponent.clamp(-5, 5))      # also clamp
        return rate
    return rhs

def predict_trajectories(t_grid, current_X, method="euler", step_size=0.25):
    rhs = get_ode_rhs(current_X)
    y0 = torch.ones(current_X.shape[0], dtype=torch.float32)  # initial value at 1
    forward = odeint(rhs, y0, t_grid[i0:], method=method, options={"step_size": step_size})  # [T, S_split]
    backward = odeint(rhs, y0, t_grid[:i0+1].flip(0), method=method, options={"step_size": step_size}).flip(0) # [T, S_split]
    F = torch.cat([backward[:-1], forward], dim=0)
    return F

# ---------------- 6. TRAINING ----------------
EPOCHS = 1000
for epoch in range(EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    preds = predict_trajectories(t_grid, X_train)  # [T, S_train]

    # print("pred range at init:", preds.min().item(), preds.max().item())
    # print("target range:", Y_train[:, :5].min().item(), Y_train[:, :5].max().item())
    loss = loss_fn(preds, Y_train)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            val_preds = predict_trajectories(t_grid, X_val)
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
            all_samples.append(predict_trajectories(t_grid, current_X))
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
    plt.plot(dage_np, mean_test[:, s].cpu())
    plt.fill_between(dage_np, low_test[:, s].cpu(), high_test[:, s].cpu(), alpha=0.1)

plt.axvline(0.0, color='gray', linestyle=':', alpha=0.7)
plt.axhline(1.0, color='red', linestyle=':', label="f(0)=1")
plt.title("Subject-Based Split: Predicted Trajectories (Test Set)")
plt.xlabel("Disease Age")
plt.ylabel("f(v)")
plt.legend()
plt.show()

# # Ranges
# d_min, d_max = float(dage_np.min()), float(dage_np.max())
# v_min, v_max = float(v_pred_np.min()), float(v_pred_np.max())
# print("dage range:", d_min, d_max)
# print("v_pred range:", v_min, v_max)

# # Inverse map: v -> dage
# def v_to_dage(v):
#     return d_min + (v - v_min) * (d_max - d_min) / (v_max - v_min)

# dage_at_v = v_to_dage(v_pred_np).astype(np.float32)    # [K]
# dage_at_v_t = torch.tensor(dage_at_v, dtype=torch.float32)

# Evaluate beta_hat at these mapped dage locations
model.eval()
with torch.no_grad():
    v_norm = norm_v(v_grid).view(-1, 1)     # [K,1] in the SAME normalization as training
    beta_hat_np = model(v_norm).cpu().numpy()    # [K,P]

# Plot beta_hat(v) vs beta_pred(v) on the v axis
fig, axes = plt.subplots(P, 1, sharex=True)
for j in range(P):
    axes[j].plot(v_np, beta_pred_np[:, j], "k--", lw=2, label="beta_pred")
    axes[j].plot(v_np, beta_hat_np[:, j], lw=2, label="beta_hat (mapped)")
    axes[j].set_ylabel(f"beta[{j}]")
    axes[j].grid(True, alpha=0.3)
    axes[j].legend()
axes[-1].set_xlabel("v (v_pred grid)")
fig.suptitle("Varying coefficients: beta_hat(v) vs beta_pred(v)\n(using affine v↔dage mapping)", y=0.995)
plt.tight_layout()
plt.show()

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "mean_test": mean_test,
    "low_test":low_test,
    "high_test": high_test,
    "v_mean": v_mean,
    "v_std": v_std,
    "X_test": X_test,
    "Y_test": Y_test
}, "model_checkpoint.pt")

model.eval()
with torch.no_grad():
    v_norm = norm_v(v_grid).view(-1, 1)
    beta_hat = model(v_norm)  # [K, P]

# Compare the linear combination for each subject
# True combined effect
true_combined = beta_pred_np @ X_test.numpy().T   # [K, S_test]
hat_combined  = beta_hat.numpy() @ X_test.numpy().T  # [K, S_test]

for s in range(5):
    corr = np.corrcoef(true_combined[:, s], hat_combined[:, s])[0, 1]
    print(f"Subject {s} combined effect correlation: {corr:.4f}")