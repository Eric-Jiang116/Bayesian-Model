from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

v  = pd.read_csv("data/v_pred.csv")    # v_pred (50, 1)
vc = pd.read_csv("data/beta_pred.csv")  # (N, P), actual VC function values
Xsub = pd.read_csv("data/Xsub.csv")   # (n_subjects, P)
rate = pd.read_csv("data/r_pred.csv")   # (N, n_subjects)
P = vc.shape[1]

# split 60/20/20
Xsub_train, Xsub_test, rate_train, rate_test = train_test_split(Xsub, rate.T, test_size=0.2, random_state=42)
Xsub_train, Xsub_val, rate_train, rate_val = train_test_split(Xsub_train, rate_train, test_size=0.25, random_state=42)

# Normalize Xsub_train
# scaler = StandardScaler()
# Xsub_train_norm = scaler.fit_transform(Xsub_train)  # fit only on train
# Xsub_val_norm   = scaler.transform(Xsub_val)        # same scale as train
# Xsub_test_norm  = scaler.transform(Xsub_test)       # same scale as train

# convert to tensor
X_sub = torch.tensor(Xsub.to_numpy()).float()
rate_tensor = torch.tensor(rate.to_numpy()).float()
v_np = v.to_numpy().reshape(-1).astype(np.float32)
v = torch.tensor(v.to_numpy()).float()

Xtrain = torch.tensor(Xsub_train.to_numpy()).float()
Xval = torch.tensor(Xsub_val.to_numpy()).float()
Xtest = torch.tensor(Xsub_test.to_numpy()).float()

Ytrain = torch.tensor(rate_train.to_numpy()).float().T
Yval = torch.tensor(rate_val.to_numpy()).float().T
Ytest = torch.tensor(rate_test.to_numpy()).float().T

# model
def VCNet(P, hidden=(128, 256), dropout=0.2):
    layers, in_dim = [], 1 # input is 1D: time variable
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = h
    layers += [nn.Linear(in_dim, P)]
    return nn.Sequential(*layers)

# rate function for training
def rate_fn(vc, X_sub):
    """
    beta: [N, P]
    X_sub: [n_subjects, P]
    returns rate: [N, n_subjects] returns 10 x 100 matrix
    """
    return torch.exp(vc @ X_sub.T) # transpose 

model = VCNet(P)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

# train
for epoch in range(2000):
    model.train()
    opt.zero_grad()

    vc_train_pred = model(v)                 # [N_train, P]
    rate_pred = rate_fn(vc_train_pred, Xtrain) # [N_train, n_subjects]
    loss = loss_fn(rate_pred, Ytrain)

    loss.backward()
    opt.step()
    if epoch % 200 == 0:
        model.eval()
        with torch.no_grad():
            rate_val_pred = rate_fn(model(v), Xval)
            val = loss_fn(rate_val_pred, Yval).item()
        print(f"Epoch {epoch:4d} | train {loss.item():.6f} | val {val:.6f}")

model.eval()
# MC dropout: evaluate on test set
for m in model.modules():
    if isinstance(m, nn.Dropout):
        m.train()  # keep dropout on

def mc_dropout_predict(model, X, n_samples=1000):
    preds = []
    for i in range(n_samples):
        with torch.no_grad():
            vc_pred = model(v)
            rate_pred = rate_fn(vc_pred, Xtest)
            preds.append(rate_pred)  # [1, N_test, n_subjects]
    preds = torch.stack(preds, dim=0) # [n_samples, N_test, n_subjects]
    print(preds.shape)
    scalar_preds = preds.mean(dim=-1)  # [n_samples, N_test] 

    # mean over MC samples
    mean_pred = scalar_preds.mean(dim=0)          # [N_test] 

    # 95% CI via percentiles over MC samples
    lower_95 = torch.quantile(scalar_preds, 0.025, dim=0)  # [N_test] percentiles of the mean predictions
    upper_95 = torch.quantile(scalar_preds, 0.975, dim=0)  # [N_test]
    return mean_pred, lower_95, upper_95, scalar_preds, preds

mean_pred, lower, upper, preds, lol= mc_dropout_predict(model, Xtest)
pred_np = preds.cpu().numpy()
print(lol.shape[2])
for j in range(len(lower)):
    print(f"Test point {j + 1}: 95% CI = {(lower[j], upper[j])}\n")

with torch.no_grad():
    vc_test_pred  = model(v)                 # [N_test, P]
    rate_test_pred = rate_fn(vc_test_pred, Xtest)  # [N_test, n_subjects]
    test_mse = loss_fn(rate_test_pred, Ytest).item()
print(f"Test MSE: {test_mse}")

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "X_test": Xtest,
    "Y_test": Ytest
}, "model_checkpoint.pt")

# --- evaluate & plot properly ---
model.eval()
vmin, vmax = float(v.min()), float(v.max()) # plot only test range for interpolation
v_grid = np.linspace(vmin, vmax, 50).astype(np.float32).reshape(-1,1)
with torch.no_grad():
    vc_hat = model(v)
    rate_pred = rate_fn(vc_hat, Xtest).cpu().numpy()

# Sort order of values
order = np.argsort(v[:,0]) # indices to sort test set, so that v and rate have the same index order for plotting
v_sorted  = v[order, 0]
rate_sorted = rate_test.to_numpy()[order]
mean_sorted = mean_pred.cpu().numpy()[order]
lower_sorted = lower.cpu().numpy()[order]
upper_sorted = upper.cpu().numpy()[order]

plt.figure(figsize=(9, 6))

for s in range(rate_pred.shape[1]):  # all subjects
    plt.plot(
        v_grid[:, 0],
        rate_pred[:, s],
        alpha=0.2,      # IMPORTANT: transparency
        lw=1
    )

plt.xlabel("v")
plt.ylabel("Predicted Rate")
plt.title("Rate vs v_pred — All Subjects")
plt.grid(True, alpha=0.3)
plt.show()


vc_pred_np = vc.to_numpy().astype(np.float32) 
vc_hat_np = vc_hat.cpu().numpy()

# Predicted vc vs true vc 
fig, axes = plt.subplots(P, 1, sharex=True)
for j in range(P):
    axes[j].plot(v_np, vc_pred_np[:, j], "k--", lw=2, label="beta_pred")
    axes[j].plot(v_np, vc_hat_np[:, j], lw=2, label="beta_hat (mapped)")
    axes[j].set_ylabel(f"beta[{j}]")
    axes[j].grid(True, alpha=0.3)
    axes[j].legend()
axes[-1].set_xlabel("v (v_pred grid)")
fig.suptitle("Varying coefficients: beta_hat(v) vs beta_pred(v)", y=0.995)
plt.tight_layout()
plt.show()

# # Plot the mean rate + 95% CI
# plt.fill_between(v_sorted, lower_sorted, upper_sorted, color='lightblue', alpha=0.4, label='95% CI')
# plt.plot(v_sorted, mean_sorted, color='blue', label='Mean Rate')
# plt.xlabel("v")
# plt.ylabel("rate(v)")
# plt.title("Mean Subject-Weighted Rate Function with 95% MC Dropout CI")
# plt.legend()
# plt.show()