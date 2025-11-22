from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

v  = pd.read_csv("v_pred.csv")      # v_pred
vc = pd.read_csv("beta_pred.csv")   # (N, P), actual VC function values
X_sub = pd.read_csv("X_sub.csv")    # (n_subjects, P)
rate = pd.read_csv("r_pred.csv")    # (N, n_subjects)
P  = vc.shape[1]

# split 60/20/20
v_train, v_test, vc_train, vc_test, rate_train, rate_test = train_test_split(v, vc, rate, test_size=0.2, random_state=42)
v_train, v_val, vc_train, vc_val, rate_train, rate_val = train_test_split(v_train, vc_train, rate_train, test_size=0.25, random_state=42)

# scale with TRAIN ONLY
v_mean, v_std = float(v_train.mean().iloc[0]), float(v_train.std().iloc[0])
norm = lambda x: (x - v_mean) / (v_std)

# model
def VCNet(P, hidden=(64,64), dropout=0.2):
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
    returns rate: [N, n_subjects]
    """
    return torch.exp(vc @ X_sub.T) # transpose 

Xtrain, Xval, Xtest = map(norm, (v_train, v_val, v_test))
# convert to tensor
X_sub = torch.tensor(X_sub.to_numpy()).float()
rate = torch.tensor(rate.to_numpy()).float()

Xtrain = torch.tensor(Xtrain.to_numpy()).float()
Xval = torch.tensor(Xval.to_numpy()).float()
Xtest = torch.tensor(Xtest.to_numpy()).float()

Ytrain = torch.tensor(rate_train.to_numpy()).float()
Yval = torch.tensor(rate_val.to_numpy()).float()
Ytest = torch.tensor(rate_test.to_numpy()).float()

model = VCNet(P)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

# train
for epoch in range(2000):
    model.train()
    opt.zero_grad()

    vc_train_pred = model(Xtrain)                 # [N_train, P]
    rate_pred     = rate_fn(vc_train_pred, X_sub) # [N_train, n_subjects]
    loss = loss_fn(rate_pred, Ytrain)

    loss.backward()
    opt.step()
    if epoch % 200 == 0:
        model.eval()
        with torch.no_grad():
            rate_val = rate_fn(model(Xval), X_sub)
            val = loss_fn(rate_val, Yval).item()
        print(f"Epoch {epoch:4d} | train {loss.item():.6f} | val {val:.6f}")

model.eval()
with torch.no_grad():
    vc_test_pred  = model(Xtest)                 # [N_test, P]
    rate_test_pred = rate_fn(vc_test_pred, X_sub)  # [N_test, n_subjects]
    test_mse = loss_fn(rate_test_pred, Ytest).item()
print(f"Test MSE: {test_mse}")
# --- evaluate & plot properly ---
vmin, vmax = float(v_test.min().iloc[0]), float(v_test.max().iloc[0]) # plot only test range for interpolation
v_grid = np.linspace(vmin, vmax, 200).astype(np.float32).reshape(-1,1)
with torch.no_grad():
    vc_pred = model(torch.from_numpy(norm(v_grid)))
    rate_pred = rate_fn(vc_pred, X_sub).cpu().numpy()

# Plot: use the full truth (sorted by v) and the smooth predicted line

order = np.argsort(v_test.to_numpy()[:,0]) # indices to sort test set, so that v and rate have the same index order for plotting
v_sorted  = v_test.to_numpy()[order, 0]
rate_sorted = rate_test.to_numpy()[order]

plt.figure(figsize=(8,5))
for j in range(P):
    if j == 0:
        plt.scatter(v_sorted, rate_sorted[:, j], s=25, alpha=0.6, label="True (test)")
        plt.plot(v_grid[:, 0], rate_pred[:, j], label="Predicted")
    else:
        plt.scatter(v_sorted, rate_sorted[:, j], s=25, alpha=0.6)
        plt.plot(v_grid[:, 0], rate_pred[:, j])
plt.xlabel("v"); plt.ylabel("rate values")
plt.legend()
plt.show()