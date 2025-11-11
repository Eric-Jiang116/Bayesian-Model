from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

VCData = pd.read_csv("vc_training_and_test_set.csv")
v  = VCData.iloc[:, 0].to_numpy(np.float32).reshape(-1, 1)   # v_pred
vc = VCData.iloc[:, 1:].to_numpy(np.float32)                  # (N, P), actual VC function values
P  = vc.shape[1]

# 60/20/20
v_tr, v_te, vc_tr, vc_te = train_test_split(v, vc, test_size=0.2, random_state=42)
v_tr, v_va, vc_tr, vc_va = train_test_split(v_tr, vc_tr, test_size=0.25, random_state=42)

# scale with TRAIN ONLY
v_mean, v_std = v_tr.mean(), v_tr.std()
norm = lambda x: (x - v_mean) / (v_std + 1e-8)

Xtr, Xva, Xte = map(norm, (v_tr, v_va, v_te))

Xtr = torch.tensor(Xtr); Ytr = torch.tensor(vc_tr)
Xva = torch.tensor(Xva); Yva = torch.tensor(vc_va)
Xte = torch.tensor(Xte); Yte = torch.tensor(vc_te)

# model
def VCNet(P, hidden=(64,64), dropout=0.2):
    layers, in_dim = [], 1
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = h
    layers += [nn.Linear(in_dim, P)]
    return nn.Sequential(*layers)

model = VCNet(P)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

# train
for epoch in range(2000):
    model.train()
    opt.zero_grad()
    loss = loss_fn(model(Xtr), Ytr)
    loss.backward(); opt.step()
    if epoch % 200 == 0:
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xva), Yva).item()
        print(f"Epoch {epoch:4d} | train {loss.item():.6f} | val {val:.6f}")

# --- evaluate & plot properly ---
vmin, vmax = v_te.min(), v_te.max() # plot only test range for interpolation
v_grid = np.linspace(vmin, vmax, 200).astype(np.float32).reshape(-1,1)
model.eval()
with torch.no_grad():
    vc_pred_grid = model(torch.from_numpy(norm(v_grid))).cpu().numpy()
# Plot: use the full truth (sorted by v) and the smooth predicted line
order = np.argsort(v_te[:,0]) # indices to sort test set, so that v and vc have the same index order for plotting
v_sorted  = v_te[order, 0]
vc_sorted = vc_te[order]

plt.figure(figsize=(8,5))
for j in range(P):
    if j == 0:
        plt.scatter(v_sorted, vc_sorted[:, j], s=25, alpha=0.6, label="True (test)")
        plt.plot(v_grid[:, 0], vc_pred_grid[:, j], label="Predicted")
    else:
        plt.scatter(v_sorted, vc_sorted[:, j], s=25, alpha=0.6)
        plt.plot(v_grid[:, 0], vc_pred_grid[:, j])
plt.xlabel("v"); plt.ylabel("VC function values")
plt.legend()
plt.show()