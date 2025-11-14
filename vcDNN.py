from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

VCData = pd.read_csv("vc_training_and_test_set.csv")
v  = pd.read_csv("v_pred.csv")   # v_pred
vc = pd.read_csv("beta_pred.csv")          # (N, P), actual VC function values
X_sub = pd.read_csv("X_sub.csv")
rate = pd.read_csv("r_pred.csv")
P  = vc.shape[1]

# split 60/20/20
v_train, v_test, vc_train, vc_test = train_test_split(v, vc, test_size=0.2, random_state=42)
v_train, v_val, vc_train, vc_val = train_test_split(v_train, vc_train, test_size=0.25, random_state=42)

# scale with TRAIN ONLY
v_mean, v_std = v_train.mean(), v_train.std()
norm = lambda x: (x - v_mean) / (v_std + 1e-8)

Xtrain, Xval, Xtest = map(norm, (v_train, v_val, v_test))

# model
def VCNet(P, hidden=(64,64), dropout=0.2):
    layers, in_dim = [], 1
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = h
    layers += [nn.Linear(in_dim, P)]
    return nn.Sequential(*layers)

def rate_fn(X_sub, vc):
    return torch.exp((X_sub * vc).sum(dim=1, keepdim=True))

# convert to tensor
X_sub = torch.tensor(X_sub).float()
rate = torch.tensor(rate).float()

Xtrain = torch.tensor(Xtrain)
Xval = torch.tensor(Xval)
Xtest = torch.tensor(Xtest)

Ytrain =   # rate train = exp(xsub * vc train)
Yval = 
Ytest = rate

model = VCNet(P)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

# train
for epoch in range(2000):
    model.train()
    opt.zero_grad()
    rate_pred = rate_fn(model(Xtrain), X_sub) # get rate prediction
    loss = loss_fn(rate_pred, Ytrain) # loss on rate
    loss.backward()
    opt.step()
    if epoch % 200 == 0:
        model.eval()
        with torch.no_grad():
            rate_val = rate_fn(model(Xval), X_sub)
            val = loss_fn(rate_val, Yval).item()
        print(f"Epoch {epoch:4d} | train {loss.item():.6f} | val {val:.6f}")

# --- evaluate & plot properly ---
vmin, vmax = v_test.min(), v_test.max() # plot only test range for interpolation
v_grid = np.linspace(vmin, vmax, 200).astype(np.float32).reshape(-1,1)
model.eval()
with torch.no_grad():
    vc_pred_grid = model(torch.from_numpy(norm(v_grid))).cpu().numpy()
# Plot: use the full truth (sorted by v) and the smooth predicted line
order = np.argsort(v_test[:,0]) # indices to sort test set, so that v and vc have the same index order for plotting
v_sorted  = v_test[order, 0]
vc_sorted = vc_test[order]

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