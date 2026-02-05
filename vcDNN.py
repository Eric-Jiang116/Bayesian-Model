from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

v  = pd.read_csv("data/v_pred.csv")    # v_pred (50, 1)
vc = pd.read_csv("data/beta_pred.csv")  # (N, P), actual VC function values
Xsub = pd.read_csv("data/Xsub.csv")   # (n_subjects, P)
rate = pd.read_csv("data/r_pred.csv")   # (N, n_subjects)
P = vc.shape[1]

# split 60/20/20
# v_train, v_test, vc_train, vc_test, rate_train, rate_test = train_test_split(v, vc, rate, test_size=0.2, random_state=42)
# v_train, v_val, vc_train, vc_val, rate_train, rate_val = train_test_split(v_train, vc_train, rate_train, test_size=0.25, random_state=42)
Xsub_train, Xsub_test, rate_train, rate_test = train_test_split(Xsub, rate.T, test_size=0.2, random_state=42)
Xsub_train, Xsub_val, rate_train, rate_val = train_test_split(Xsub_train, rate_train, test_size=0.25, random_state=42)

# # scale with TRAIN ONLY 
#v_mean, v_std = float(v_train.mean().iloc[0]), float(v_train.std().iloc[0])
# norm = lambda x: (x - v_mean) / (v_std)

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
    returns rate: [N, n_subjects] returns 10 x 100 matrix
    """
    return torch.exp(vc @ X_sub.T) # transpose 

# Xtrain, Xval, Xtest = map(norm, (v_train, v_val, v_test))
# convert to tensor
X_sub = torch.tensor(Xsub.to_numpy()).float()
rate = torch.tensor(rate.to_numpy()).float()
v = torch.tensor(v.to_numpy()).float()

Xtrain = torch.tensor(Xsub_train.to_numpy()).float()
Xval = torch.tensor(Xsub_val.to_numpy()).float()
Xtest = torch.tensor(Xsub_test.to_numpy()).float()

Ytrain = torch.tensor(rate_train.to_numpy()).float().T
Yval = torch.tensor(rate_val.to_numpy()).float().T
Ytest = torch.tensor(rate_test.to_numpy()).float().T

model = VCNet(P)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

v_norm = (v - v.mean()) / v.std()
# train
for epoch in range(2000):
    model.train()
    opt.zero_grad()

    vc_train_pred = model(v_norm)                 # [N_train, P]
    rate_pred = rate_fn(vc_train_pred, Xtrain) # [N_train, n_subjects]
    loss = loss_fn(rate_pred, Ytrain)

    loss.backward()
    opt.step()
    if epoch % 200 == 0:
        model.eval()
        with torch.no_grad():
            rate_val = rate_fn(model(v_norm), Xval)
            val = loss_fn(rate_val, Yval).item()
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
            vc_pred = model(v_norm)
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
    vc_test_pred  = model(v_norm)                 # [N_test, P]
    rate_test_pred = rate_fn(vc_test_pred, Xtest)  # [N_test, n_subjects]
    test_mse = loss_fn(rate_test_pred, Ytest).item()
print(f"Test MSE: {test_mse}")

# --- evaluate & plot properly ---
model.eval()
vmin, vmax = float(v.min()), float(v.max()) # plot only test range for interpolation
v_grid = np.linspace(vmin, vmax, 50).astype(np.float32).reshape(-1,1)
with torch.no_grad():
    vc_pred = model(v_norm)
    rate_pred = rate_fn(vc_pred, Xtest).cpu().numpy()

# Sort order of values
order = np.argsort(v[:,0]) # indices to sort test set, so that v and rate have the same index order for plotting
v_sorted  = v[order, 0]
rate_sorted = rate_test.to_numpy()[order]
mean_sorted = mean_pred.cpu().numpy()[order]
lower_sorted = lower.cpu().numpy()[order]
upper_sorted = upper.cpu().numpy()[order]

# # Plot predictions vs truth
# plt.figure(figsize=(8,5))
# for j in range(6):  # for each subject
#     if j == 0:
#         plt.scatter(v_sorted, rate_sorted[:, j], s=25, alpha=0.6, label="True (test)")
#         plt.plot(v_grid[:, 0], rate_pred[:, j], label="Predicted")
#     else:
#         plt.scatter(v_sorted, rate_sorted[:, j], s=25, alpha=0.6)
#         plt.plot(v_grid[:, 0], rate_pred[:, j])
# plt.xlabel("v"); plt.ylabel("rate values")
# plt.legend()
# plt.show()

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

# # Plot the mean rate + 95% CI
# plt.fill_between(v_sorted, lower_sorted, upper_sorted, color='lightblue', alpha=0.4, label='95% CI')
# plt.plot(v_sorted, mean_sorted, color='blue', label='Mean Rate')
# plt.xlabel("v")
# plt.ylabel("rate(v)")
# plt.title("Mean Subject-Weighted Rate Function with 95% MC Dropout CI")
# plt.legend()
# plt.show()