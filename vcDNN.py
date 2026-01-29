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

out_data = pd.read_csv("out.data.csv")
v  = pd.read_csv("v_pred.csv")    # v_pred
vc = pd.read_csv("beta_pred.csv")  # (N, P), actual VC function values
X_sub = pd.read_csv("X_sub.csv")   # (n_subjects, P)
rate = pd.read_csv("r_pred.csv")   # (N, n_subjects)
P = vc.shape[1]
f = pd.read_csv("f_pred.csv")       # f_pred = integrated rate value
dage = out_data["dage"]             #true disease age
t = out_data["t"]

# --------- TRAIN_VAL_TEST_SPLIT -----------
v_train, v_test, vc_train, vc_test = train_test_split(v, vc, test_size=0.2, random_state=42)
v_train, v_val, vc_train, vc_val = train_test_split(v_train, vc_train, test_size=0.25, random_state=42)

f_train, f_test = train_test_split(f, test_size=0.2, random_state=42)
f_train, f_val = train_test_split(f_train, test_size=0.25, random_state=42)

# scale with TRAIN ONLY 
v_mean, v_std = float(v_train.mean().iloc[0]), float(v_train.std().iloc[0])
norm = lambda x: (x - v_mean) / (v_std)

# --------- DNN MODEL ------------
def VCNet(P, hidden=(64,64), dropout=0.2):
    layers, in_dim = [], 1 # input is 1D: time variable
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = h
    layers += [nn.Linear(in_dim, P)]
    return nn.Sequential(*layers)

# --------- RATE FUNCTION ----------
def rate_fn(vc, X_sub):
    """
    beta: [N, P]
    X_sub: [n_subjects, P]
    returns rate: [N, n_subjects] returns 10 x 100 matrix
    """
    return torch.exp(vc @ X_sub.T) # transpose 

# --------- ODE SOLVER FUNC ------
def ode_fn(v, t, model, X_sub):
    """
    v: [n_subjects]
    returns dv/dt: [n_subjects]
    """
    rate = torch.exp(model @ X_sub.T).diag()  # [1, n_subjects]
    return rate  # [n_subjects, 1], use own covariate

Xtrain, Xval, Xtest = map(norm, (v_train, v_val, v_test))

# ------- CONVERT INPUT & OUTPUTS INTO TENSOR --------
X_sub = torch.tensor(X_sub.to_numpy()).float()
integrated_rate = torch.tensor(f.to_numpy()).float()

Xtrain = torch.tensor(Xtrain.to_numpy()).float()
Xval = torch.tensor(Xval.to_numpy()).float()
Xtest = torch.tensor(Xtest.to_numpy()).float()

Ytrain = torch.tensor(f_train.to_numpy()).float()
Yval = torch.tensor(f_val.to_numpy()).float()
Ytest = torch.tensor(f_test.to_numpy()).float()

# --------- INITIALIZE ----------
model = VCNet(P)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()
y0 = torch.zeros(X_sub.shape[0])       # initial y/accumulation values

# --------- TRAIN LOOP -----------
for epoch in range(2000):
    model.train()
    opt.zero_grad()

    vc_train_pred = model(Xtrain)                 # [N_train, P]
    rate_pred = ode_fn(v, t, vc_train_pred, X_sub)
    ode_pred = odeint(rate_pred, y0, t, method="euler", options=dict(step_size = 0.25))
    loss = loss_fn(ode_pred, Ytrain)
    loss.backward()
    opt.step()

    if epoch % 200 == 0:
        model.eval()
        with torch.no_grad():
            integrated_val = rate_fn(model(Xval), X_sub)
            val = loss_fn(integrated_val, Yval).item()
        print(f"Epoch {epoch:4d} | train {loss.item():.6f} | val {val:.6f}")

model.eval()
# MC dropout: evaluate on test set
for m in model.modules():
    if isinstance(m, nn.Dropout):
        m.train()  # keep dropout on

# MCMC Dropout (need to be updated)
def mc_dropout_predict(model, X, n_samples=1000):
    preds = []
    for i in range(n_samples):
        with torch.no_grad():
            vc_pred = model(X)
            rate_pred = ode_fn(v, t, vc_pred, X_sub)
            ode_pred = odeint(rate_pred, y0, t, method="euler", options=dict(step_size = 0.25))
            preds.append(ode_pred) 
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
    vc_test_pred  = model(Xtest)                 # [N_test, P]
    rate_test_pred = ode_fn(v, t, vc_test_pred, X_sub)  # [N_test, n_subjects]
    ode_test_pred = odeint(rate_test_pred, y0, t, method="euler", options=dict(step_size = 0.25))
    test_mse = loss_fn(ode_test_pred, Ytest).item()
print(f"Test MSE: {test_mse}")

# --- Evaluate & Plot Rate vs Value Curves ---
vmin, vmax = float(v_test.min().iloc[0]), float(v_test.max().iloc[0]) # plot only test range for interpolation
v_grid = np.linspace(vmin, vmax, 200).astype(np.float32).reshape(-1,1)
model.eval()
with torch.no_grad():
    vc_pred = model(torch.from_numpy(norm(v_grid)))
    rate_pred = ode_fn(vc_pred, X_sub).cpu().numpy()
    ode_pred = odeint(rate_pred, y0, t, method="euler", options=dict(step_size = 0.25))

ode_solution = ode_pred.detach().numpy()
plt.figure(figsize=(8,5))
for i in range(ode_solution.shape[1]):       # loop over subjects
    plt.plot(dage.numpy(), ode_solution[:, i], label=f"Subject {i+1}")

plt.xlabel("Disease Age")
plt.ylabel("ODE Solution / Accumulated Rate")
plt.title("ODE Trajectories per Subject")
plt.legend()
plt.show()

# Align order of values
# order = np.argsort(v_test.to_numpy()[:,0]) # indices to sort test set, so that v and rate have the same index order for plotting
# v_sorted  = v_test.to_numpy()[order, 0]
# rate_sorted = rate_test.to_numpy()[order]
# mean_sorted = mean_pred.cpu().numpy()[order]
# lower_sorted = lower.cpu().numpy()[order]
# upper_sorted = upper.cpu().numpy()[order]

# # ----- Integration ------
# model.eval() # disable dropout

# with torch.no_grad():
#     vc_grid = model(torch.from_numpy(norm(v_grid)))
#     # Get rates for ALL subjects at once
#     rate_pred_all = rate_fn(vc_grid, X_sub).numpy() # Shape: [200, n_subjects]
# plt.figure(figsize=(9, 6))

# # for s in range(rate_pred_all.shape[1]):  # all subjects
# #     plt.plot(
# #         v_grid[:, 0],
# #         rate_pred_all[:, s],
# #         alpha=0.2, 
# #         lw=1
# #     )

# plt.xlabel("v")
# plt.ylabel("Predicted Rate (dv/dt)")
# plt.title("Rate vs v_pred — All Subjects")
# plt.grid(True, alpha=0.3)
# plt.show()
