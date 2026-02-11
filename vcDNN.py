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

# ------- LOAD DATA --------

v  = pd.read_csv("data/v_pred.csv")    # v_pred (50)
vc = pd.read_csv("data/beta_pred.csv")  # (n_points, P), actual VC function values (50, 4), used by all patients
X_sub = pd.read_csv("data/Xsub.csv")   # (n_subjects, P) = (1000, 4)
rate = pd.read_csv("data/r_pred.csv")   # (n_points, n_subjects) = (50, 1000)
P = vc.shape[1]                         # number of variables
f = pd.read_csv("data/f_pred.csv")       # integrated rate (n_points of integration, n_subjects) = (250, 1000)
dage = pd.read_csv("data/dage.csv")      # disease age (250)

# --------- TRAIN_VAL_TEST_SPLIT -----------
n_subjects = X_sub.shape[0]
indices = np.arange(n_subjects)

train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42)

Xtrain = torch.tensor(X_sub.iloc[train_idx].to_numpy()).float()
Xval = torch.tensor(X_sub.iloc[val_idx].to_numpy()).float()
Xtest = torch.tensor(X_sub.iloc[test_idx].to_numpy()).float()

Ytrain = torch.tensor(f.iloc[:, train_idx].to_numpy()).float()  # [250, 600]
Yval = torch.tensor(f.iloc[:, val_idx].to_numpy()).float()      # [250, 200]
Ytest = torch.tensor(f.iloc[:, test_idx].to_numpy()).float()    # [250, 200]

# --------- DNN MODEL ------------
def VCNet(P, hidden=(64,64), dropout=0.2):
    """
    Estimates varying coefficient functions β_r(v)
    Input: v (1D scalar, normalized)
    Output: [β_0(v), β_1(v), ..., β_{P-1}(v)]
    """
    layers, in_dim = [], 1 # input is 1D: time variable
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = h
    layers += [nn.Linear(in_dim, P)]
    return nn.Sequential(*layers)

# --------- BIDIDRECTIONAL INTEGRATION ------
def integration(model, X, v, dage):
    """
    Integrate forward and backward from midpoint
    """
    n_subjects = X.shape[0]
    f0 = torch.zeros(n_subjects, device=X.device)
    mid = len(v) // 2
   
    # ODE function
    def ode_fn(v, f):
        v_norm = (v - v_mean) / v_std 
        vc = model(v_norm.view(1, 1))                   
        rate = torch.exp(vc @ X.T).squeeze(0) 
        return rate
    # at dage = 0, f_pred = 1
    # use quarter years step size
    
    # Forward integration
    sol_forward = odeint(ode_fn, f0, v[mid:], method='euler')
    # Backward integration
    sol_backward = odeint(ode_fn, f0, v[:mid+1].flip(0), method='euler').flip(0)
    # Combine backward and forward results
    f_v = torch.cat([sol_backward[:-1], sol_forward], dim=0)
    # Add batch dimentions for interpolation

    f_v = f_v.T.unsqueeze(0) # [1, n_subjects, len(v)]
    # Linear interpolate to disease age (make last dimension to size)
    f_pred = nn.functional.interpolate(f_v, size=len(dage), mode='linear', align_corners=True) #[1, n_subjects, len(dage)]
    return f_pred.squeeze(0).T #[len(dage), n_subjects]

# ------- CONVERT INPUT & OUTPUTS INTO TENSOR --------
v = torch.tensor(v.to_numpy()).float().flatten()
v_mean, v_std = v.mean(), v.std()
dage = torch.tensor(dage.to_numpy()).float().flatten()

# --------- INITIALIZE ----------
model = VCNet(P)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

# --------- TRAIN LOOP -----------
for epoch in range(2000):
    model.train()
    opt.zero_grad()
    
    fpred_train = integration(model, Xtrain, v, dage)
    loss = loss_fn(fpred_train, Ytrain)
    loss.backward()
    opt.step()

    if epoch % 200 == 0:
        model.eval()
        with torch.no_grad():
            fpred_val = integration(model, Xval, v, dage)
            val = loss_fn(fpred_val, Yval).item()
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
            fpred_test = integration(model, X, v ,dage)
            preds.append(fpred_test) 
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
    fpred_test = integration(model, Xtest, v, dage)  
    test_mse = loss_fn(fpred_test, Ytest).item()
    print(f"Test MSE: {test_mse}")

    # --- Evaluate & Plot Rate vs Value Curves ---

    plt.figure(figsize=(8,5))
    for i in range(Xtest.shape[0]):       # loop over subjects
        plt.plot(dage.numpy(), fpred_test[:, i].numpy())

    plt.xlabel("Disease Age")
    plt.ylabel("F_pred/Accumulated Rate")
    plt.title("Integrated Rate vs Value Curve per Subject")
    plt.legend()
    plt.show()

    # Plot Varying Coefficient Function
    for i in range(Xtest.shape[0]):
        plt.plot()