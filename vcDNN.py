from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from torchdiffeq import odeint
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

v  = pd.read_csv("v_pred.csv")    # v_pred
vc = pd.read_csv("beta_pred.csv")  # (N, P), actual VC function values
X_sub = pd.read_csv("X_sub.csv")   # (n_subjects, P)
rate = pd.read_csv("r_pred.csv")   # (N, n_subjects)
P = vc.shape[1]

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
    returns rate: [N, n_subjects] returns 10 x 100 matrix
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
    rate_pred = rate_fn(vc_train_pred, X_sub) # [N_train, n_subjects]
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
# MC dropout: evaluate on test set
for m in model.modules():
    if isinstance(m, nn.Dropout):
        m.train()  # keep dropout on

def mc_dropout_predict(model, X, n_samples=1000):
    preds = []
    for i in range(n_samples):
        with torch.no_grad():
            vc_pred = model(X)
            rate_pred = rate_fn(vc_pred, X_sub)
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

# Align order of values
# order = np.argsort(v_test.to_numpy()[:,0]) # indices to sort test set, so that v and rate have the same index order for plotting
# v_sorted  = v_test.to_numpy()[order, 0]
# rate_sorted = rate_test.to_numpy()[order]
# mean_sorted = mean_pred.cpu().numpy()[order]
# lower_sorted = lower.cpu().numpy()[order]
# upper_sorted = upper.cpu().numpy()[order]

# ----- Integration ------
model.eval() # disable dropout

# Plot our rate functions for 5 subjects
with torch.no_grad():
    vc_grid = model(torch.from_numpy(norm(v_grid)))
    # Get rates for ALL subjects at once
    rate_pred_all = rate_fn(vc_grid, X_sub).numpy() # Shape: [200, n_subjects]

subjects = [0, 14, 23, 67, 89]

plt.figure(figsize=(8, 5))
for s in subjects:
    plt.plot(v_grid[:, 0], rate_pred_all[:, s], lw=2, label=f"Subject {s}")

plt.xlabel("v")
plt.ylabel("Predicted Rate (dy/dv)")
plt.title("Neural Network Output: Rate Functions Input to ODE Solver")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

class RateODEFunc(nn.Module):
    '''
    Define an ODE function that calls our NN to compute exponential rate functions
    '''
    def __init__(self, model, X_sub):
        super().__init__()
        self.model = model
        self.X_sub = X_sub
    def forward(self, v, y):
        '''
        v: scalar tensor
        y: amyloid accumulation [n_subjects]
        returns dy/dv
        '''
        v_in = v.view(1,1)                   # shape [1, 1]
        vc = self.model(norm(v_in))          # [1, P]
        rate = torch.exp(vc @ self.X_sub.T)  # [1, n_subjects]
        return rate.squeeze(0)               # [n_subjects]
    
t = torch.linspace(vmin, vmax, 200)    # update time
y0 = torch.zeros(X_sub.shape[0])       # initial y/accumulation values
ode_func = RateODEFunc(model, X_sub)
step_size = (vmax - vmin) / 1000     # small step size for fixed-step solvers

# --- 1. RUN STANDARD SOLUTIONS (For Trajectory Plots) ---
# Adaptive Solvers
ref = odeint(ode_func, y0, t, method="dopri5", rtol=1e-9, atol=1e-11) # default (our standard reference)
sol_dopri8 = odeint(ode_func, y0, t, method="dopri8",rtol=1e-9, atol=1e-11)
sol_bosh3 = odeint(ode_func, y0, t, method="bosh3",rtol=1e-9, atol=1e-11)

# Fixed-Step Solvers
sol_euler = odeint(ode_func, y0, t, method="euler", options=dict(step_size = step_size))
sol_rk4 = odeint(ode_func, y0, t, method="rk4", options=dict(step_size = step_size))
sol_explicitAdams = sol_rk4 = odeint(ode_func, y0, t, method="explicit_adams", options=dict(step_size = step_size))

solutions = {
    "euler": sol_euler,
    "rk4": sol_rk4,
    "expAdams": sol_explicitAdams,
    "dopri8": sol_dopri8,
    "bosh3": sol_bosh3,
    "dopri5": ref,
}

# Compute MSE over trajectory
print("MSE over trajectory")
for name, sol in solutions.items():
    err = torch.mean((sol - ref)**2).item()
    print(f"{name:7s} | MSE vs dopri5 = {err:.3e}")

# --- 2. TRAJECTORY & RESIDUAL PLOTS ---
print("Plot Solvers for Subject s")
styles = {
    "euler":    {"color": "blue",   "linestyle": "-",  "lw": 2, "label": "Euler"},
    "rk4":      {"color": "orange", "linestyle": "-",  "lw": 5, "label": "RK4 (Thick Line)"}, # Make this THICK
    "expAdams": {"color": "green",  "linestyle": "--", "lw": 2, "label": "ExpAdams"},         # Dashed
    "dopri8":   {"color": "red",    "linestyle": ":",  "lw": 2, "label": "Dopri8"},          # Dotted
    "bosh3":    {"color": "purple", "linestyle": "--", "lw": 2, "label": "bosh3"},
    "dopri5":   {"color": "black",  "linestyle": "-",  "lw":1,  "label": "Dopri5", "alpha": 0.5}
}
plt.figure(figsize=(8,5))
for s in subjects:
    for name, sol in solutions.items():
        plt.plot(t.numpy(), sol[:, s].detach().numpy(), **styles[name]) # use ** to unpack styles dict
    plt.xlabel("v")
    plt.ylabel("Integrated rate")
    plt.title(f"Solver comparison – subject {s}")
    plt.legend()
    plt.show()

print("Plot Residuals relative to Dopri5")
for s in subjects: 
    ref_traj = solutions['dopri5'][:, s].detach()
    for name, sol in solutions.items():
        if name == 'dopri5': continue # Don't plot ref vs ref
        
        traj = sol[:, s].detach()
        error = torch.abs(traj - ref_traj) + 1e-10 # add epsilon for log scale
        
        plt.semilogy(t.numpy(), error.numpy(), **styles[name]) # plots Y axis in power of 10

    plt.title("Integration Error vs Reference (Log Scale)")
    plt.xlabel("v")
    plt.ylabel("Absolute Error (|y - y_ref|)")
    plt.legend()
    plt.show()


print("\n--- Starting Efficiency Race (Time vs Error) ---")
# Setup: We pick one "hard" subject for the race
race_subject = 0 # Subject 0 usually has the steepest curve
y0 = torch.zeros(X_sub.shape[0])

# 1. Get "True" final value (High precision Dopri5)
with torch.no_grad():
    truth_sol = odeint(ode_func, y0, t, method="dopri5", rtol=1e-12, atol=1e-14)
    true_final = truth_sol[-1, race_subject].item()

# Define the racers
# For fixed-step solvers, we vary steps. For others (Adaptive), we vary tolerance.
racers = {
    # Fixed Step Group
    "euler":          {"type": "fixed",    "color": "blue",   "marker": "v"},
    "rk4":            {"type": "fixed",    "color": "orange", "marker": "s"},
    "explicit_adams": {"type": "fixed",    "color": "green",  "marker": "x"}, # Moved to Fixed!
    
    # Adaptive Group
    "dopri5":         {"type": "adaptive", "color": "black",  "marker": "o"},
    "dopri8":         {"type": "adaptive", "color": "red",    "marker": "^"},
    "bosh3":          {"type": "adaptive", "color": "purple", "marker": "D"},
}

race_results = {name: {"time": [], "error": []} for name in racers}

# 3. Race Settings
steps_list = [10, 50, 100, 200, 400, 800, 1600, 3200]
tols_list  = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

# 4. Run The Race
with torch.no_grad():
    for name, cfg in racers.items():
        print(f"Racing {name}...")
        
        # STRATEGY A: Fixed Step (Vary Step Size)
        if cfg["type"] == "fixed":
            for steps in steps_list:
                # Force the step size manually
                t_race = torch.linspace(vmin, vmax, steps)
                dt = (vmax - vmin) / (steps - 1)
                
                start = time.perf_counter()
                sol = odeint(ode_func, y0, t_race, method=name, options=dict(step_size=dt))
                end = time.perf_counter()
                
                err = abs(sol[-1, race_subject].item() - true_final)
                race_results[name]["time"].append(end - start)
                race_results[name]["error"].append(err)
        
        # STRATEGY B: Adaptive (Vary Tolerance)
        else:
            for tol in tols_list:
                start = time.perf_counter()
                sol = odeint(ode_func, y0, t, method=name, rtol=tol, atol=tol/10)
                end = time.perf_counter()
                
                err = abs(sol[-1, race_subject].item() - true_final)
                race_results[name]["time"].append(end - start)
                race_results[name]["error"].append(err)

# 5. Plot the Race Results
plt.figure(figsize=(12, 9))

for name, res in race_results.items():
    cfg = racers[name]
    sorted_data = sorted(zip(res["time"], res["error"]))
    times, errors = zip(*sorted_data)
    
    label_str = f"{name} (Fixed Steps)" if cfg["type"] == "fixed" else f"{name} (Adaptive Tol)"
    
    plt.loglog(times, errors, label=label_str, 
               color=cfg["color"], marker=cfg["marker"], 
               linestyle='-', linewidth=2, markersize=8, alpha=0.8)

plt.xlabel("Execution Time (seconds) [Log Scale]")
plt.ylabel("Absolute Error [Log Scale]")
plt.title(f"Efficiency Benchmark (Adams as Fixed-Step)", fontsize=14)
plt.grid(True, which="both", alpha=0.3)
plt.legend()

# Add arrow to show best corner
plt.annotate('Better (Fast & Accurate)', xy=(0.02, 0.05), xycoords='axes fraction', 
             xytext=(0.15, 0.15), arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()