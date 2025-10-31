import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

VCData = pd.read_csv("vc_training_and_test_set.csv")

v = VCData["v_pred"].to_numpy(dtype='float32').reshape(-1, 1) # shape (nv_pred,)
columns = VCData.shape[1]
vc = VCData.iloc[:, 1:].to_numpy(dtype='float32')   # shape (nv_pred, P)

v_mean, v_std = v.mean(), v.std()
v_norm = (v - v_mean) / (v_std + 1e-8)

# Split (train/test)
v_train, v_test, vc_train, vc_test = train_test_split(v_norm, vc, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train = torch.tensor(v_train)
Y_train = torch.tensor(vc_train)
X_test  = torch.tensor(v_test)
Y_test  = torch.tensor(vc_test)

P = vc.shape[1]   # number of VC functions

model = nn.Sequential(
    nn.Linear(1, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, P)
)

assert X_train.shape[1] == model[0].in_features == 1
assert Y_train.shape[1] == model[-1].out_features == P
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(2000):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    loss = loss_fn(pred, Y_train)
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_test), Y_test).item()
        print(f"Epoch {epoch:4d} | train {loss.item():.6f} | val {val_loss:.6f}")

with torch.no_grad():
    v_grid = torch.linspace(v_norm.min(), v_norm.max(), 200).reshape(-1,1)
    vc_pred = model(v_grid).numpy()

print(v_grid.shape, vc_pred.shape)
# Convert normalized v back to original
v_grid_orig = v_grid.numpy() * v_std + v_mean
print(P)
plt.figure(figsize=(8,5))
for j in range(P):
    plt.plot(v, vc[:, j], 'o', label=f'True f{j+1}', alpha=0.4)
    plt.plot(v_grid_orig, vc_pred[:, j], '-', label=f'Pred f{j+1}')
plt.xlabel('v')
plt.ylabel('VC function values')
plt.legend()
plt.show()