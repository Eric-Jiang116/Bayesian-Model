import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

VCData = pd.read_csv("vc_training_and_test_set.csv")

# Extract v_pred and vc from the dataset
v = VCData["v_pred"].to_numpy(dtype='float32').reshape(-1, 1) # shape (nv_pred,)
vc = VCData.iloc[:, 1:].to_numpy(dtype='float32')   # shape (nv_pred, P)

# Normalize v for training stability
v_mean, v_std = v.mean(), v.std()
v_norm = (v - v_mean) / (v_std + 1e-8)

# Split (train/validation/test) 60, 20, 20
v_train, v_test, vc_train, vc_test = train_test_split(v_norm, vc, test_size=0.2, random_state=42)
v_train, v_val, vc_train, vc_val = train_test_split(v_train, vc_train, test_size=0.25, random_state=42)

# Convert to torch tensors
X_train = torch.tensor(v_train, dtype=torch.float32)
Y_train = torch.tensor(vc_train, dtype=torch.float32)
X_val = torch.tensor(v_val, dtype=torch.float32)
Y_val = torch.tensor(vc_val, dtype=torch.float32)
X_test  = torch.tensor(v_test, dtype=torch.float32)
Y_test  = torch.tensor(vc_test, dtype=torch.float32)

P = vc.shape[1]   # number of VC functions

def VCNet(P, hidden_dim, dropout):
    layers = []
    in_dim = 1
    for h in hidden_dim:
        #start with non linear activation function
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        in_dim = h
    layers.append(nn.Linear(in_dim, P)) # last layer is linear to output 
    return nn.Sequential(*layers)

def train(model, epochs, optimizer, X_train, Y_train, X_val, Y_val, loss_fn, device="cpu"):
    model.to(device)
    for epoch in range(epochs):
        model.train() # look into dropout 
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, Y_train)
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_val), Y_val).item()
            print(f"Epoch {epoch:4d} | train {loss.item():.6f} | val {val_loss:.6f}")


def plot_predictions(v_grid_orig, vc_pred, v, vc):
    plt.figure(figsize=(8,5))
    for j in range(P):
        plt.plot(v, vc[:, j], 'o', label=f'True f{j+1}', alpha=0.4)
        plt.plot(v_grid_orig , vc_pred[:, j], '-', label=f'Pred f{j+1}')
    plt.xlabel('v')
    plt.ylabel('VC function values')
    plt.legend()
    plt.show()

#Create model, optimizer, loss function
hidden_dim = [64, 64]
model = VCNet(P, hidden_dim, 0.2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) # regularization
loss_fn = nn.MSELoss()

# Check input and output dimensions
#assert X_train.shape[1] == model[0].in_features == 1
#assert Y_train.shape[1] == model[-1].out_features == P

# Train model
train(model, 2000, optimizer, X_train, Y_train, X_val, Y_val, loss_fn) #epoch = 2000

# Predict on X_test
model.train()
with torch.no_grad():
    #v_grid = torch.linspace(v_norm.min(), v_norm.max(), 200).reshape(-1,1)
    vc_pred = model(X_test)
    test_loss = loss_fn(vc_pred, Y_test).item()
print(f"Test Loss: {test_loss:.6f}")
print(v_test.shape, vc_pred.shape)

# Convert numpy for plotting
vc_pred = vc_pred.numpy()
v_test_orig = v_test * v_std + v_mean

# Plot predictions
plot_predictions(v_test_orig, vc_pred, v_test_orig, vc_test)