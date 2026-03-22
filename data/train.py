# train
import torch
import torch.nn as nn
from model import VCNet

model = VCNet(P)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()
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
