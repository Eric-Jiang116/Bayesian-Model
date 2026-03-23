import torch
import torch.nn as nn

from data import load_data, split_and_normalize, to_tensors
from model import VCNet, rate_fn

# Load data
v, vc, Xsub, rate = load_data()
P = vc.shape[1]

Xsub_train, Xsub_val, Xsub_test, rate_train, rate_val, rate_test = split_and_normalize(Xsub, rate)
Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, v_tensor, v_np = to_tensors(
    Xsub_train, Xsub_val, Xsub_test, rate_train, rate_val, rate_test, v
)

def train(model, v, Xtrain, Ytrain, Xval, Yval, epochs=2000, lr=1e-3, weight_decay=1e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()

        vc_pred = model(v)
        rate_pred = rate_fn(vc_pred, Xtrain)
        loss = loss_fn(rate_pred, Ytrain)

        loss.backward()
        opt.step()

        if epoch % 200 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(rate_fn(model(v), Xval), Yval).item()
            print(f"Epoch {epoch:4d} | train {loss.item():.6f} | val {val_loss:.6f}")

    return model, opt


def save_checkpoint(model, opt, Xtest, Ytest, path="model_checkpoint.pt"):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "X_test": Xtest,
        "Y_test": Ytest,
    }, path)
    print(f"Checkpoint saved to {path}")

# Train and save
model = VCNet(P)
model, opt = train(model, v_tensor, Xtrain, Ytrain, Xval, Yval)
save_checkpoint(model, opt, Xtest, Ytest)