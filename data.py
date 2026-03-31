import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    v = pd.read_csv("data/v_pred.csv")         # (N, 1)
    vc = pd.read_csv("data/beta_pred.csv")     # (N, P)
    Xsub = pd.read_csv("data/Xsub.csv")        # (n_subjects, P)
    rate = pd.read_csv("data/r_pred.csv")      # (N, n_subjects)
    return v, vc, Xsub, rate


def split_and_normalize(Xsub, rate):
    # 60 / 20 / 20 split
    Xsub_train, Xsub_test, rate_train, rate_test = train_test_split(
        Xsub, rate.T, test_size=0.2, random_state=42
    )
    Xsub_train, Xsub_val, rate_train, rate_val = train_test_split(
        Xsub_train, rate_train, test_size=0.25, random_state=42
    )
    # Normalize Xsub, specifically the continuous covariate V4, leave V1/V2/V3 alone
    
    # Apply to all splits
    # Xsub_train = (Xsub_train - train_mean) / train_std
    # Xsub_val   = (Xsub_val   - train_mean) / train_std
    # Xsub_test  = (Xsub_test  - train_mean) / train_std
    # # Normalize using train stats only
    # Xsub_train = scaler.fit_transform(Xsub_train)
    # Xsub_val   = scaler.transform(Xsub_val)
    # Xsub_test  = scaler.transform(Xsub_test)

    return Xsub_train, Xsub_val, Xsub_test, rate_train, rate_val, rate_test

# Convert data to tensors
def to_tensors(Xsub_train, Xsub_val, Xsub_test, rate_train, rate_val, rate_test, v):
    Xtrain = torch.tensor(Xsub_train.to_numpy()).float()
    Xval = torch.tensor(Xsub_val.to_numpy()).float()
    Xtest = torch.tensor(Xsub_test.to_numpy()).float()

    Ytrain = torch.tensor(rate_train.to_numpy()).float().T
    Yval = torch.tensor(rate_val.to_numpy()).float().T
    Ytest = torch.tensor(rate_test.to_numpy()).float().T

    v_tensor = torch.tensor(v.to_numpy()).float()
    v_np = v.to_numpy().reshape(-1).astype(np.float32)

    return Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, v_tensor, v_np