import pandas as pd
from sklearn.model_selection import train_test_split
import torch

VCData = pd.read_csv("vc_training_and_test_set.csv")

v = VCData["v_pred"].to_numpy(dtype='float32')  # shape (nv_pred,)
columns = VCData.shape[1]
vc = VCData.iloc[:, 1: columns-1].to_numpy(dtype='float32')   # shape (nv_pred, P)

v_mean, v_std = v.mean(), v.std()
v_norm = (v - v_mean) / (v_std + 1e-8)

# Split (train/test)
v_train, v_test, vc_train, vc_test = train_test_split(v_norm, vc, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train = torch.tensor(v_train)
Y_train = torch.tensor(vc_train)
X_test  = torch.tensor(v_test)
Y_test  = torch.tensor(vc_test)