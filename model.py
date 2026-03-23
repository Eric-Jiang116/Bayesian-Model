import torch
import torch.nn as nn


def VCNet(P, hidden=(128, 256), dropout=0.2):
    layers, in_dim = [], 1  # input is 1D: time variable
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = h
    layers += [nn.Linear(in_dim, P)]
    return nn.Sequential(*layers)


def rate_fn(vc, X_sub):
    """
    vc:    [N, P]
    X_sub: [n_subjects, P]
    returns rate: [N, n_subjects]
    """
    return torch.exp(vc @ X_sub.T)