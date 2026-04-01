"""
VCNet architecture and ODE-based trajectory prediction.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint

# ──────────────────────────────────────────────
# Network
# ──────────────────────────────────────────────

class VCNet(nn.Module):
    """
    Varying-Coefficient Network.

    Maps a scalar v to a P-dimensional coefficient vector β(v).
    The final linear layer is zero-initialised so the network starts close to
    the trivial solution and training is stable from epoch 0.

    Parameters
    ----------
    P       : number of covariates / output coefficients
    hidden  : tuple of hidden-layer widths
    dropout : dropout probability (used for MC-dropout uncertainty)
    """

    def __init__(self, P: int, hidden: tuple = (64, 64), dropout: float = 0.2):
        super().__init__()

        layers = []
        in_dim = 1
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h

        last = nn.Linear(in_dim, P, bias=True)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers.append(last)

        self.net = nn.Sequential(*layers)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        v : [batch, 1]  scalar input

        Returns
        -------
        beta   : [batch, P]
        """
        return self.net(v)

def get_dropout(x, p=0.5, mc=False):
    return nn.Dropout(p)(x) if mc else x

# ──────────────────────────────────────────────
# ODE wrapper
# ──────────────────────────────────────────────
def build_rhs(model, X, norm_v_fn, sign=1.0):
    def rhs(t, y):
        y_norm = norm_v_fn(y).view(-1, 1)
        beta   = model(y_norm)
        rate   = torch.exp((beta * X).sum(dim=1))
        return sign * rate
    return rhs

def build_rhs(model: VCNet, X: torch.Tensor, norm_v_fn, sign=1.0):
    """
    Return the ODE dy/dt = exp(β(y) · x)  for all subjects.

    Parameters
    ----------
    model     : trained / partially-trained VCNet
    X         : [S, P] subject covariates
    """
    def rhs(t, y):
        # y : [S]  current trajectory values for every subject
        y_norm = norm_v_fn(y).view(-1, 1)   # [S, 1]
        beta   = model(y_norm)              # [S, P]
        rate   = torch.exp((beta * X).sum(dim=1))  # [S]
        return sign * rate
    return rhs

def predict_trajectories(
    model: VCNet,
    t_grid: torch.Tensor,
    X: torch.Tensor,
    norm_v_fn,
    i0: int,
    method: str = "rk4",
    step_size: float = 0.5,
) -> torch.Tensor:
    """
    Integrate the ODE both forward and backward from the onset index i0.

    Parameters
    ----------
    model      : VCNet
    t_grid     : [T]    disease-age grid
    X          : [S, P] subject covariates
    norm_v_fn  : normalisation callable
    i0         : onset index (where t ≈ 0)
    method     : torchdiffeq solver name
    step_size  : fixed step size for explicit solvers

    Returns
    -------
    F : [T, S]  predicted trajectories
    """
    # in predict_trajectories:
    rhs_fwd  = build_rhs(model, X, norm_v_fn, sign=+1.0)
    #rhs_bwd  = build_rhs(model, X, norm_v_fn, sign=-1.0)

    y0   = torch.ones(X.shape[0], dtype=torch.float32)
    opts = {"step_size": step_size}

    forward  = odeint(rhs_fwd, y0, t_grid[i0:], method=method, options=opts)             # [T-i0, S]
    backward = odeint(rhs_fwd, y0, t_grid[:i0+1].flip(0), method=method, options=opts)   # [i0+1, S]
    backward = backward.flip(0)

    F = torch.cat([backward[:-1], forward], dim=0)  # [T, S]
    return F