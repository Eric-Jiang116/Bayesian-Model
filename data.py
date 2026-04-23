"""
Load, split, and normalize all datasets.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ──────────────────────────────────────────────
# Raw loading
# ──────────────────────────────────────────────

def load_raw(data_dir: str = "data"):
    """Load all CSV files and return a dict of DataFrames."""
    return {
        "X_sub":    pd.read_csv(f"{data_dir}/Xsub.csv"),       # [S, P]
        "f_pred":   pd.read_csv(f"{data_dir}/f_pred.csv"),     # [T, S]
        "dage":     pd.read_csv(f"{data_dir}/dage.csv"),       # [T, 1]
        "v_pred":   pd.read_csv(f"{data_dir}/v_pred.csv"),     # [K, 1]
        "beta_pred":pd.read_csv(f"{data_dir}/beta_pred.csv"),  # [K, P]
        "r_pred":   pd.read_csv(f"{data_dir}/r_pred.csv"),     # [K, S]
    }


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class VCDataset:
    """
    Holds all tensors and split indices derived from the raw CSVs.

    Attributes
    ----------
    P            : int          number of covariates
    n_subjects   : int
    T            : int          number of time-points
    K            : int          number of v-grid points

    t_grid       : [T]   float  disease-age grid
    v_grid       : [K]   float  v-domain grid
    beta_pred    : [K,P] float  ground-truth varying coefficients
    r_pred_np    : [K,S] np     ground-truth rate curves
    X_all        : [S,P] float  all subject covariates
    f_np         : [T,S] np     all outcome trajectories

    X_train/val/test : [S_split, P]
    Y_train/val/test : [T, S_split]
    idx_train/val/test : 1-D int arrays of subject indices
    """

    def __init__(self, data_dir: str = "data"):
        raw = load_raw(data_dir)

        # ── numpy / tensor conversion ──────────────────
        v_np            = raw["v_pred"].to_numpy().reshape(-1).astype(np.float32)
        dage_np         = raw["dage"].to_numpy().reshape(-1).astype(np.float32)
        f_np            = raw["f_pred"].to_numpy().astype(np.float32)
        beta_pred_np    = raw["beta_pred"].to_numpy().astype(np.float32)
        r_pred_np       = raw["r_pred"].to_numpy().astype(np.float32)
        X_np            = raw["X_sub"].to_numpy().astype(np.float32)

        self.P          = X_np.shape[1]
        self.n_subjects = X_np.shape[0]
        self.T          = len(dage_np)
        self.K          = len(v_np)

        assert f_np.shape == (self.T, self.n_subjects), (
            f"f_pred shape mismatch: expected {(self.T, self.n_subjects)}, got {f_np.shape}"
        )
        assert beta_pred_np.shape == (self.K, self.P), (
            f"beta_pred shape mismatch: expected {(self.K, self.P)}, got {beta_pred_np.shape}"
        )
    

        self.t_grid     = torch.tensor(dage_np, dtype=torch.float32)
        self.v_grid     = torch.tensor(v_np, dtype=torch.float32)
        self.beta_pred  = torch.tensor(beta_pred_np, dtype=torch.float32)
        self.r_pred_np  = r_pred_np
        self.f_np       = f_np
        self.X_all      = torch.tensor(X_np, dtype=torch.float32)
        self.v_np       = v_np
        # ── normalization statistics fit on Xsub (need to add) ───
        self.v_mean = float(v_np.mean())
        self.v_std  = float(v_np.std() + 1e-8)
 
        # ── splits (60/20/20) ─────────────────────────────────────
        # f_pred_T = self.f_np.T # transpose f_pred to get [S, T]

        # X_train_full, X_test, f_train_full, f_test = train_test_split(
        #     self.X_all, f_pred_T, test_size=0.2, random_state=SEED, shuffle=True
        # )

        # X_train, X_val, f_train, f_val = train_test_split(
        #     X_train_full, f_train_full, test_size=0.25, random_state=SEED, shuffle=True
        # )

        # # Transpose back to [T, S] for your ODE solver
        # self.X_train, self.Y_train = X_train, f_train.T
        # self.X_val,   self.Y_val   = X_val,   f_val.T
        # self.X_test,  self.Y_test  = X_test,  f_test.T

        # print(self.X_train.shape, self.Y_train.shape)

        # print(self.X_test.shape, self.Y_test.shape)
        
        all_idx = np.arange(self.n_subjects)

        idx_train_full, idx_test = train_test_split(
            all_idx, test_size=0.2, random_state=SEED, shuffle=True
        )
        idx_train, idx_val = train_test_split(
            idx_train_full, test_size=0.25, random_state=SEED, shuffle=True
        )

        self.idx_train = idx_train
        self.idx_val   = idx_val
        self.idx_test  = idx_test

        self.X_train, self.Y_train = self._slice(idx_train)
        self.X_val,   self.Y_val   = self._slice(idx_val)
        self.X_test,  self.Y_test  = self._slice(idx_test)

    # ── helpers ────────────────────────────────────────

    def _slice(self, indices):
        x = self.X_all[indices, :]
        y = torch.tensor(self.f_np[:, indices], dtype=torch.float32)
        return x, y

    def norm_v(self, v: torch.Tensor) -> torch.Tensor:
        """Normalise a v (or y) tensor using v_grid statistics."""
        return (v - self.v_mean) / self.v_std

    def onset_index(self) -> int:
        """Return the index in t_grid closest to disease onset (t=0)."""
        return int(torch.argmin(torch.abs(self.t_grid - 0.0)).item())
    
dataset = VCDataset(data_dir="data")