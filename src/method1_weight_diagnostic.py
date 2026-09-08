"""
Weight diagnostics for Method 1 (subject-level aggregation).

Computes effective sample size (ESS) for every resampling weight vector
generated during the leave-one-out evaluation, to verify the KDE-based
reweighting concentrates weight sensibly rather than degenerating onto
a single or handful of candidates.

ESS = (sum(w))^2 / sum(w^2)

An ESS close to N means weights are close to uniform (no real reweighting
happening); an ESS close to 1 means weight has collapsed almost entirely
onto one candidate (a real problem for the resampling step). This script
reports both, across every weight vector actually used during Method 1's
evaluation, on the real dataset.

Usage:
    python -m src.method1_weight_diagnostics
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

Y_COL = "motor_UPDRS"


def silverman_bandwidth(x):
    n = len(x)
    std = np.std(x, ddof=1)
    return 1.06 * std * n ** (-1 / 5)


def fit_sex_stratified_kdes(train_df):
    y = train_df[Y_COL].to_numpy()
    sex = train_df["sex"].to_numpy()
    return {s: gaussian_kde(y[sex == s]) for s in np.unique(sex)}


def compute_ess(w):
    """Effective sample size of a (possibly unnormalised) weight vector."""
    w = w / w.sum()
    return (w.sum() ** 2) / np.sum(w ** 2)


def run_diagnostics(subject_df):
    """Recomputes every weight vector Method 1 actually uses during its
    leave-one-out evaluation, and returns their ESS values."""
    N = len(subject_df)
    all_ess = []

    for held_out_idx in range(N):
        train_df = subject_df.drop(subject_df.index[held_out_idx]).reset_index(drop=True)
        kdes = fit_sex_stratified_kdes(train_df)
        y = train_df[Y_COL].to_numpy()
        sex = train_df["sex"].to_numpy()
        Ntr = len(train_df)
        h_y = silverman_bandwidth(y)

        for n in range(Ntr):
            kernel_vals = np.exp(-0.5 * ((y - y[n]) / h_y) ** 2)
            p_hat = np.array([kdes[sex[i]].evaluate(y[n])[0] for i in range(Ntr)])
            p_hat = np.clip(p_hat, 1e-6, None)
            w = kernel_vals / p_hat
            all_ess.append(compute_ess(w))

    return np.array(all_ess)


if __name__ == "__main__":
    subject_df = pd.read_csv("data/subject_level.csv")
    all_ess = run_diagnostics(subject_df)

    results = pd.DataFrame({"ess": all_ess})
    results.to_csv("method1_weight_diagnostics.csv", index=False)

    n_candidates = len(subject_df) - 1
    print(f"Total weight vectors examined: {len(all_ess)}")
    print(f"Mean ESS: {all_ess.mean():.2f} (out of ~{n_candidates} candidates, "
          f"{100*all_ess.mean()/n_candidates:.1f}%)")
    print(f"Median ESS: {np.median(all_ess):.2f}")
    print(f"Min ESS: {all_ess.min():.2f} ({100*all_ess.min()/n_candidates:.1f}%)")
    print(f"Max ESS: {all_ess.max():.2f}")
    print(f"Fraction with ESS < 10: {(all_ess < 10).mean()*100:.1f}%")
    print(f"Fraction with ESS < 5: {(all_ess < 5).mean()*100:.1f}%")