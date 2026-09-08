"""
Weight diagnostics for Method 2 (joint distribution modelling).

Computes effective sample size (ESS) for resampling weight vectors on a
representative random sample of target points, since computing this for
all 5,875 recordings exhaustively is not computationally feasible within
a reasonable runtime (see method2_5fold_eval.py's own runtime note).

ESS = (sum(w))^2 / sum(w^2)

Usage:
    python -m src.method2_weight_diagnostics
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

Y_COL = "motor_UPDRS"
N_SAMPLE_TARGETS = 300
RANDOM_SEED = 0


def silverman_bandwidth(x):
    n = len(x)
    std = np.std(x, ddof=1)
    return 1.06 * std * n ** (-1 / 5)


def fit_joint_and_marginal_kdes(df):
    joint_kdes, age_kdes = {}, {}
    for s in df["sex"].unique():
        subset = df[df["sex"] == s]
        joint_data = np.vstack([subset["age"].to_numpy(), subset[Y_COL].to_numpy()])
        joint_kdes[s] = gaussian_kde(joint_data)
        age_kdes[s] = gaussian_kde(subset["age"].to_numpy())
    return joint_kdes, age_kdes


def conditional_density_batch(y_val, ages, sexes, joint_kdes, age_kdes, floor=1e-6):
    p_hat = np.zeros(len(ages))
    for s in np.unique(sexes):
        mask = sexes == s
        ages_s = ages[mask]
        query_points = np.vstack([ages_s, np.full(len(ages_s), y_val)])
        joint_vals = joint_kdes[s].evaluate(query_points)
        age_vals = age_kdes[s].evaluate(ages_s)
        p_hat[mask] = joint_vals / np.clip(age_vals, floor, None)
    return np.clip(p_hat, floor, None)


def compute_ess(w):
    """Effective sample size of a (possibly unnormalised) weight vector."""
    w = w / w.sum()
    return (w.sum() ** 2) / np.sum(w ** 2)


def run_diagnostics(df, n_sample_targets=N_SAMPLE_TARGETS, seed=RANDOM_SEED):
    y = df[Y_COL].to_numpy()
    age = df["age"].to_numpy()
    sex = df["sex"].to_numpy()
    N = len(df)

    joint_kdes, age_kdes = fit_joint_and_marginal_kdes(df)
    h_y = silverman_bandwidth(y)

    rng = np.random.default_rng(seed)
    sample_targets = rng.choice(N, size=n_sample_targets, replace=False)

    all_ess = []
    for n in sample_targets:
        kernel_vals = np.exp(-0.5 * ((y - y[n]) / h_y) ** 2)
        p_hat = conditional_density_batch(y[n], age, sex, joint_kdes, age_kdes)
        w = kernel_vals / p_hat
        all_ess.append(compute_ess(w))

    return np.array(all_ess)


if __name__ == "__main__":
    df = pd.read_csv("data/raw/parkinsons_telemonitoring.csv")
    all_ess = run_diagnostics(df)

    results = pd.DataFrame({"ess": all_ess})
    results.to_csv("method2_weight_diagnostics.csv", index=False)

    n_candidates = len(df) - 1
    print(f"Sampled target points: {len(all_ess)} (out of N={len(df)})")
    print(f"Mean ESS: {all_ess.mean():.2f} (out of ~{n_candidates} candidates, "
          f"{100*all_ess.mean()/n_candidates:.3f}%)")
    print(f"Median ESS: {np.median(all_ess):.2f}")
    print(f"Min ESS: {all_ess.min():.2f} ({100*all_ess.min()/n_candidates:.4f}%)")
    print(f"Max ESS: {all_ess.max():.2f}")
    print(f"Fraction with ESS < 50: {(all_ess < 50).mean()*100:.1f}%")
    print(f"Fraction with ESS < 10: {(all_ess < 10).mean()*100:.1f}%")