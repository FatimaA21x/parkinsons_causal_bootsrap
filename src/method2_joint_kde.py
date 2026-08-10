import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

FEATURE_COLS = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11",
    "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE",
]
Y_COL = "motor_UPDRS"


def fit_joint_and_marginal_kdes(train_df):
    """
    For each sex group, fits two things:
    1. A joint density over (age, UPDRS) together -- captures how age
       and severity co-occur, with age treated as a soft, continuous
       quantity (biological age uncertainty) rather than a fixed point.
    2. A marginal density over age alone -- needed to convert the joint
       density into a proper conditional p_hat(y | age, sex) in the next
       step.
    """
    joint_kdes = {}
    age_kdes = {}

    for s in train_df["sex"].unique():
        subset = train_df[train_df["sex"] == s]
        joint_data = np.vstack([subset["age"].to_numpy(), subset[Y_COL].to_numpy()])
        joint_kdes[s] = gaussian_kde(joint_data)
        age_kdes[s] = gaussian_kde(subset["age"].to_numpy())

    return joint_kdes, age_kdes

df = pd.read_csv("../data/raw/parkinsons_telemonitoring.csv")
joint_kdes, age_kdes = fit_joint_and_marginal_kdes(df)
print(joint_kdes.keys())
print("Joint density at age=65, UPDRS=20, male:", joint_kdes[0].evaluate([65, 20]))
print("Marginal age density at 65, male:", age_kdes[0].evaluate([65]))
def conditional_density_batch(y_val, ages, sexes, joint_kdes, age_kdes, floor=1e-6):
    """
    Vectorized version: computes p_hat(y_val | age_i, sex_i) for an
    entire array of candidates at once, instead of one at a time. Same
    maths as before, just batched so scipy can evaluate many points in
    one call rather than paying Python overhead millions of times.
    """
    p_hat = np.zeros(len(ages))
    for s in np.unique(sexes):
        mask = sexes == s
        ages_s = ages[mask]
        query_points = np.vstack([ages_s, np.full(len(ages_s), y_val)])
        joint_vals = joint_kdes[s].evaluate(query_points)
        age_vals = age_kdes[s].evaluate(ages_s)
        p_hat[mask] = joint_vals / np.clip(age_vals, floor, None)
    return np.clip(p_hat, floor, None)

p = conditional_density_batch(20, 65, 0, joint_kdes, age_kdes)
print("p_hat(UPDRS=20 | age=65, male):", p)

def silverman_bandwidth(x):
    n = len(x)
    std = np.std(x, ddof=1)
    return 1.06 * std * n ** (-1 / 5)

def backdoor_bootstrap_joint(train_df, joint_kdes, age_kdes, seed=0):
    rng = np.random.default_rng(seed)
    y = train_df[Y_COL].to_numpy()
    age = train_df["age"].to_numpy()
    sex = train_df["sex"].to_numpy()
    feats = train_df[FEATURE_COLS].to_numpy()
    N = len(train_df)

    h_y = silverman_bandwidth(y)
    resampled_feats = np.zeros_like(feats)

    for n in range(N):
        kernel_vals = np.exp(-0.5 * ((y - y[n]) / h_y) ** 2)
        p_hat = conditional_density_batch(y[n], age, sex, joint_kdes, age_kdes)

        weights = kernel_vals / p_hat
        weights = weights / weights.sum()

        chosen_i = rng.choice(N, p=weights)
        resampled_feats[n] = feats[chosen_i]

    out = train_df.copy()
    out[FEATURE_COLS] = resampled_feats
    return out

df = pd.read_csv("../data/raw/parkinsons_telemonitoring.csv")
small_df = df.sample(n=200, random_state=0).reset_index(drop=True)

joint_kdes, age_kdes = fit_joint_and_marginal_kdes(small_df)
boot_small = backdoor_bootstrap_joint(small_df, joint_kdes, age_kdes, seed=0)
print(boot_small[["age", "sex", "motor_UPDRS"]].head())
print(boot_small[FEATURE_COLS].head())