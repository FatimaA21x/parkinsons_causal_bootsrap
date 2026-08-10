import time
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

FEATURE_COLS = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11",
    "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE",
]
Y_COL = "motor_UPDRS"


def fit_joint_and_marginal_kdes(train_df):
    joint_kdes, age_kdes = {}, {}
    for s in train_df["sex"].unique():
        subset = train_df[train_df["sex"] == s]
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


def run_5fold_benchmark(df, n_splits=5, seed=0):
    gkf = GroupKFold(n_splits=n_splits)
    preds_assoc, preds_causal, actuals = [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df["subject#"])):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        model_assoc = Ridge(alpha=1.0)
        model_assoc.fit(train_df[FEATURE_COLS], train_df[Y_COL])
        preds_assoc.extend(model_assoc.predict(test_df[FEATURE_COLS]))

        joint_kdes, age_kdes = fit_joint_and_marginal_kdes(train_df)
        boot_train = backdoor_bootstrap_joint(train_df, joint_kdes, age_kdes, seed=seed + fold)
        model_causal = Ridge(alpha=1.0)
        model_causal.fit(boot_train[FEATURE_COLS], boot_train[Y_COL])
        preds_causal.extend(model_causal.predict(test_df[FEATURE_COLS]))

        actuals.extend(test_df[Y_COL].values)
        print(f"  fold {fold+1}/{n_splits} done "
              f"(train={len(train_df)}, test={len(test_df)})")

    actuals = np.array(actuals)
    print("\nAssociational  MAE:", mean_absolute_error(actuals, preds_assoc),
          " R2:", r2_score(actuals, preds_assoc))
    print("Causal boot.   MAE:", mean_absolute_error(actuals, preds_causal),
          " R2:", r2_score(actuals, preds_causal))


if __name__ == "__main__":
    df = pd.read_csv("../data/raw/parkinsons_telemonitoring.csv")

    t0 = time.time()
    run_5fold_benchmark(df)
    print(f"\nElapsed: {time.time()-t0:.1f} seconds")