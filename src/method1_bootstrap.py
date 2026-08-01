import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

FEATURE_COLS = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11",
    "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE",
]
Y_COL = "motor_UPDRS"


def silverman_bandwidth(x):
    n = len(x)
    std = np.std(x, ddof=1)
    return 1.06 * std * n ** (-1 / 5)


def fit_sex_stratified_kdes(train_df):
    """
    Fits one probability-density curve of UPDRS scores separately for
    each sex group. Gives p_hat(y | sex): how likely is this UPDRS value
    for this sex?
    """
    y = train_df[Y_COL].to_numpy()
    sex = train_df["sex"].to_numpy()

    kdes = {}
    for s in np.unique(sex):
        y_this_sex = y[sex == s]
        kdes[s] = gaussian_kde(y_this_sex)
    return kdes

def backdoor_bootstrap(train_df, kdes, seed=0):
    """
    Produces a deconfounded version of train_df: for each original
    subject, we look at their actual UPDRS score, then find a
    replacement set of voice features drawn from someone whose UPDRS
    was similar AND whose sex made that UPDRS value comparatively rare
    (rather than just anyone with a similar UPDRS score at random).
    """
    rng = np.random.default_rng(seed)
    y = train_df[Y_COL].to_numpy()
    sex = train_df["sex"].to_numpy()
    feats = train_df[FEATURE_COLS].to_numpy()
    N = len(train_df)

    h_y = silverman_bandwidth(y)
    resampled_feats = np.zeros_like(feats)

    for n in range(N):
        # how close is every OTHER subject's UPDRS to this subject's?
        kernel_vals = np.exp(-0.5 * ((y - y[n]) / h_y) ** 2)

        # how typical was subject n's UPDRS, given each candidate's own sex?
        p_hat = np.array([kdes[sex[i]].evaluate(y[n])[0] for i in range(N)])
        p_hat = np.clip(p_hat, 1e-6, None)

        weights = kernel_vals / p_hat
        weights = weights / weights.sum()

        chosen_i = rng.choice(N, p=weights)
        resampled_feats[n] = feats[chosen_i]

    out = train_df.copy()
    out[FEATURE_COLS] = resampled_feats
    return out

df = pd.read_csv("../data/subject_level.csv")
kdes = fit_sex_stratified_kdes(df)
print(kdes.keys())
print("How dense is UPDRS=20 for males?", kdes[0].evaluate(20))
print("How dense is UPDRS=20 for females?", kdes[1].evaluate(20))
boot_df = backdoor_bootstrap(df, kdes, seed=0)
print(boot_df[["subject#", "sex", "motor_UPDRS"]].head())
print(boot_df[FEATURE_COLS].head())



def run_loo_benchmark(df, seed=0):
    N = len(df)
    preds_assoc, preds_causal, actuals = [], [], []

    for held_out_idx in range(N):
        train_df = df.drop(df.index[held_out_idx]).reset_index(drop=True)
        test_row = df.iloc[[held_out_idx]]

        # associational: train directly on the raw training data
        model_assoc = Ridge(alpha=1.0)
        model_assoc.fit(train_df[FEATURE_COLS], train_df[Y_COL])
        preds_assoc.append(model_assoc.predict(test_row[FEATURE_COLS])[0])

        # causal: fit the KDEs and bootstrap ONLY from the 41 training subjects
        kdes = fit_sex_stratified_kdes(train_df)
        boot_train = backdoor_bootstrap(train_df, kdes, seed=seed + held_out_idx)
        model_causal = Ridge(alpha=1.0)
        model_causal.fit(boot_train[FEATURE_COLS], boot_train[Y_COL])
        preds_causal.append(model_causal.predict(test_row[FEATURE_COLS])[0])

        actuals.append(test_row[Y_COL].values[0])

    actuals = np.array(actuals)
    print("Associational  MAE:", mean_absolute_error(actuals, preds_assoc),
          " R2:", r2_score(actuals, preds_assoc))
    print("Causal boot.   MAE:", mean_absolute_error(actuals, preds_causal),
          " R2:", r2_score(actuals, preds_causal))

run_loo_benchmark(df)