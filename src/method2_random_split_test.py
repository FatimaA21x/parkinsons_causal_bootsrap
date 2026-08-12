import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from method2_5fold_eval import fit_joint_and_marginal_kdes, backdoor_bootstrap_joint, FEATURE_COLS, Y_COL


def run_random_5fold_benchmark(df, n_splits=5, seed=0):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds_assoc, preds_causal, actuals = [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
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
        print(f"  fold {fold+1}/{n_splits} done (train={len(train_df)}, test={len(test_df)})")

    actuals = np.array(actuals)
    print("\nAssociational  MAE:", mean_absolute_error(actuals, preds_assoc), " R2:", r2_score(actuals, preds_assoc))
    print("Causal boot.   MAE:", mean_absolute_error(actuals, preds_causal), " R2:", r2_score(actuals, preds_causal))


if __name__ == "__main__":
    df = pd.read_csv("../data/raw/parkinsons_telemonitoring.csv")
    t0 = time.time()
    run_random_5fold_benchmark(df)
    print(f"Elapsed: {time.time()-t0:.1f}s")