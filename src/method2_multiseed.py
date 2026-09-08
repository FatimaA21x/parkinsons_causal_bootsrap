import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from src.method2_5fold_eval import fit_joint_and_marginal_kdes, backdoor_bootstrap_joint, FEATURE_COLS, Y_COL


def run_multiseed(df, n_seeds=10, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(df, groups=df["subject#"]))

    # Associational model is deterministic -- compute it once, not per seed
    preds_assoc, actuals = [], []
    for train_idx, test_idx in splits:
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        model = Ridge(alpha=1.0)
        model.fit(train_df[FEATURE_COLS], train_df[Y_COL])
        preds_assoc.extend(model.predict(test_df[FEATURE_COLS]))
        actuals.extend(test_df[Y_COL].values)
    actuals = np.array(actuals)
    mae_assoc = mean_absolute_error(actuals, preds_assoc)
    r2_assoc = r2_score(actuals, preds_assoc)
    print(f"Associational (fixed, all seeds): MAE={mae_assoc:.3f} R2={r2_assoc:.3f}")

    seed_results = []
    for seed in range(n_seeds):
        t0 = time.time()
        preds_causal, actuals2 = [], []
        for fold_i, (train_idx, test_idx) in enumerate(splits):
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)
            joint_kdes, age_kdes = fit_joint_and_marginal_kdes(train_df)
            boot_train = backdoor_bootstrap_joint(train_df, joint_kdes, age_kdes, seed=seed * 100 + fold_i)
            model = Ridge(alpha=1.0)
            model.fit(boot_train[FEATURE_COLS], boot_train[Y_COL])
            preds_causal.extend(model.predict(test_df[FEATURE_COLS]))
            actuals2.extend(test_df[Y_COL].values)

        mae_c = mean_absolute_error(actuals2, preds_causal)
        r2_c = r2_score(actuals2, preds_causal)
        seed_results.append({"seed": seed, "MAE": mae_c, "R2": r2_c})
        print(f"seed {seed + 1}/{n_seeds}: MAE={mae_c:.3f} R2={r2_c:.3f}  ({time.time() - t0:.0f}s)")

        # Checkpoint after every seed -- safe to stop early without losing progress
        pd.DataFrame(seed_results).to_csv("method2_grouped_multiseed_results.csv", index=False)

    results_df = pd.DataFrame(seed_results)
    print(f"\nCausal bootstrap (n={n_seeds} seeds): "
          f"MAE={results_df['MAE'].mean():.3f}+/-{results_df['MAE'].std():.3f}  "
          f"R2={results_df['R2'].mean():.3f}+/-{results_df['R2'].std():.3f}")


if __name__ == "__main__":
    df = pd.read_csv("data/raw/parkinsons_telemonitoring.csv")
    run_multiseed(df, n_seeds=10)