import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

FEATURE_COLS = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11",
    "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE",
]


def vocal_femininity_score(df, feature_cols=FEATURE_COLS, sex_col="sex",
                            group_col="subject#", n_splits=5, seed=0):
    """
    Out-of-fold predicted probability (0-1) that a recording's voice
    features indicate the 'female' (sex=1) class, using subject-grouped
    cross-validation so a subject's own recordings never leak into
    their own score.
    """
    X = df[feature_cols].to_numpy()
    y = df[sex_col].to_numpy()
    groups = df[group_col].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    scores = np.full(len(df), np.nan)

    for train_idx, test_idx in gkf.split(X, y, groups):
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X[train_idx], y[train_idx])
        scores[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

    auc = roc_auc_score(y, scores)
    print(f"Out-of-fold AUC predicting recorded sex from voice features: {auc:.3f}")
    return scores