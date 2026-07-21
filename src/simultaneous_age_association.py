import pandas as pd
import statsmodels.api as sm

FEATURE_COLS = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11",
    "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE",
]


def simultaneous_age_association(df, feature_cols=FEATURE_COLS, sex_col="sex",
                                  age_col="age", severity_col="motor_UPDRS",
                                  group_col="subject#"):
    results = []
    groups = df[group_col].to_numpy()

    for feature in feature_cols:
        y = df[feature].to_numpy()
        X = sm.add_constant(df[[age_col, sex_col, severity_col]].to_numpy())

        model = sm.MixedLM(y, X, groups=groups)
        fit = model.fit(reml=True)

        age_coef = fit.params[1]
        age_se = fit.bse[1]
        age_p = fit.pvalues[1]
        ci_low, ci_high = fit.conf_int()[1]

        results.append({
            "feature": feature, "age_coef": age_coef, "age_se": age_se,
            "age_p": age_p, "ci_low": ci_low, "ci_high": ci_high,
            "significant": age_p < 0.05,
        })

    return pd.DataFrame(results).sort_values("age_p")