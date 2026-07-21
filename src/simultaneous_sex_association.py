import statsmodels.api as sm
import pandas as pd

FEATURE_COLS = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11",
    "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE",
]


def simultaneous_sex_association(df, feature_cols=FEATURE_COLS, sex_col="sex",
                                  age_col="age", severity_col="motor_UPDRS",
                                  group_col="subject#"):
    """
    Models each voice feature (X) as the outcome, with sex (G), age (A),
    and severity (U) as simultaneous fixed-effect predictors, and subject
    identity (I) as a random intercept. Matches Max's causal graph
    (U<-A->X, U<-G->X, U<-I->X, U->X) directly.
    """
    results = []
    groups = df[group_col].to_numpy()

    for feature in feature_cols:
        y = df[feature].to_numpy()
        X = sm.add_constant(df[[sex_col, age_col, severity_col]].to_numpy())

        model = sm.MixedLM(y, X, groups=groups)
        fit = model.fit(reml=True)

        sex_coef = fit.params[1]
        sex_se = fit.bse[1]
        sex_p = fit.pvalues[1]
        ci_low, ci_high = fit.conf_int()[1]

        results.append({
            "feature": feature, "sex_coef": sex_coef, "sex_se": sex_se,
            "sex_p": sex_p, "ci_low": ci_low, "ci_high": ci_high,
            "significant": sex_p < 0.05,
        })

    return pd.DataFrame(results).sort_values("sex_p")
