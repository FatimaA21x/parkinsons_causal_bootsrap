import pandas as pd

FEATURE_COLS = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11",
    "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE",
]

df = pd.read_csv("../data/raw/parkinsons_telemonitoring.csv")

agg_dict = {col: "mean" for col in FEATURE_COLS}
agg_dict["age"] = "first"
agg_dict["sex"] = "first"
agg_dict["motor_UPDRS"] = "mean"

subject_df = df.groupby("subject#").agg(agg_dict).reset_index()
subject_df.to_csv("../data/subject_level.csv", index=False)

print(subject_df.shape)
print(subject_df[["subject#", "age", "sex", "motor_UPDRS"]].head(10))