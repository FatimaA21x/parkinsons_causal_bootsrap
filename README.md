# Addressing Identity Confounding in Causal Bootstrapping

**A Case Study on Parkinson's Voice-Based UPDRS Prediction**

MSc Data Science Dissertation — Fatima Akhtar, University of Birmingham

This project extends Little & Badawy's (2019) causal bootstrapping method to
Parkinson's voice-based UPDRS prediction, discovers a positivity violation
caused by identity confounding in longitudinal, per-subject clinical data,
and develops and evaluates two methods for addressing it.

Full write-up: see the accompanying dissertation report.

---

## Overview

- **Dataset**: UCI Parkinson's Telemonitoring dataset (Tsanas et al., 2010),
  5,875 voice recordings from 42 subjects over a 6-month trial
- **Problem discovered**: age/sex confounding cannot be adjusted for using
  standard back-door causal bootstrapping, because subject identity locks
  age and sex to a single value per subject, creating empty "support holes"
  in the age × sex space and sending resampling weights to infinity
- **Method 1**: Subject-level aggregation, collapses to 42 independent
  subjects, adjusts for sex only
- **Method 2**: Joint distribution modelling, keeps all 5,875 recordings,
  models age as a soft distribution via joint KDE + Bayes' rule, adjusts
  for age and sex jointly
- **Confounder validity check**: mixed-effects models testing whether age
  and sex genuinely relate to any of the 16 voice features
- **Weight diagnostics**: effective sample size checks confirming both
  methods' resampling weights are well-behaved, not degenerate

---

## Repository Structure

```
├── data/
│   ├── raw/
│   │   ├── parkinsons_telemonitoring.csv   # UCI dataset
│   │   └── parkinsons_updrs.names          # official dataset documentation
│   └── subject_level.csv                   # aggregated 42-row dataset (Method 1)
├── notebook/
│   └── 01_eda.ipynb                        # exploratory data analysis
├── reports/
│   └── Dissertation Presentation 24th.pptx # demo slide deck
├── src/
│   ├── aggregate_subjects.py               # builds subject_level.csv (Method 1 prep)
│   ├── method1_bootstrap.py                # Method 1: subject-level aggregation, LOO evaluation
│   ├── method1_weight_diagnostic.py        # Method 1: effective sample size check on fitted weights
│   ├── method2_joint_kde.py                # Method 2: joint KDE + Bayes' rule density estimation
│   ├── method2_5fold_eval.py               # Method 2: single-run subject-grouped + random-split evaluation
│   ├── method2_random_split_test.py        # Method 2: naive random-split diagnostic (Little, 2017)
│   ├── method2_multiseed.py                # Method 2: 10-seed averaged subject-grouped evaluation
│   ├── method2_weight_diagnostic.py        # Method 2: effective sample size check on fitted weights
│   ├── simultaneous_sex_association.py     # confounder validity check: mixed-effects model (sex)
│   └── simultaneous_age_association.py     # confounder validity check: mixed-effects model (age)
├── method2_grouped_multiseed_results.csv   # saved output of the 10-seed Method 2 run
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
git clone <your-repo-url>
cd parkinsons_causal_bootsrap
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the Code

**1. Exploratory data analysis**

```bash
jupyter notebook notebook/01_eda.ipynb
```

**2. Method 1 (subject-level aggregation)**

```bash
python -m src.aggregate_subjects            # produces data/subject_level.csv
python -m src.method1_bootstrap             # runs leave-one-out benchmark, 10 seeds
python -m src.method1_weight_diagnostic     # effective sample size check on fitted weights
```

**3. Method 2 (joint distribution modelling)**

```bash
python -m src.method2_5fold_eval            # single-run subject-grouped + random-split evaluation
python -m src.method2_random_split_test     # naive random-split diagnostic (Little, 2017)
python -m src.method2_multiseed             # 10-seed averaged subject-grouped evaluation
python -m src.method2_weight_diagnostic     # effective sample size check on a representative sample
```
Note: a full Method 2 run takes ~20-25 minutes due to the cost of resampling
across all 5,875 recordings.

**4. Confounder validity check**

```bash
python -m src.simultaneous_sex_association
python -m src.simultaneous_age_association
```

---

## Key Results

| Method | Evaluation | Associational | Causal Bootstrap |
|---|---|---|---|
| Method 1 (N=42) | Leave-one-out | MAE 7.05, R² −0.056 | MAE 6.96 (SD 0.20), R² −0.046 |
| Method 2 (N=5,875) | Subject-grouped 5-fold | MAE 7.09, R² −0.035 | MAE 7.23 (SD 0.025), R² −0.064 |
| Method 2 (diagnostic) | Random 5-fold | MAE 6.60, R² +0.087 | MAE 6.69, R² +0.065 |

Method 1's improvement is not statistically significant (t(9) = −1.56,
p = 0.154); Method 2's decline is highly significant (t(9) = 17.51,
p < .001). A mixed-effects confounder check found neither age nor sex
significantly relates to any of the 16 voice features individually,
consistent with the small effect sizes observed for both methods.

Weight diagnostics confirm both methods' resampling weights are
well-behaved: Method 1's mean effective sample size was 18.3 out of ~41
candidates (minimum 6.0), and Method 2's mean was 824 out of ~5,874
candidates (minimum 257, on a representative sample of 300 target points),
neither degenerating onto a single or handful of points.

---

## Citation

If referencing this work, please cite the accompanying dissertation:

> Akhtar, F. (2026). *Addressing Identity Confounding in Causal
> Bootstrapping: A Case Study on Parkinson's Voice-Based UPDRS Prediction*
> [MSc dissertation]. University of Birmingham.