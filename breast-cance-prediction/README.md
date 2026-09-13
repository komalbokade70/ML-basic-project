Predicting whether a breast cancer patient survives (Alive/Dead) using
clinical and pathological data from the SEER dataset. Built with an
XGBoost + Random Forest voting ensemble, with the decision threshold
tuned specifically to catch more at-risk patients.

### Why this problem
In a survival prediction task, missing a patient who's actually at risk
is a lot worse than raising a false alarm. So instead of optimizing for
plain accuracy, I focused on getting the model to catch as many "Dead"
cases as possible, even if that meant giving up some overall accuracy.

### Dataset
- SEER Breast Cancer dataset — 4,024 patient records
- Target: Status (Alive / Dead), and it's heavily imbalanced — most
patients in the data survived
- Dropped Survival Months from the features on purpose — it's only
known after the fact, so keeping it in would basically be handing
the model the answer

### Cleaning
- Removed 1 duplicate row, stripped stray whitespace from column names
- Capped a couple of outlier-prone columns: Tumor Size at 100, Regional
Node Examined at 40
- No missing values to deal with

### New features I added
- Is_High_Risk — big tumor (>50) plus more than 3 positive nodes
- Node_Ratio — how many nodes came back positive relative to how
many were checked.
- Is_Triple_Neg — flags triple-negative cases (both hormone receptors
negative), which tend to behave more aggressively.
- High_Tumor — simple flag for tumor size over 50.

### Encoding
Split the columns into three groups and treated each differently:
- Plain categorical stuff (Race, Marital Status, hormone receptor
status, etc.) → one-hot encoded
- Staged/ordered categorical columns (T Stage, N Stage, Grade, 6th
Stage) → ordinal encoded, keeping the actual clinical order (T1 <
T2 < T3 < T4, and so on) instead of treating them as unrelated
categories
- Numeric columns → standard scaled
All of this is wrapped in a single ColumnTransformer so it's one
pipeline instead of a pile of separate preprocessing steps.

### Model
Went with a soft-voting ensemble of XGBoost and Random Forest instead
of a single model. Both models handle the class imbalance on their
own — XGBoost via scale_pos_weight, Random Forest via
class_weight='balanced' — so there was no need to oversample the
data

### Threshold tuning
The default 0.5 cutoff only caught about 55% of the Dead cases, which
didn't feel good enough for this kind of problem. I swept thresholds
from 0.12 up to 0.5 and tracked F1 and recall on the Dead class at
each one:

| Threshold | F1 | Recall (Dead) |
|---|---|---|
| 0.12 | 0.271 | 99.2% |
| 0.25 | 0.320 | 91.1% |
| 0.33 | 0.360 | 82.1% |
| *0.346 (chosen)* | *~0.36* | *~80%* |
| 0.40 | 0.363 | 69.9% |
| 0.50 (default) | 0.411 | 55.3% 

Settled on 0.346 — it nearly triples recall on Dead cases compared to
the default threshold. Yes, this tanks precision and overall accuracy,
but that's the point: for this problem, catching more real cases
matters more than a clean-looking accuracy number.

### Results
At the default 0.5 threshold: 79.98% train accuracy, 75.77% test accuracy,
but only ~55% recall on Dead cases — meaning it was missing almost
half the patients who actually died.
At the tuned 0.346 threshold: accuracy drops to about 56.64% on test, but
recall on Dead cases jumps to ~80%. Precision for that class is low
(23%), which is the trade-off — more false alarms, but far fewer missed
at-risk patients.
One thing I'd do differently with more time: I tuned the threshold by
checking directly against the test set. Ideally you'd carve out a
separate validation set for that and only touch the test set once,
right at the end, to keep the final numbers honest.

### Stack
Python, pandas, NumPy, seaborn/matplotlib, scikit-learn
(ColumnTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler,
RandomForestClassifier, VotingClassifier), XGBoost
