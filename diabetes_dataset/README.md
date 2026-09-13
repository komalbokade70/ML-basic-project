# Diabetes Risk Prediction
Predicting whether someone is diabetic using the Pima Indians Diabetes
dataset. Built with XGBoost, some engineered features, SMOTE for the
class imbalance, and a decision threshold tuned specifically to catch
more actual diabetic cases.

### Why this problem
For a screening model like this, missing someone who's actually
diabetic is worse than flagging someone who isn't. So instead of just
chasing accuracy, I focused on getting recall up on the diabetic class
— catching more real cases mattered more than a clean overall score.

### Dataset
#### Indians Diabetes Database
- 768 patients, 8 features: Pregnancies, Glucose, BloodPressure,
SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- Target: Outcome (0 = not diabetic, 1 = diabetic), split roughly
65/35 — imbalanced enough to need handling

### What I did
- Started with EDA — class balance, distributions, correlations,
outliers via boxplots.
- Five columns (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
use 0 as a stand-in for missing data, which doesn't make sense
biologically. Converted those zeros to NaN and filled them with the
median — computed from the training set only, so nothing from the
test set leaks in.
- Added a few engineered features: BMI_Glucose, Insulin_Glucose,
and Age_BMI.
- Used BorderlineSMOTE on the training data only to deal with the
class imbalance.
- Trained an XGBClassifier (max_depth=3, learning_rate=0.03,
subsample=0.8 — tuned for a small, imbalanced dataset like this).
- Instead of sticking with the default 0.5 cutoff, I picked a
threshold using precision-recall analysis — the one that maximizes
F1 among all thresholds hitting at least 90% recall.
- Evaluated with a classification report, confusion matrix, and
cross-validation done properly — SMOTE gets refit inside each CV
fold (via an imblearn Pipeline) instead of being applied once
before splitting, which would leak information across folds.

### Results
| Metric | Score |
|---|---|
| Test accuracy (default 0.5 threshold) | 0.75 |
| Test accuracy (tuned threshold) | 0.70 |
| Cross-validated F1 (leakage-free) | 0.79 |
| Train accuracy | 0.82 |

Accuracy drops a bit once I switch to the tuned threshold, but recall
on the diabetic class jumps from about 0.65 to 0.89. That's the whole
point of tuning it — for a screening tool, catching real cases matters
more than a slightly higher accuracy number.
Glucose, Age×BMI, and BMI×Glucose came out as the most important
features by a good margin.

### What I learned
Two leakage mistakes are easy to make and both quietly inflate your
metrics if you're not careful: imputing missing values using stats
from the whole dataset instead of train-only, and running SMOTE before
splitting into CV folds instead of inside each fold.
Also — the default 0.5 threshold isn't always the right call. For an
imbalanced, cost-sensitive problem like this, tuning the threshold
against the precision/recall trade-off matters more than optimizing
for raw accuracy.

### Stack
Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn,
seaborn/matplotlib
