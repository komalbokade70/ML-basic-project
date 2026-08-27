Used Column Transformer with One Hot + Ordinal + Standard Scaler feeding into a soft Voting Classifier (XGBoost + Balanced RandomForest).
This improved test accuracy from 72% to 74.28% with cross-val-score 75.01% and reduced overfitting. At threshold 0.50, F1=0.40, at threshold 0.30, Recall for dead class improves to 88%.
