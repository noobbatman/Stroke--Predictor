# Stroke Risk Prediction — Model Comparison Study

A systematic comparison of 6 classification algorithms on the Kaggle
Healthcare Stroke Dataset to identify the best approach for handling
extreme class imbalance in a medical prediction task.

## About this project

**Sole developer** — I ran this as a structured ML research exercise,
training each model separately in its own notebook to compare performance.

**What I implemented:**
- Data cleaning and preprocessing on the Kaggle stroke dataset
  (5,110 patients, ~4.9% positive stroke cases)
- Handled severe class imbalance using SMOTE (Synthetic Minority
  Oversampling) in the feature engineering notebook (`ferue.ipynb`)
- Trained and evaluated 6 models, each in a dedicated notebook:
  Logistic Regression, Logistic Regression + GridSearchCV tuning,
  Random Forest, K-Nearest Neighbours, Gaussian Naive Bayes,
  Voting Ensemble (combining the above)
- Compared models using AUC-ROC and recall (not accuracy — accuracy
  is misleading on a 95:5 imbalanced dataset)
- Documented findings in `stroke-prediction-research.ipynb`

**What I learnt:**
- Why accuracy is a useless metric for imbalanced medical datasets —
  a model predicting "no stroke" every time gets 95% accuracy but
  zero clinical value
- SMOTE — how synthetic oversampling works and when it helps vs. hurts
- The precision-recall trade-off in life-critical predictions: high
  recall (catch real strokes) matters more than low false positives
- How ensemble methods (voting classifier) stabilise predictions across
  weak individual models
- Hyperparameter tuning with GridSearchCV and its effect on generalisation

## Dataset
Kaggle: Healthcare Dataset Stroke Data
- 5,110 patient records | 11 features
- Features: age, hypertension, heart disease, avg_glucose_level,
  BMI, smoking_status, etc.
- Class imbalance: ~4.9% stroke cases

## Models compared
| Model | Notes |
|---|---|
| Logistic Regression | Baseline |
| Logistic Regression + GridSearchCV | Hyperparameter tuned |
| Random Forest | Ensemble of decision trees |
| K-Nearest Neighbours | Distance-based |
| Gaussian Naive Bayes | Probabilistic, fast |
| Voting Ensemble | Combines all above |

## Tech stack
Python · scikit-learn · imbalanced-learn (SMOTE) · pandas ·
matplotlib · seaborn · Jupyter Notebook
