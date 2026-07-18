# Titanic Survival Prediction

A binary classification project predicting passenger survival on the Titanic, comparing eight ML models on accuracy, precision, recall, and F1.

## Problem Statement

Given passenger attributes (class, sex, age, family size, fare, port of embarkation, title), predict whether a passenger survived the sinking of the Titanic.

## Target Variable

`Survived` — binary: `1` = survived, `0` = did not survive.
Class balance in the training set (891 rows): ~62% did not survive, ~38% survived — a mild imbalance, not severe enough to require resampling, but worth keeping in mind when interpreting accuracy.

## Machine Learning Task

This is a binary classification problem because the target variable contains two discrete outcomes: 0 and 1. The model must assign each observation to one of these two classes rather than predict a continuous numerical quantity.

## Evaluation Metric

Recall is the priority metric because false negatives carry a high cost here: predicting a survivor as "did not survive" (a missed positive case) is treated as more harmful than the reverse. Accuracy, precision, and F1 are also tracked for comparison, but model selection favors recall-sensitive performance over raw accuracy.

## Baseline Strategy

The naive baseline is predicting the majority class for every passenger — always "did not survive" — which gets ~62% accuracy for free. Any model in this project needs to clear that bar by a meaningful margin to be worth using. A slightly stronger baseline (predict survival based on `Sex` alone, since "women and children first" is a strong known prior) sits well above the majority-class baseline and is a useful sanity check when interpreting the trained models' scores below.

## Potential Data Leakage Risks

- **`PassengerId`, `Ticket`, `Cabin`** are dropped before training. `PassengerId` is an arbitrary index; `Ticket` and `Cabin` are high-cardinality/mostly-missing (Cabin is ~77% null) and risk the model memorizing per-passenger identity rather than learning generalizable patterns.
- **Age imputation** is done inside an sklearn `Pipeline` (`SimpleImputer` fit only on the training fold), not on the full dataset before the split — this avoids leaking test-set statistics into training.
- **Title extraction from `Name`** (Mr/Mrs/Miss/Master/Rare) is a legitimate engineered feature, not leakage, since it's derivable from information available at prediction time.
- **Family size** (`SibSp + Parch + 1`) is similarly safe — no target information involved.
- No leakage from the target itself was identified; all transformations only use pre-outcome passenger attributes.

## Project Structure

```
├── data/raw/           # train.csv, test.csv (original Kaggle Titanic data)
├── notebooks/          # 01_eda.ipynb — exploratory data analysis
├── src/
│   ├── preprocessing.py   # title extraction, family size, column drops
│   ├── encoding.py        # sklearn ColumnTransformer pipelines (scaled/unscaled)
│   ├── train.py           # trains 8 models, saves the best as models/model.joblib
│   ├── evaluate.py        # accuracy/precision/recall/f1/confusion matrix
│   ├── visualize.py       # metric bar charts, confusion matrices, ROC curve
│   └── ml_utils.py        # missing-value summary helper
├── models/model.joblib # best trained model (Logistic Regression), saved after running main.py
├── reports/            # saved figures (age imputation comparisons, model metrics, ROC curve)
└── main.py             # trains all models, evaluates, visualizes, predicts on one sample
```

## Models Compared

Eight classifiers are trained and compared: Decision Tree, Random Forest, AdaBoost, XGBoost, ExtraTrees (no scaling needed), and Logistic Regression, KNN, SVM (scaled features). The best model by accuracy is saved to `models/model.joblib`.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** | **0.844** | 0.825 | 0.754 | 0.788 |
| ExtraTrees | 0.832 | 0.800 | 0.754 | 0.776 |
| Decision Tree | 0.816 | 0.765 | 0.754 | 0.759 |
| Random Forest | 0.810 | 0.778 | 0.710 | 0.742 |
| AdaBoost | 0.810 | 0.740 | 0.783 | 0.761 |
| XGBoost | 0.804 | 0.736 | 0.768 | 0.752 |
| KNN | 0.765 | 0.708 | 0.667 | 0.687 |
| SVM | 0.637 | 0.563 | 0.261 | 0.356 |

Logistic Regression is the best performer on accuracy and is the model saved for inference.

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

This trains all models, prints the best one, saves it to `models/model.joblib`, generates comparison plots, and predicts survival for one sample from `test.csv`.

## Known Limitations

- `requirements.txt` is missing `xgboost`, `plotly`, and `seaborn` — install these manually if `pip install -r requirements.txt` isn't enough.
- `train()` currently runs twice per execution of `main.py` (once via a module-level call inside `visualize.py`, once explicitly in `main.py`) — harmless but wasteful.
- The ROC curve is plotted on hard class predictions rather than predicted probabilities; using `predict_proba` would give a smoother, more accurate curve.
