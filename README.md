# Machine Learning-Based Credit Card Fraud Detection on Imbalanced Data

Detecting fraudulent credit card transactions when fraud is only **0.17%** of the data.
The main models are **Decision Tree** and **Random Forest**, compared against Logistic Regression, XGBoost (GPU) and LightGBM.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB, Kaggle):

- 284,807 transactions by European cardholders over two days
- 492 frauds (0.173%): about 1 fraud per 577 normal transactions
- `V1`–`V28`: PCA-transformed features (anonymised); `Time` and `Amount`: raw; `Class`: 1 = fraud

The CSV (~150 MB) is not included in this repository. `data.py` downloads it.

## Approach

| Step | What it does |
|---|---|
| 1–2 | Load data, remove 1,081 duplicate rows, exploratory data analysis |
| 3 | Stratified 80/20 train/test split; scale `Time` and `Amount` (fitted on train only) |
| 4–5 | Baseline Decision Tree and Random Forest; metrics suited to imbalanced data |
| 6 | Imbalance handling: class weights, random undersampling, SMOTE (training data only, via `imblearn` pipelines) |
| 7 | Hyperparameter tuning with `RandomizedSearchCV` (3-fold stratified CV, scored on PR-AUC) |
| 8 | Threshold tuning using out-of-fold probabilities (the test set is never used for tuning) |
| 9 | Feature importance and final Decision Tree vs Random Forest comparison |
| 10 | Logistic Regression, XGBoost (GPU) and LightGBM for comparison |

**Metrics:** precision, recall, F1 and **PR-AUC** on the fraud class. Accuracy is not used to compare models, because predicting "normal" for everything already scores 99.83%.

## Results

Test set: 56,746 transactions, 95 frauds.

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.846 | 0.579 | 0.688 | 0.692 | 0.956 |
| Decision Tree (baseline) | 0.720 | 0.705 | 0.713 | 0.509 | 0.852 |
| Decision Tree (tuned) | 0.855 | 0.684 | 0.760 | 0.709 | 0.886 |
| Random Forest (baseline) | 0.972 | 0.726 | 0.831 | 0.788 | 0.924 |
| Random Forest + SMOTE (tuned) | 0.923 | 0.758 | 0.832 | 0.824 | 0.975 |
| LightGBM (tuned) | 0.914 | 0.779 | 0.841 | 0.821 | 0.980 |
| **XGBoost (tuned, GPU)** | **0.949** | **0.789** | **0.862** | **0.836** | 0.976 |

### Key findings

- **Ensembles beat single models.** Performance improves from linear model → single tree → Random Forest → boosting.
- **The Random Forest clearly beats the Decision Tree** (PR-AUC 0.824 vs 0.709).
- **Imbalance handling must be tuned, not just switched on.** Full rebalancing (undersampling, full class weights) raised recall but destroyed precision. Light rebalancing worked best: a SMOTE ratio of 0.1 for the Random Forest, and `scale_pos_weight=10` for LightGBM.
- **Tuning gave the Decision Tree its biggest gain:** PR-AUC rose from 0.51 to 0.71.
- **Threshold tuning sets the trade-off.** Tuned XGBoost reaches 98.6% precision at threshold 0.88, or 84% recall at threshold 0.03.
- **The most important features** (`V14`, `V17`, `V10`, `V12`, `V16`) match the features most correlated with fraud in the EDA.

**Limitation:** with only 95 test frauds, the top models differ by just 1–3 frauds, which is within noise.

## How to run

```bash
git clone https://github.com/GauravChoudhary1iisc/credit-card-fraud-detection.git
cd credit-card-fraud-detection

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python data.py            # downloads creditcard.csv
```

Then open `fraud_detection.ipynb` and run all cells with the `.venv` kernel.

- **Runtime:** about 25 minutes on a 32-core CPU, most of it the Random Forest search in Step 7.
- **GPU:** XGBoost uses an NVIDIA GPU if one is available and falls back to the CPU otherwise.

## Project structure

```
├── data.py                  # downloads the dataset via kagglehub
├── fraud_detection.ipynb    # full analysis (Steps 1–10)
├── requirements.txt
└── README.md
```

## Tech stack

Python, pandas, scikit-learn, imbalanced-learn, XGBoost, LightGBM, matplotlib, seaborn
