# Credit Card Default Prediction

Predicting whether a customer will default on their credit card payment next month, using EDA, feature engineering, and multiple classification models.

## 📌 Overview

This project analyzes customer credit card data to predict `next_month_default` (binary target). It covers:

- Data cleaning and missing value handling
- Exploratory Data Analysis (EDA) — distributions of sex, education, marriage, age, credit limit, etc.
- Feature engineering:
  - **Payment-to-Bill Ratio** trend analysis
  - **Credit Utilization** analysis
- Class imbalance handling using **SMOTE**
- Model training and comparison:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
  - Hyperparameter-tuned XGBoost (via `RandomizedSearchCV`)
- Threshold tuning per model to maximize **F2 score** (recall-weighted, since catching defaulters matters more than precision here)
- Final predictions generated on a validation set using the best-performing model

## 📊 Model Performance Summary

| Model                          | Accuracy | F2 Score |
|---------------------------------|----------|----------|
| Logistic Regression             | 0.4188   | 0.5599   |
| Decision Tree                   | 0.5780   | 0.5707   |
| Random Forest                   | 0.5604   | 0.6023   |
| XGBoost                         | 0.5190   | 0.5858   |
| Hyperparameter-Tuned XGBoost    | 0.5612   | 0.5959   |

Random Forest gave the best F2 score and was used to generate the final validation predictions.

## 🗂️ Project Structure

```
credit-card-default-prediction/
├── credit_default.ipynb     # Main notebook: EDA, feature engineering, modelling
├── data/                     # (not tracked) place train/validation CSVs here
├── requirements.txt
├── .gitignore
└── README.md
```

## 📁 Dataset

The notebook expects two CSV files:
- `train_dataset_final1.csv`
- `validate_dataset_final.csv`

These were originally loaded from Google Drive (via Colab). Since the data isn't included in this repo, update the file paths in the first cell to point to a local `data/` folder, e.g.:

```python
train_path = "data/train_dataset_final1.csv"
val_path = "data/validate_dataset_final.csv"
```

> If your dataset is public, consider adding a download link or source here. If it's private/proprietary, keep it excluded (already handled by `.gitignore`).

## ⚙️ Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/credit-card-default-prediction.git
   cd credit-card-default-prediction
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Add your data files to a `data/` folder, update paths in the notebook, then run:
   ```bash
   jupyter notebook credit_default.ipynb
   ```

## 🛠️ Tech Stack

- Python
- pandas, numpy
- seaborn, matplotlib
- scikit-learn
- imbalanced-learn (SMOTE)
- xgboost

## 📈 Future Improvements

- Try additional models (LightGBM, CatBoost)
- Feature selection / SHAP-based interpretability
- Cross-validation instead of a single train/test split
- Deploy as an API or simple web app for inference

## 📄 License

This project is open-sourced under the MIT License — feel free to use and modify.