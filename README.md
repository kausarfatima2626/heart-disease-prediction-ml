
# 🫀 Heart Disease Prediction - End-to-End ML Pipeline

A production-grade Machine Learning web application designed to predict the likelihood of heart disease based on patient clinical parameters.

## 📌 Project Architecture
- **Data Engineering**: Robust preprocessing pipeline handling categorical and continuous features correctly.
- **Machine Learning**: Model comparison using Logistic Regression, Random Forest, XGBoost, and Naive Bayes.
- **Web Interface**: Lightweight Flask Web Application for real-time predictions.
- **Deployment**: CI/CD-ready structure deployed on Cloud.

## 📂 Project Structure
```text
heart-disease-prediction-ml/
├── data/             # Raw & processed datasets
├── models/           # Exported ML pipelines (.joblib/.pkl)
├── notebooks/        # Data exploration & modeling scripts
├── src/              # Modular source code
├── templates/        # HTML templates for Flask
├── app.py            # Flask backend controller
└── requirements.txt  # Project dependencies

## 🚀 Day 2 Progress
- Built robust `ColumnTransformer` preprocessing pipeline using `StandardScaler` for continuous variables and `OneHotEncoder` for categorical variables.
- Applied stratified train-test splitting (80-20) to prevent class distribution imbalance.
- Verified preprocessor transformation pipeline while strictly preventing Data Leakage.
