import pandas as pd
import numpy as np

# 1. Dataset Load Karna
df = pd.read_csv('../data/hearts.csv')

print("=== 1. Dataset Overview ===")
print(df.head())
print("\nShape of Dataset:", df.shape)

print("\n=== 2. Data Types & Missing Values ===")
print(df.info())
print("\nMissing Values Count:")
print(df.isnull().sum())

print("\n=== 3. Target Distribution (Class Balance Check) ===")
# Target column target/HeartDisease ke distribution ko check karna
target_col = 'HeartDisease' if 'HeartDisease' in df.columns else df.columns[-1]
print(df[target_col].value_counts(normalize=True) * 100)

print("\n=== 4. Categorical vs Continuous Columns ===")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("Categorical Columns:", categorical_cols)
print("Numerical Columns:", numerical_cols)
