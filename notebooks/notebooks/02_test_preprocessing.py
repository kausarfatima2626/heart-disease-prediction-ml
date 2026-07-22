import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Add 'src' directory to python path to import our preprocessing module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import get_preprocessor

# 1. Load Dataset
df = pd.read_csv('../data/hearts.csv')

# 2. Separate Features (X) and Target Label (y)
target_col = 'HeartDisease' if 'HeartDisease' in df.columns else df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

# 3. Train-Test Split (80% Train, 20% Test)
# stratify=y preserves the ratio of Heart Disease vs Normal patients in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Dataset Split Successfully!")
print(f"Training Features Shape: {X_train.shape}")
print(f"Testing Features Shape: {X_test.shape}")

# 4. Initialize Preprocessor & Apply Transformations
preprocessor = get_preprocessor()

# Fit only on Training Data (prevents Data Leakage)
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

print("\n=== Preprocessing Verification ===")
print(f"Transformed Training Shape: {X_train_transformed.shape}")
print(f"Sample Transformed Row (First patient):")
print(X_train_transformed[0])
