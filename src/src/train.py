import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

from preprocessing import get_preprocessor

def train_and_evaluate():
    # 1. Load Dataset
    data_path = os.path.join(os.path.dirname(__file__), '../data/hearts.csv')
    df = pd.read_csv(data_path)

    target_col = 'HeartDisease' if 'HeartDisease' in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Get Preprocessing Pipeline
    preprocessor = get_preprocessor()

    # 4. Define Candidate Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    best_recall = 0.0
    best_pipeline = None
    best_model_name = ""

    print("=== Model Training and Evaluation ===\n")

    for name, model in models.items():
        # Create full end-to-end pipeline (Preprocessor + Model)
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Train pipeline
        full_pipeline.fit(X_train, y_train)

        # Predict on Test set
        y_pred = full_pipeline.predict(X_test)

        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Model: {name}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1 Score : {f1:.4f}\n")

        # Select model based on highest Recall (Medical Safety Priority)
        if rec > best_recall:
            best_recall = rec
            best_pipeline = full_pipeline
            best_model_name = name

    print(f"🏆 Best Model Selected: {best_model_name} (Recall: {best_recall:.4f})")

    # 5. Export Saved Pipeline Artifact
    models_dir = os.path.join(os.path.dirname(__file__), '../models')
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, 'heart_disease_pipeline.joblib')
    
    joblib.dump(best_pipeline, save_path)
    print(f"Saved best model pipeline successfully to '{save_path}'!")

if __name__ == "__main__":
    train_and_evaluate()
