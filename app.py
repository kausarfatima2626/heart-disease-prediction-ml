import os
from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'heart_disease_pipeline.joblib')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'heart.csv')

def get_or_train_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    
    # Auto-train if missing
    if os.path.exists(DATA_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        df = pd.read_csv(DATA_PATH)
        X = df.drop('HeartDisease', axis=1)
        y = df['HeartDisease']

        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        pipeline.fit(X, y)
        joblib.dump(pipeline, MODEL_PATH)
        return pipeline
    return None

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    probability = None
    error = None

    if request.method == 'POST':
        try:
            model = get_or_train_model()
            if model is None:
                raise Exception("Dataset/Model not found on server.")

            input_data = {
                'Age': float(request.form.get('Age', 0)),
                'Sex': request.form.get('Sex', 'M'),
                'ChestPainType': request.form.get('ChestPainType', 'ASY'),
                'RestingBP': float(request.form.get('RestingBP', 0)),
                'Cholesterol': float(request.form.get('Cholesterol', 0)),
                'FastingBS': int(request.form.get('FastingBS', 0)),
                'RestingECG': request.form.get('RestingECG', 'Normal'),
                'MaxHR': float(request.form.get('MaxHR', 0)),
                'ExerciseAngina': request.form.get('ExerciseAngina', 'N'),
                'Oldpeak': float(request.form.get('Oldpeak', 0.0)),
                'ST_Slope': request.form.get('ST_Slope', 'Flat')
            }

            input_df = pd.DataFrame([input_data])
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            prediction = "High Risk" if pred == 1 else "Low Risk"
            probability = f"{round(prob * 100, 2)}%"

        except Exception as e:
            error = str(e)

    return render_template('index.html', prediction=prediction, probability=probability, error=error)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
