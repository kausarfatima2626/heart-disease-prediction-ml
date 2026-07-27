import os
from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)



       

# Robust Model Initializer (Zero External Dependencies)
def build_and_train_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(BASE_DIR, 'data', 'heart.csv'),
        os.path.join(BASE_DIR, 'heart.csv'),
        os.path.join(BASE_DIR, '..', 'data', 'heart.csv')
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                break
            except Exception:
                pass
                
    # Fallback to local embedded dataset if file paths fail
    if df is None:
        data = {
            'Age': [63, 67, 67, 37, 41, 56, 62, 57, 63, 53],
            'Sex': ['M', 'M', 'M', 'M', 'F', 'M', 'F', 'F', 'M', 'M'],
            'ChestPainType': ['TA', 'ASY', 'ASY', 'NAP', 'ATA', 'NAP', 'ASY', 'ASY', 'ASY', 'ASY'],
            'RestingBP': [145, 160, 120, 130, 130, 120, 140, 120, 130, 140],
            'Cholesterol': [233, 286, 229, 250, 204, 236, 268, 354, 254, 203],
            'FastingBS': [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            'RestingECG': ['LVH', 'LVH', 'LVH', 'Normal', 'LVH', 'Normal', 'LVH', 'Normal', 'LVH', 'Normal'],
            'MaxHR': [150, 108, 129, 187, 172, 178, 160, 163, 147, 155],
            'ExerciseAngina': ['N', 'Y', 'Y', 'N', 'N', 'N', 'N', 'Y', 'Y', 'Y'],
            'Oldpeak': [2.3, 1.5, 2.6, 3.5, 1.4, 0.8, 3.6, 0.6, 1.4, 3.1],
            'ST_Slope': ['Down', 'Flat', 'Flat', 'Down', 'Up', 'Up', 'Down', 'Up', 'Flat', 'Down'],
            'HeartDisease': [1, 1, 1, 0, 0, 0, 1, 0, 1, 1]
        }
        df = pd.DataFrame(data)
            
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
    return pipeline

# Load/Train Model
try:
    GLOBAL_MODEL = build_and_train_model()
    STARTUP_ERROR = None
except Exception as e:
    GLOBAL_MODEL = None
    STARTUP_ERROR = str(e)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    probability = None
    error = STARTUP_ERROR

    if request.method == 'POST':
        try:
            if GLOBAL_MODEL is None:
                raise Exception(f"Model initialization failed: {STARTUP_ERROR}")

            input_data = {
                'Age': float(request.form.get('Age', 0)),
                'Sex': str(request.form.get('Sex', 'M')),
                'ChestPainType': str(request.form.get('ChestPainType', 'ASY')),
                'RestingBP': float(request.form.get('RestingBP', 0)),
                'Cholesterol': float(request.form.get('Cholesterol', 0)),
                'FastingBS': int(request.form.get('FastingBS', 0)),
                'RestingECG': str(request.form.get('RestingECG', 'Normal')),
                'MaxHR': float(request.form.get('MaxHR', 0)),
                'ExerciseAngina': str(request.form.get('ExerciseAngina', 'N')),
                'Oldpeak': float(request.form.get('Oldpeak', 0.0)),
                'ST_Slope': str(request.form.get('ST_Slope', 'Flat'))
            }

            input_df = pd.DataFrame([input_data])
            pred = GLOBAL_MODEL.predict(input_df)[0]
            prob = GLOBAL_MODEL.predict_proba(input_df)[0][1]

            prediction = "High Risk" if pred == 1 else "Low Risk"
            probability = f"{round(prob * 100, 2)}%"

        except Exception as e:
            error = str(e)

    return render_template_string(HTML_TEMPLATE, prediction=prediction, probability=probability, error=error)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
