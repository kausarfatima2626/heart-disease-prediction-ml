
import os
from flask import Flask, request, render_template_string
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)



# Inline HTML Template to avoid missing file errors# Base Paths with Smart Fallback
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOCAL_DATA = os.path.join(BASE_DIR, 'data', 'heart.csv')
ALT_DATA = os.path.join(BASE_DIR, '..', 'data', 'heart.csv')
ONLINE_DATA = 'https://raw.githubusercontent.com/fedesoriano/heart-failure-prediction/main/heart.csv'

if os.path.exists(LOCAL_DATA):
    DATA_PATH = LOCAL_DATA
elif os.path.exists(ALT_DATA):
    DATA_PATH = ALT_DATA
else:
    DATA_PATH = ONLINE_DATA
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Heart Disease Prediction</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
</head>
<body class="bg-light p-4">
    <div class="container" style="max-width: 600px;">
        <h2 class="text-center mb-4 text-primary">Heart Disease Risk Predictor</h2>
        
        {% if error %}
            <div class="alert alert-danger">{{ error }}</div>
        {% endif %}

        {% if prediction %}
            <div class="alert {% if prediction == 'High Risk' %}alert-danger{% else %}alert-success{% endif %} text-center">
                <h3>Prediction: {{ prediction }}</h3>
                <p>Confidence / Probability: {{ probability }}</p>
            </div>
        {% endif %}

        <form method="POST" class="card p-4 shadow-sm bg-white">
            <div class="mb-3"><label>Age:</label><input type="number" name="Age" class="form-control" value="50" required></div>
            <div class="mb-3"><label>Sex (M/F):</label><input type="text" name="Sex" class="form-control" value="M" required></div>
            <div class="mb-3"><label>Chest Pain Type (TA/ATA/NAP/ASY):</label><input type="text" name="ChestPainType" class="form-control" value="ASY" required></div>
            <div class="mb-3"><label>Resting BP:</label><input type="number" name="RestingBP" class="form-control" value="140" required></div>
            <div class="mb-3"><label>Cholesterol:</label><input type="number" name="Cholesterol" class="form-control" value="280" required></div>
            <div class="mb-3"><label>Fasting BS (0 or 1):</label><input type="number" name="FastingBS" class="form-control" value="0" required></div>
            <div class="mb-3"><label>Resting ECG (Normal/ST/LVH):</label><input type="text" name="RestingECG" class="form-control" value="Normal" required></div>
            <div class="mb-3"><label>Max HR:</label><input type="number" name="MaxHR" class="form-control" value="150" required></div>
            <div class="mb-3"><label>Exercise Angina (Y/N):</label><input type="text" name="ExerciseAngina" class="form-control" value="N" required></div>
            <div class="mb-3"><label>Oldpeak:</label><input type="number" step="0.1" name="Oldpeak" class="form-control" value="1.0" required></div>
            <div class="mb-3"><label>ST Slope (Up/Flat/Down):</label><input type="text" name="ST_Slope" class="form-control" value="Flat" required></div>
            
            <button type="submit" class="btn btn-primary w-100">Analyze Risk</button>
        </form>
    </div>
</body>
</html>
'''

def get_trained_model():
    if not os.path.exists(DATA_PATH):
        return None
        
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
    return pipeline

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    probability = None
    error = None

    if request.method == 'POST':
        try:
            model = get_trained_model()
            if model is None:
                raise Exception("Data file missing at path: " + DATA_PATH)

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
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            prediction = "High Risk" if pred == 1 else "Low Risk"
            probability = f"{round(prob * 100, 2)}%"

        except Exception as e:
            error = str(e)

    return render_template_string(HTML_TEMPLATE, prediction=prediction, probability=probability, error=error)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
