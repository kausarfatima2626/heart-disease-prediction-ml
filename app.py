import os
from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Heart Disease Risk Predictor</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            /* Aesthetic Medical Background Image with Overlay */
            background: linear-gradient(rgba(15, 23, 42, 0.75), rgba(15, 23, 42, 0.85)), 
                        url('https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=1920&q=80') no-repeat center center fixed;
            background-size: cover;
            min-height: 100vh;
            color: #333;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.92);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
        }
        .form-label {
            font-weight: 600;
            color: #1e293b;
            font-size: 0.9rem;
        }
        .form-control {
            border-radius: 10px;
            border: 1px solid #cbd5e1;
            padding: 10px 14px;
        }
        .form-control:focus {
            box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.25);
            border-color: #0ea5e9;
        }
        .btn-custom {
            background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%);
            border: none;
            border-radius: 12px;
            font-weight: 600;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
        }
        .btn-custom:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.4);
        }
        .title-text {
            color: #ffffff;
            font-weight: 600;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }
    </style>
</head>
<body class="d-flex align-items-center justify-content-center py-5">
    <div class="container" style="max-width: 650px;">
        <h2 class="text-center mb-4 title-text">❤️ Heart Disease Risk Predictor</h2>
        
        {% if error %}
            <div class="alert alert-danger rounded-4 shadow-sm mb-4">{{ error }}</div>
        {% endif %}

        {% if prediction %}
            <div class="alert {% if prediction == 'High Risk' %}alert-danger{% else %}alert-success{% endif %} text-center shadow-lg rounded-4 p-4 mb-4">
                <h3 class="fw-bold mb-1">Prediction: {{ prediction }}</h3>
                <p class="mb-0 fs-5">Confidence Level: {{ probability }}</p>
            </div>
        {% endif %}

        <form method="POST" class="glass-card p-4 p-md-5">
            <div class="row g-3">
                <div class="col-md-6"><label class="form-label">Age</label><input type="number" name="Age" class="form-control" value="50" required></div>
                <div class="col-md-6"><label class="form-label">Sex (M/F)</label><input type="text" name="Sex" class="form-control" value="M" required></div>
                <div class="col-md-6"><label class="form-label">Chest Pain Type</label><input type="text" name="ChestPainType" class="form-control" value="ASY" required></div>
                <div class="col-md-6"><label class="form-label">Resting BP (mm Hg)</label><input type="number" name="RestingBP" class="form-control" value="140" required></div>
                <div class="col-md-6"><label class="form-label">Cholesterol (mm/dl)</label><input type="number" name="Cholesterol" class="form-control" value="280" required></div>
                <div class="col-md-6"><label class="form-label">Fasting BS (0 or 1)</label><input type="number" name="FastingBS" class="form-control" value="0" required></div>
                <div class="col-md-6"><label class="form-label">Resting ECG</label><input type="text" name="RestingECG" class="form-control" value="Normal" required></div>
                <div class="col-md-6"><label class="form-label">Max HR</label><input type="number" name="MaxHR" class="form-control" value="150" required></div>
                <div class="col-md-6"><label class="form-label">Exercise Angina (Y/N)</label><input type="text" name="ExerciseAngina" class="form-control" value="N" required></div>
                <div class="col-md-6"><label class="form-label">Oldpeak</label><input type="number" step="0.1" name="Oldpeak" class="form-control" value="1.0" required></div>
                <div class="col-12"><label class="form-label">ST Slope (Up/Flat/Down)</label><input type="text" name="ST_Slope" class="form-control" value="Flat" required></div>
            </div>
            
            <button type="submit" class="btn btn-primary btn-custom w-100 py-3 text-white mt-4 fs-5">Analyze Risk Now</button>
        </form>
    </div>
</body>
</html>
'''

       

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
