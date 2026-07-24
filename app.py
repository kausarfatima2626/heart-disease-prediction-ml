import os
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model pipeline
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'heart_disease_pipeline.joblib')

pipeline = None
if os.path.exists(MODEL_PATH):
    pipeline = joblib.load(MODEL_PATH)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    probability = None
    error = None

    if request.method == 'POST':
        try:
            if pipeline is None:
                raise Exception("Model file not found. Please train the model first.")

            # Form data extraction
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
            pred = pipeline.predict(input_df)[0]
            prob = pipeline.predict_proba(input_df)[0][1]

            prediction = "High Risk" if pred == 1 else "Low Risk"
            probability = f"{round(prob * 100, 2)}%"

        except Exception as e:
            error = str(e)

    return render_template('index.html', prediction=prediction, probability=probability, error=error)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
