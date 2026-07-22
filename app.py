import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# 1. Load the trained pipeline (Preprocessor + Model)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'heart_disease_pipeline.joblib')

pipeline = None
if os.path.exists(MODEL_PATH):
    try:
        pipeline = joblib.load(MODEL_PATH)
        print("✅ Trained pipeline loaded successfully!")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")

@app.route('/')
def home():
    """Renders the main input form page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives form inputs, runs pipeline prediction, and returns results."""
    if pipeline is None:
        return jsonify({'error': 'Model pipeline is not loaded on server.'}), 500

    try:
        # Extract inputs from HTML form submission
        form_data = {
            'Age': float(request.form.get('Age', 0)),
            'Sex': request.form.get('Sex', 'M'),
            'ChestPainType': request.form.get('ChestPainType', 'ASY'),
            'RestingBP': float(request.form.get('RestingBP', 120)),
            'Cholesterol': float(request.form.get('Cholesterol', 200)),
            'FastingBS': int(request.form.get('FastingBS', 0)),
            'RestingECG': request.form.get('RestingECG', 'Normal'),
            'MaxHR': float(request.form.get('MaxHR', 150)),
            'ExerciseAngina': request.form.get('ExerciseAngina', 'N'),
            'Oldpeak': float(request.form.get('Oldpeak', 0.0)),
            'ST_Slope': request.form.get('ST_Slope', 'Flat')
        }

        # Convert input dictionary into DataFrame matching model features
        input_df = pd.DataFrame([form_data])

        # Run prediction & get probability score
        prediction = pipeline.predict(input_df)[0]
        
        # Get risk probability if classifier supports it
        probability = None
        if hasattr(pipeline, "predict_proba"):
            prob_scores = pipeline.predict_proba(input_df)[0]
            probability = round(prob_scores[1] * 100, 2)

        result_text = "High Risk of Heart Disease" if prediction == 1 else "Normal / Low Risk"
        result_class = "danger" if prediction == 1 else "success"

        return render_template(
            'index.html',
            prediction_text=result_text,
            result_class=result_class,
            probability=probability,
            inputs=form_data
        )

    except Exception as e:
        return render_template('index.html', error_text=f"Invalid Input Data: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
