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
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>CardioSense — Heart Disease Risk Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root{
    --navy-950:#0b1524;
    --paper:#f6f4ef;
    --coral:#f2545b;
    --coral-dim:#f2545b22;
    --teal:#3fbf9f;
    --teal-dim:#3fbf9f22;
    --ink:#101418;
    --ink-soft:#5a6270;
    --line:#dedad0;
    --radius:16px;
  }
  *{box-sizing:border-box;}
  html{scroll-behavior:smooth;}
  body{
    margin:0;
    background:var(--paper);
    color:var(--ink);
    font-family:'Inter',sans-serif;
    -webkit-font-smoothing:antialiased;
  }

  header.hero{
    background:radial-gradient(120% 140% at 15% 0%, #1a2c47 0%, var(--navy-950) 60%);
    color:var(--paper);
    padding:7vh 6vw 9vh;
    text-align:center;
  }
  .eyebrow{
    font-family:'IBM Plex Mono',monospace; font-size:.78rem; letter-spacing:.14em;
    text-transform:uppercase; color:var(--teal); margin-bottom:16px;
    display:flex; align-items:center; justify-content:center; gap:10px;
  }
  .eyebrow::before{
    content:""; width:8px; height:8px; border-radius:50%; background:var(--coral);
    animation:pulse-dot 1.8s infinite;
  }
  @keyframes pulse-dot{
    0%{box-shadow:0 0 0 0 #f2545b66;}
    70%{box-shadow:0 0 0 10px #f2545b00;}
    100%{box-shadow:0 0 0 0 #f2545b00;}
  }
  h1.headline{
    font-family:'Fraunces',serif; font-weight:600;
    font-size:clamp(1.9rem,4vw,2.9rem); line-height:1.12; margin:0 auto 14px; max-width:18ch;
  }
  h1.headline em{ font-style:normal; color:var(--coral); }
  .hero p.sub{
    font-size:1rem; line-height:1.6; color:#c8cedb; max-width:46ch; margin:0 auto;
  }
  .ecg-wrap{
    max-width:640px; margin:34px auto 0; height:110px;
    border:1px solid #ffffff1c; border-radius:var(--radius);
    background:linear-gradient(180deg,#0f1e33,#0c1830);
    overflow:hidden;
  }
  .ecg-wrap svg{ width:100%; height:100%; }
  .ecg-line{
    fill:none; stroke:var(--teal); stroke-width:2.4; stroke-linecap:round; stroke-linejoin:round;
    stroke-dasharray:1400; stroke-dashoffset:1400;
    animation:draw 3.2s ease-in-out infinite;
    filter:drop-shadow(0 0 6px #3fbf9f88);
  }
  @keyframes draw{
    0%{ stroke-dashoffset:1400; }
    55%{ stroke-dashoffset:0; }
    100%{ stroke-dashoffset:-1400; }
  }

  main{
    max-width:760px; margin:-50px auto 0; padding:0 6vw 10vh; position:relative;
  }
  .card{
    background:#fff; border:1px solid var(--line); border-radius:var(--radius);
    box-shadow:0 24px 60px -30px rgba(15,30,51,.35);
    padding:40px 36px;
  }
  .card h2{
    font-family:'Fraunces',serif; font-weight:600; font-size:1.4rem; margin:0 0 6px;
  }
  .card > p.lead{ color:var(--ink-soft); margin:0 0 30px; font-size:.92rem; }

  .grid-fields{
    display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:18px; margin-bottom:28px;
  }
  .field label{
    display:block; font-size:.8rem; font-weight:600; color:var(--ink); margin-bottom:6px;
  }
  .field input, .field select{
    width:100%; padding:11px 13px; border:1px solid var(--line); border-radius:10px;
    font-family:'IBM Plex Mono',monospace; font-size:.88rem; color:var(--ink);
    background:var(--paper); transition:border-color .2s, box-shadow .2s;
  }
  .field input:focus, .field select:focus{
    outline:none; border-color:var(--teal); box-shadow:0 0 0 3px var(--teal-dim);
  }

  button.predict-btn{
    background:var(--navy-950); color:#fff; border:none; cursor:pointer; width:100%;
    padding:15px 30px; border-radius:999px; font-weight:600; font-size:.95rem;
    font-family:'Inter',sans-serif; transition:transform .2s ease, background .2s ease;
  }
  button.predict-btn:hover{ background:var(--coral); transform:translateY(-2px); }

  .result{
    margin-bottom:26px; padding:24px 26px; border-radius:14px;
    display:flex; align-items:center; gap:16px; border:1px solid var(--line);
  }
  .result.low{ background:var(--teal-dim); border-color:#3fbf9f55; }
  .result.high{ background:var(--coral-dim); border-color:#f2545b55; }
  .result .badge{
    width:44px; height:44px; border-radius:50%; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; font-size:1.2rem; color:#fff;
  }
  .result.low .badge{ background:var(--teal); }
  .result.high .badge{ background:var(--coral); }
  .result h3{ margin:0 0 4px; font-family:'Fraunces',serif; font-size:1.05rem; }
  .result p{ margin:0; color:var(--ink-soft); font-size:.86rem; }

  .error-box{
    margin-bottom:26px; padding:16px 20px; border-radius:12px;
    background:var(--coral-dim); border:1px solid #f2545b55; color:#9c2a30; font-size:.86rem;
  }

  footer{
    text-align:center; padding:28px 6vw; color:var(--ink-soft); font-size:.8rem;
  }
</style>
</head>
<body>

<header class="hero">
  <div class="eyebrow">ML-powered · Instant result</div>
  <h1 class="headline">Read your heart's <em>signal</em>, before it becomes a symptom.</h1>
  <p class="sub">Enter a few clinical values and a trained model estimates your risk of heart disease in seconds.</p>
  <div class="ecg-wrap">
    <svg viewBox="0 0 400 110" preserveAspectRatio="none">
      <path class="ecg-line" d="M0,55 L60,55 L80,55 L95,25 L110,85 L125,10 L140,95 L155,55 L180,55 L220,55 L235,40 L250,70 L265,55 L400,55" />
    </svg>
  </div>
</header>

<main>
  <div class="card">
    <h2>Enter your clinical details</h2>
    <p class="lead">All fields are used only to generate your prediction — nothing is stored.</p>

    {% if error %}
    <div class="error-box">{{ error }}</div>
    {% endif %}

    {% if prediction %}
    <div class="result {{ 'high' if prediction == 'High Risk' else 'low' }}">
      <div class="badge">{{ '⚠' if prediction == 'High Risk' else '✓' }}</div>
      <div>
        <h3>Prediction: {{ prediction }}</h3>
        <p>Confidence level: {{ probability }} — this is a model estimate, not a medical diagnosis.</p>
      </div>
    </div>
    {% endif %}

    <form method="POST">
      <div class="grid-fields">
        <div class="field">
          <label>Age</label>
          <input type="number" name="Age" value="50" required>
        </div>
        <div class="field">
          <label>Sex</label>
          <select name="Sex" required>
            <option value="M">Male</option>
            <option value="F">Female</option>
          </select>
        </div>
        <div class="field">
          <label>Chest pain type</label>
          <select name="ChestPainType" required>
            <option value="TA">Typical angina (TA)</option>
            <option value="ATA">Atypical angina (ATA)</option>
            <option value="NAP">Non-anginal pain (NAP)</option>
            <option value="ASY" selected>Asymptomatic (ASY)</option>
          </select>
        </div>
        <div class="field">
          <label>Resting BP (mm Hg)</label>
          <input type="number" name="RestingBP" value="140" required>
        </div>
        <div class="field">
          <label>Cholesterol (mg/dl)</label>
          <input type="number" name="Cholesterol" value="280" required>
        </div>
        <div class="field">
          <label>Fasting blood sugar &gt;120 mg/dl</label>
          <select name="FastingBS" required>
            <option value="0">No</option>
            <option value="1">Yes</option>
          </select>
        </div>
        <div class="field">
          <label>Resting ECG</label>
          <select name="RestingECG" required>
            <option value="Normal" selected>Normal</option>
            <option value="ST">ST-T wave abnormality</option>
            <option value="LVH">Left ventricular hypertrophy</option>
          </select>
        </div>
        <div class="field">
          <label>Max heart rate</label>
          <input type="number" name="MaxHR" value="150" required>
        </div>
        <div class="field">
          <label>Exercise induced angina</label>
          <select name="ExerciseAngina" required>
            <option value="N">No</option>
            <option value="Y">Yes</option>
          </select>
        </div>
        <div class="field">
          <label>Oldpeak (ST depression)</label>
          <input type="number" step="0.1" name="Oldpeak" value="1.0" required>
        </div>
        <div class="field">
          <label>ST slope</label>
          <select name="ST_Slope" required>
            <option value="Up">Upsloping</option>
            <option value="Flat" selected>Flat</option>
            <option value="Down">Downsloping</option>
          </select>
        </div>
      </div>

      <button type="submit" class="predict-btn">Check my risk →</button>
    </form>
  </div>
</main>

<footer>
  Built by Kausar Fatima
</footer>

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
