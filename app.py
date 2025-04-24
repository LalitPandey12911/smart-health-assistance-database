# === Imports ===
from flask import Flask, send_file, render_template, request, redirect, url_for, flash, session
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import pickle
import os
import pandas as pd
import numpy as np
from textblob import TextBlob
import logging
from sklearn.metrics import accuracy_score
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from flask_migrate import Migrate

app = Flask(__name__)
load_dotenv()

app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL", "your_default_db_url")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.urandom(24)

logging.basicConfig(level=logging.INFO)

db = SQLAlchemy(app)
migrate = Migrate(app, db)

MODEL_DIR = 'models'
DATA_DIR = 'data'
last_prediction_data = {}
last_disease_probs = {}

def safe_load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)

def load_models():
    return {
        "rf": pickle.load(open(os.path.join(MODEL_DIR, "rf.pkl"), "rb")),
        "xgb": pickle.load(open(os.path.join(MODEL_DIR, "xgb.pkl"), "rb")),
        "label_encoder": pickle.load(open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb"))
    }

# === Load Dataset ===
def load_data():
    data = {
        "description": safe_load_csv('description.csv'),
        "precautions": safe_load_csv('precautions_df.csv'),
        "medications": safe_load_csv('medications.csv'),
        "diets": safe_load_csv('diets.csv'),
        "workout": safe_load_csv('workout_df.csv'),
        "symptoms": safe_load_csv('symtoms_df.csv'),
        "training": safe_load_csv("Training.csv")
    }
    for key in data:
        if 'Disease' in data[key].columns:
            data[key]['Disease'] = data[key]['Disease'].str.strip()
    return data

models = load_models()
data = load_data()
training_data = data["training"]
all_symptoms = training_data.columns[:-1].tolist()

class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(500), nullable=False)

def correct_symptom(symptom):
    return str(TextBlob(symptom).correct()).lower()

def predict_disease(symptoms_list):
    input_vector = np.zeros(len(all_symptoms))
    unrecognized = []

    for s in symptoms_list:
        corrected = correct_symptom(s)
        if corrected in all_symptoms:
            index = all_symptoms.index(corrected)
            input_vector[index] = 1
        else:
            unrecognized.append(s)

    if unrecognized:
        logging.warning(f"Unrecognized symptoms: {', '.join(unrecognized)}")

    df_input = pd.DataFrame([input_vector], columns=all_symptoms)

    label_encoder = models['label_encoder']
    y_true = label_encoder.transform(training_data['prognosis'])

    rf_acc = accuracy_score(y_true, models['rf'].predict(training_data[all_symptoms]))
    xgb_acc = accuracy_score(y_true, models['xgb'].predict(training_data[all_symptoms]))
    selected_model = 'xgb' if xgb_acc > rf_acc else 'rf'
    model_used = "XGBoost" if selected_model == 'xgb' else "Random Forest"
    model = models[selected_model]

    pred = model.predict(df_input)[0]
    predicted_disease = label_encoder.inverse_transform([pred])[0]

    probs = model.predict_proba(df_input)[0]
    top_probs = dict(sorted(zip(model.classes_, probs), key=lambda x: x[1], reverse=True)[:4])
    decoded_probs = {
        label_encoder.inverse_transform([int(cls)])[0]: round(prob, 4)
        for cls, prob in top_probs.items()
    }

    return predicted_disease, model_used, decoded_probs

def get_disease_info(disease):
    disease = disease.strip()
    desc = data['description'].set_index('Disease').loc[disease, 'Description']
    pre = data['precautions'][data['precautions']['Disease'] == disease].iloc[0, 1:].dropna().tolist()
    med = data['medications'][data['medications']['Disease'] == disease]['Medication'].tolist()
    diet = data['diets'][data['diets']['Disease'] == disease]['Diet'].tolist()
    workout = data['workout'][data['workout']['disease'] == disease]['workout'].tolist()
    return desc, pre, med, diet, workout


@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_count = User.query.count()
    return render_template('index.html', user_count=user_count)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            flash('Username already taken.', 'warning')
            return redirect(url_for('signup'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'warning')
            return redirect(url_for('signup'))

        hashed_pw = generate_password_hash(password)
        new_user = User(username=username, email=email, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()

        session['user_id'] = new_user.id
        flash('Signup successful!', 'success')
        return redirect('/login')

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id  
            flash('Login successful!', 'success')
            return redirect('/')
        else:
            flash('Invalid credentials.', 'danger')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()  
    flash('Logged out successfully!', 'success') 
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    global last_prediction_data, last_disease_probs
    name = request.form['name']
    age = request.form['age']
    location = request.form['location']
    symptoms_raw = request.form['symptoms']
    symptoms = [s.strip() for s in symptoms_raw.split(',') if s.strip()]

    if len(symptoms) < 3:
        return render_template('index.html', message="Please enter at least 3 symptoms.")

    result = predict_disease(symptoms)
    if isinstance(result, str):
        return render_template('index.html', message=result)

    predicted_disease, model_used, disease_probs = result
    dis_des, pre, med, diet, workout = get_disease_info(predicted_disease)

    last_prediction_data = {
        "Name": name,
        "Age": age,
        "Location": location,
        "Symptoms": ", ".join(symptoms),
        "Disease": predicted_disease,
        "Description": dis_des,
        "Precautions": ", ".join(pre),
        "Medications": ", ".join(med),
        "Diet Recommendations": ", ".join(diet),
        "Workouts": ", ".join(workout),
        "Model Used": model_used
    }
    last_disease_probs = disease_probs

    return redirect(url_for('result'))

@app.route('/result')
def result():
    if not last_prediction_data:
        return "No prediction data available.", 400
    return render_template('results.html',
                           predicted_disease=last_prediction_data["Disease"],
                           dis_des=last_prediction_data["Description"],
                           my_precautions=last_prediction_data["Precautions"].split(', '),
                           medications=last_prediction_data["Medications"].split(', '),
                           my_diet=last_prediction_data["Diet Recommendations"].split(', '),
                           workout=last_prediction_data["Workouts"].split(', '),
                           model_used=last_prediction_data["Model Used"],
                           disease_probs=last_disease_probs)

@app.route('/probability_chart')
def probability_chart():
    if not last_disease_probs:
        return "No probability data to display.", 400
    return render_template('probability_chart.html', disease_probs=last_disease_probs)

@app.route('/download_pdf')
def download_pdf():
    if not last_prediction_data:
        return "No prediction data available to generate a PDF.", 400

    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    y = 750
    for key, value in last_prediction_data.items():
        c.drawString(100, y, f"{key}: {value}")
        y -= 15
    c.save()

    pdf_buffer.seek(0)
    return send_file(pdf_buffer, as_attachment=True, download_name="prediction.pdf")

@app.route('/medications')
def medications():
    return render_template('medications.html', medications=data['medications'])

@app.route('/precautions')
def precautions():
    return render_template('precautions.html', precautions=data['precautions'])

@app.route('/workout')
def workout():
    return render_template('workout.html', workout=data['workout'])

@app.route('/diet')
def diet():
    return render_template('diet.html', diets=data['diets'])

@app.route('/developer')
def developer():
    return render_template('developer.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        flash("Thanks for contacting us!", "success")
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/privacy-policy')
def privacy_policy():
    return render_template('privacy-policy.html')

if __name__ == '__main__':
    # Dynamically set the port from the environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app on host '0.0.0.0' (to be accessible externally), with dynamic port
    app.run(debug=False, host='0.0.0.0', port=port)

