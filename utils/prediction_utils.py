import numpy as np
from textblob import TextBlob # type: ignore
from sklearn.metrics import accuracy_score
import logging
import pandas as pd

def correct_symptom(symptom):
    return str(TextBlob(symptom).correct()).lower()

def helper(dis, data):
    desc_df = data["description"]
    pre_df = data["precautions"]
    med_df = data["medications"]
    diet_df = data["diets"]
    workout_df = data["workout"]

    desc = " ".join(desc_df[desc_df['Disease'] == dis]['Description'].values)
    pre = pre_df[pre_df['Disease'] == dis].iloc[0, 1:].dropna().tolist()
    med = med_df[med_df['Disease'] == dis]['Medication'].tolist()
    die = diet_df[diet_df['Disease'] == dis]['Diet'].tolist()
    wrkout = workout_df[workout_df['disease'] == dis]['workout'].tolist()

    return desc, pre, med, die, wrkout

def predict_disease(symptoms_list, data, models):
    input_vector = np.zeros(len(data["training"].columns) - 1)
    unrecognized_symptoms = []

    for symptom in symptoms_list:
        corrected = correct_symptom(symptom.strip().lower())
        if corrected in data["training"].columns[:-1]:
            index = data["training"].columns.get_loc(corrected)
            input_vector[index] = 1
        else:
            unrecognized_symptoms.append(symptom)

    if unrecognized_symptoms:
        logging.warning(f"Unrecognized symptoms: {', '.join(unrecognized_symptoms)}")
        return f"Some symptoms were unrecognized: {', '.join(unrecognized_symptoms)}. Please check your input."

    input_df = pd.DataFrame([input_vector], columns=data["training"].columns[:-1])

    predictions = {}
    for model_key in ["svc", "rf"]:
        model = models[model_key]
        pred = model.predict(input_df)[0]
        predictions[model_key] = pred

    selected_model = "svc"
    selected_pred = predictions[selected_model]
    selected_model_name = "SVC"

    if predictions["svc"] != predictions["rf"]:
        acc_df = data["training"]
        X = acc_df.drop(columns=["prognosis"])
        y = acc_df["prognosis"]

        svc_acc = accuracy_score(y, models["svc"].predict(X))
        rf_acc = accuracy_score(y, models["rf"].predict(X))

        if rf_acc > svc_acc:
            selected_model = "rf"
            selected_pred = predictions["rf"]
            selected_model_name = "Random Forest"

    predicted_disease = models["label_encoder"].inverse_transform([selected_pred])[0]
    logging.info(f"Predicted using model [{selected_model_name}]: {predicted_disease}")
    return predicted_disease, selected_model_name
