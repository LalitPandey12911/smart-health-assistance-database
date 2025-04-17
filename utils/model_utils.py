import pickle
import os
from config import MODEL_DIR

def load_models():
    models = {
        "svc": pickle.load(open(os.path.join(MODEL_DIR, "svc.pkl"), "rb")),
        "rf": pickle.load(open(os.path.join(MODEL_DIR, "rf.pkl"), "rb")),
        "label_encoder": pickle.load(open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb"))
    }
    return models
