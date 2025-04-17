import os
import pandas as pd
from config import DATA_DIR
import logging

def safe_load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        logging.error(f"CSV file not found: {path}")
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)
