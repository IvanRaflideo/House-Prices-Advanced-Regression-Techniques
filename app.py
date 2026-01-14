import joblib
import pandas as pd
import streamlit as st
import os

MODEL_PATH = "model/rf_model.joblib"
DATA_PATH = "train.csv"

st.set_page_config(page_title="Prediksi Harga Rumah", layout="centered")

# ======================
# LOAD FEATURE SCHEMA
# ======================
@st.cache_data
def load_feature_columns():
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=["SalePrice"]).columns.tolist()

FEATURE_COLS = load_feature_columns()

# ======================
# LOAD MODEL ONLY (NO TRAINING!)
# ======================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model belum ada. Pastikan rf_model.joblib sudah diupload.")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# ======================
# UI
# ======================
st.title("🏠 Prediksi Harga

