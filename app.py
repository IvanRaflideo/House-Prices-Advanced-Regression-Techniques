import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Prediksi Harga Rumah", layout="centered")

MODEL_PATH = "model/rf_model.joblib"
DATA_PATH = "train.csv"

# ======================
# LOAD FEATURE COLUMNS
# ======================
@st.cache_data(show_spinner=False)
def load_feature_columns():
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=["SalePrice"]).columns.tolist()

FEATURE_COLS = load_feature_columns()

# ======================
# LOAD MODEL
# ======================
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ======================
# FORCE UI RENDER
# ======================
st.markdown("## Prediksi Harga Rumah")
st.write("Masukkan spesifikasi rumah")

st.markdown("---")

# ======================
# INPUT FORM
# ======================
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        overall_qual = st.slider("OverallQual", 1, 10, 5)
        gr_liv_area = st.number_input("GrLivArea (sqft)", 200, 6000, 1500)
        year_built = st.number_input("YearBuilt", 1800, 2025, 2000)
        garage_cars = st.number_input("GarageCars", 0, 6, 2)

    with col2:
        total_bsmt_sf = st.number_input("TotalBsmtSF", 0, 6000, 800)
        full_bath = st.number_input("FullBath", 0, 5, 2)
        bedroom_abvgr = st.number_input("BedroomAbvGr", 0, 10, 3)
        neighborhood = st.text_input("Neighborhood", "NAmes")

    submitted = st.form_submit_button("Prediksi Harga")

# ======================
# PREDICTION
# ======================
if submitted:
    input_min = pd.DataFrame([{
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "YearBuilt": year_built,
        "GarageCars": garage_cars,
        "TotalBsmtSF": total_bsmt_sf,
        "FullBath": full_bath,
        "BedroomAbvGr": bedroom_abvgr,
        "Neighborhood": neighborhood,
    }])

    input_df = input_min.reindex(columns=FEATURE_COLS)
    pred = model.predict(input_df)[0]

    st.success(f"Perkiraan Harga Rumah: ${pred:,.0f}")
