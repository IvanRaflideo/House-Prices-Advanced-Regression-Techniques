import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "model/rf_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.set_page_config(page_title="House Price Prediction (RF Regression)", layout="centered")
st.title("🏠 Prediksi Harga Rumah - Random Forest Regression")
st.write("Prototype deployment ML dengan dataset Kaggle House Prices.")

model = load_model()

st.subheader("Input Fitur Rumah")

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("OverallQual (1-10)", 1, 10, 5)
    gr_liv_area = st.number_input("GrLivArea (luas bangunan, sqft)", min_value=200, max_value=6000, value=1500)
    year_built = st.number_input("YearBuilt", min_value=1800, max_value=2025, value=2000)
    garage_cars = st.number_input("GarageCars", min_value=0, max_value=6, value=2)

with col2:
    total_bsmt_sf = st.number_input("TotalBsmtSF (luas basement, sqft)", min_value=0, max_value=6000, value=800)
    full_bath = st.number_input("FullBath", min_value=0, max_value=5, value=2)
    bedroom_abvgr = st.number_input("BedroomAbvGr", min_value=0, max_value=10, value=3)
    neighborhood = st.text_input("Neighborhood (contoh: NAmes, CollgCr)", value="NAmes")

st.caption("Tips: Neighborhood harus sesuai kategori dataset. Kalau tidak yakin, pakai 'NAmes' atau 'CollgCr'.")

# Bentuk input sesuai nama kolom dataset
input_df = pd.DataFrame([{
    "OverallQual": overall_qual,
    "GrLivArea": gr_liv_area,
    "YearBuilt": year_built,
    "GarageCars": garage_cars,
    "TotalBsmtSF": total_bsmt_sf,
    "FullBath": full_bath,
    "BedroomAbvGr": bedroom_abvgr,
    "Neighborhood": neighborhood,
}])

st.write("### Data Input")
st.dataframe(input_df, use_container_width=True)

if st.button("Prediksi Harga"):
    pred = model.predict(input_df)[0]
    st.success(f"Perkiraan Harga Rumah: **${pred:,.0f}**")
    st.info("Ini hasil prototype. Akurasi bisa ditingkatkan dengan tuning + fitur input lebih lengkap.")
