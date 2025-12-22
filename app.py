import os
import joblib
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# ======================
# PATH CONFIG
# ======================
MODEL_PATH = "model/rf_model.joblib"
DATA_PATH = "train.csv"

st.set_page_config(page_title="Prediksi Harga Rumah (RF)", layout="centered")

# ======================
# DATA SCHEMA (FEATURE COLUMNS)
# ======================
@st.cache_data
def get_feature_columns():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "train.csv tidak ditemukan. Upload file train.csv dari Kaggle ke folder yang sama dengan app.py."
        )
    df = pd.read_csv(DATA_PATH)
    if "SalePrice" not in df.columns:
        raise ValueError("Kolom target 'SalePrice' tidak ditemukan di train.csv.")
    return df.drop(columns=["SalePrice"]).columns.tolist()

FEATURE_COLS = get_feature_columns()

# ======================
# LOAD OR TRAIN MODEL
# ======================
@st.cache_resource
def load_or_train_model():
    # Load model if exists
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    # Train model if not exists
    df = pd.read_csv(DATA_PATH)
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop"
    )

    rf_model = RandomForestRegressor(
        n_estimators=350,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", rf_model),
    ])

    pipeline.fit(X, y)

    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    return pipeline

# ======================
# APP UI
# ======================
st.title("🏠 Prediksi Harga Rumah - Random Forest Regression")
st.write("Prototype deployment ML menggunakan dataset Kaggle **House Prices – Advanced Regression Techniques**.")

with st.expander("📌 Info Model", expanded=False):
    st.write("- Algoritma: Random Forest Regression")
    st.write("- Target: `SalePrice`")
    st.write("- Model otomatis dilatih pada first run jika file model belum ada.")
    st.write(f"- Total fitur training: **{len(FEATURE_COLS)}** kolom (input kamu akan diisi NaN untuk kolom lain).")

model = load_or_train_model()

st.subheader("Masukkan Data Rumah (beberapa fitur utama)")

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("OverallQual (1-10)", 1, 10, 5)
    gr_liv_area = st.number_input("GrLivArea (sqft)", min_value=200, max_value=6000, value=1500)
    year_built = st.number_input("YearBuilt", min_value=1800, max_value=2025, value=2000)
    garage_cars = st.number_input("GarageCars", min_value=0, max_value=6, value=2)

with col2:
    total_bsmt_sf = st.number_input("TotalBsmtSF (sqft)", min_value=0, max_value=6000, value=800)
    full_bath = st.number_input("FullBath", min_value=0, max_value=5, value=2)
    bedroom_abvgr = st.number_input("BedroomAbvGr", min_value=0, max_value=10, value=3)
    neighborhood = st.text_input("Neighborhood (contoh: NAmes, CollgCr)", value="NAmes")

st.caption("Kolom lain yang tidak kamu isi akan otomatis dianggap kosong (NaN) dan di-handle oleh imputer di pipeline.")

# Input minimal dari user
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

# 🔥 INI KUNCI FIX: samakan kolom input dengan kolom training
input_df = input_min.reindex(columns=FEATURE_COLS)

st.write("### Preview Input (subset yang kamu isi)")
st.dataframe(input_min, use_container_width=True)

if st.button("Prediksi Harga"):
    pred = model.predict(input_df)[0]
    st.success(f"Perkiraan Harga Rumah: **${pred:,.0f}**")
