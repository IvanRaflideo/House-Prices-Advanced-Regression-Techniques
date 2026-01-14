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
# LOAD FEATURE COLUMNS
# ======================
@st.cache_data
def get_feature_columns():
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=["SalePrice"]).columns.tolist()

FEATURE_COLS = get_feature_columns()

# ======================
# LOAD / TRAIN MODEL
# ======================
@st.cache_resource
def load_or_train_model():
    os.makedirs("model", exist_ok=True)

    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    model = Pipeline([
        ("prep", preprocess),
        ("rf", RandomForestRegressor(
            n_estimators=350,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    return model

model = load_or_train_model()

# ======================
# UI
# ======================
st.title("🏠 Prediksi Harga Rumah - Random Forest")
st.caption("House Prices Kaggle Dataset")

col1, col2 = st.columns(2)

with col1:
    OverallQual = st.slider("OverallQual", 1, 10, 5)
    GrLivArea = st.number_input("GrLivArea", 200, 6000, 1500)
    YearBuilt = st.number_input("YearBuilt", 1800, 2025, 2000)
    GarageCars = st.number_input("GarageCars", 0, 6, 2)

with col2:
    TotalBsmtSF = st.number_input("TotalBsmtSF", 0, 6000, 800)
    FullBath = st.number_input("FullBath", 0, 5, 2)
    BedroomAbvGr = st.number_input("BedroomAbvGr", 0, 10, 3)
    Neighborhood = st.text_input("Neighborhood", "NAmes")

# ======================
# CREATE FULL FEATURE ROW
# ======================
user_input = pd.DataFrame([{
    "OverallQual": OverallQual,
    "GrLivArea": GrLivArea,
    "YearBuilt": YearBuilt,
    "GarageCars": GarageCars,
    "TotalBsmtSF": TotalBsmtSF,
    "FullBath": FullBath,
    "BedroomAbvGr": BedroomAbvGr,
    "Neighborhood": Neighborhood
}])

# align kolom training
input_df = user_input.reindex(columns=FEATURE_COLS)

st.dataframe(user_input, use_container_width=True)

if st.button("Prediksi Harga"):
    price = model.predict(input_df)[0]
    st.success(f"💰 Estimasi Harga Rumah: **${price:,.0f}**")
