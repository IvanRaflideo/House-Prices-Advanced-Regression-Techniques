import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

st.set_page_config(page_title="Heart Disease Prediction - Logistic Regression", layout="wide")
st.title("🫀 Prediksi Penyakit Jantung (Logistic Regression)")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# ====== DATA ======
st.sidebar.header("Data")
use_upload = st.sidebar.checkbox("Upload CSV sendiri (opsional)", value=False)

if use_upload:
    up = st.sidebar.file_uploader("Upload heart.csv", type=["csv"])
    if up is None:
        st.info("Upload file CSV untuk lanjut.")
        st.stop()
    df = pd.read_csv(up)
else:
    df = load_data("data/heart.csv")

st.subheader("Preview Data")
st.dataframe(df.head(15), use_container_width=True)

# ====== TARGET ======
# Banyak versi dataset heart memakai kolom target bernama: "target"
if "target" not in df.columns:
    st.error("Kolom target 'target' tidak ditemukan. Pastikan file adalah Heart Disease-UCI (heart.csv).")
    st.stop()

# Bersihkan target jadi biner 0/1 (kalau sudah 0/1 ya aman)
y = df["target"].astype(int)
X = df.drop(columns=["target"])

# ====== Pilih fitur (opsional) ======
st.sidebar.header("Fitur")
default_features = X.columns.tolist()  # pakai semua fitur by default
selected_features = st.sidebar.multiselect(
    "Pilih fitur yang dipakai",
    options=X.columns.tolist(),
    default=default_features
)
if len(selected_features) == 0:
    st.warning("Pilih minimal 1 fitur.")
    st.stop()

X = X[selected_features].copy()

# ====== Split ======
st.sidebar.header("Training Config")
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=int(random_state),
    stratify=y
)

# ====== Pipeline: impute + scale + logistic ======
pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train)

# ====== Evaluasi ======
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

st.subheader("Hasil Evaluasi")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{acc:.3f}")
c2.metric("Precision", f"{prec:.3f}")
c3.metric("Recall", f"{rec:.3f}")
c4.metric("F1", f"{f1:.3f}")
c5.metric("ROC-AUC", f"{auc:.3f}")

st.write("Confusion Matrix (baris = aktual, kolom = prediksi):")
st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

# ====== Prediksi manual ======
st.subheader("Coba Prediksi 1 Pasien")

with st.form("predict_form"):
    input_data = {}
    for col in selected_features:
        # default: median kolom
        default_val = float(pd.to_numeric(df[col], errors="coerce").median())
        input_data[col] = st.number_input(col, value=default_val)

    submitted = st.form_submit_button("Prediksi")

if submitted:
    input_df = pd.DataFrame([input_data])
    prob = float(pipe.predict_proba(input_df)[0, 1])
    pred = int(prob >= 0.5)

    st.success(f"✅ Prediksi kelas: **{pred}** (Probabilitas sakit jantung ≈ **{prob:.2%}**)")
    st.caption("Catatan: ini model pembelajaran mesin untuk tugas/deployment, bukan diagnosis medis.")
