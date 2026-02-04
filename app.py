# app.py
"""
Streamlit app untuk prediksi diabetes sesuai notebook Diabetes_LR_vs_RF.ipynb.

Notebook melatih beberapa model (LogReg & RF), memilih yang terbaik berdasarkan F1,
lalu menyimpan pipeline sklearn ke file joblib. App ini membaca file itu dan
menyediakan UI untuk input fitur + output probabilitas & kelas.

Struktur folder (sederhana, TANPA folder "model/"):
project/
  app.py
  model.joblib

Jalankan:
  pip install -r requirements.txt
  streamlit run app.py

Env opsional:
  MODEL_PATH : override lokasi model.joblib
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


def resolve_model_path() -> str:
    """Baca model.joblib sejajar file app, atau via env MODEL_PATH."""
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        return env_path

    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "model.joblib")
    return path


MODEL_PATH = resolve_model_path()


@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model tidak ditemukan di: {path}\n\n"
            "Solusi:\n"
            "1) Jalankan notebook sampai cell 'joblib.dump(...)' menghasilkan 'model.joblib'.\n"
            "2) Pastikan 'model.joblib' berada SEFOLDER dengan app.py.\n"
            "   Struktur minimal:\n"
            "     project/\n"
            "       app.py\n"
            "       model.joblib\n\n"
            "Opsional: set environment variable MODEL_PATH jika lokasi berbeda."
        )
    return joblib.load(path)


def get_expected_features(model) -> List[str]:
    """
    Ambil nama fitur mentah yang diharapkan pipeline.
    Notebook menggunakan pandas DataFrame + ColumnTransformer,
    jadi pipeline biasanya punya attribute feature_names_in_.
    """
    feats = getattr(model, "feature_names_in_", None)
    if feats is not None:
        return list(feats)

    # Fallback jika attribute tidak ada (umumnya 8 fitur dataset Kaggle)
    return [
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "smoking_history",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
    ]


def get_model_step(model):
    """Ambil estimator terakhir (step 'model') jika pipeline, kalau tidak ya model itu sendiri."""
    try:
        named_steps = getattr(model, "named_steps", None)
        if named_steps and "model" in named_steps:
            return named_steps["model"]
    except Exception:
        pass
    return model


def extract_categorical_options(model) -> Dict[str, List[str]]:
    """
    Coba ambil opsi kategori dari OneHotEncoder yang tersimpan di pipeline preprocessing.
    Ini membuat dropdown persis sesuai kategori saat training.
    Jika gagal, kembalikan dict kosong.
    """
    options: Dict[str, List[str]] = {}
    try:
        preprocess = model.named_steps.get("preprocess", None)
        if preprocess is None:
            return options

        # ColumnTransformer fitted menyimpan transformers_ (nama, transformer, columns)
        transformers = getattr(preprocess, "transformers_", None)
        if not transformers:
            return options

        for name, trans, cols in transformers:
            if name != "cat":
                continue

            # 'trans' biasanya Pipeline(imputer, onehot)
            onehot = getattr(trans, "named_steps", {}).get("onehot", None)
            if onehot is None:
                return options

            cats = getattr(onehot, "categories_", None)
            if cats is None:
                return options

            # cols bisa list nama kolom kategorikal
            if isinstance(cols, (list, tuple)) and len(cols) == len(cats):
                for col, cat_values in zip(cols, cats):
                    options[str(col)] = [str(x) for x in list(cat_values)]
        return options
    except Exception:
        return options


@st.cache_data
def build_model_info(_model) -> Dict[str, Any]:
    """Info ringkas model + (opsional) top coef/importance seperti notebook."""
    info: Dict[str, Any] = {}
    est = get_model_step(_model)
    info["estimator"] = est.__class__.__name__

    # Coba ambil feature names setelah one-hot, untuk interpretasi
    try:
        preprocess = _model.named_steps.get("preprocess", None)
        if preprocess is not None and hasattr(preprocess, "get_feature_names_out"):
            info["feature_names_out"] = list(preprocess.get_feature_names_out())
    except Exception:
        pass

    # Koefisien untuk Logistic Regression
    try:
        if hasattr(est, "coef_") and "feature_names_out" in info:
            coefs = np.array(est.coef_).ravel()
            fn = info["feature_names_out"]
            df = pd.DataFrame({"feature": fn, "value": coefs})
            df["abs"] = df["value"].abs()
            info["top_coef"] = df.sort_values("abs", ascending=False).head(15).drop(columns=["abs"])
    except Exception:
        pass

    # Feature importance untuk RandomForest
    try:
        if hasattr(est, "feature_importances_") and "feature_names_out" in info:
            imp = np.array(est.feature_importances_).ravel()
            fn = info["feature_names_out"]
            df = pd.DataFrame({"feature": fn, "value": imp})
            info["top_importance"] = df.sort_values("value", ascending=False).head(15)
    except Exception:
        pass

    return info


def predict_probability(model, X: pd.DataFrame) -> float:
    """Probabilitas kelas positif (diabetes=1)."""
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[:, 1][0])

    # fallback untuk model tanpa predict_proba
    score = float(model.decision_function(X)[0])
    return float(1 / (1 + np.exp(-score)))


def widget_for_feature(
    name: str,
    cat_options: Dict[str, List[str]],
) -> Any:
    """Buat widget input yang sesuai dengan tipe fitur umum pada dataset."""
    label = name.replace("_", " ").title()

    # categorical (pakai opsi dari encoder kalau ada)
    if name in cat_options:
        opts = cat_options[name]
        # fallback index aman
        idx = 0
        # prefer nilai yang sering muncul di dataset Kaggle
        preferred = {"gender": "Female", "smoking_history": "never"}
        if name in preferred and preferred[name] in opts:
            idx = opts.index(preferred[name])
        return st.selectbox(label, options=opts, index=idx)

    # binary flag umum
    if name in {"hypertension", "heart_disease"}:
        return st.selectbox(label, options=[0, 1], index=0)

    # numerik umum
    if name == "age":
        return st.number_input(label, min_value=0, max_value=120, value=45, step=1)

    if name == "bmi":
        return st.number_input(label, min_value=0.0, max_value=80.0, value=27.5, step=0.1, format="%.2f")

    if name == "HbA1c_level":
        return st.number_input(label, min_value=0.0, max_value=20.0, value=6.2, step=0.1, format="%.2f")

    if name == "blood_glucose_level":
        return st.number_input(label, min_value=0.0, max_value=500.0, value=140.0, step=1.0, format="%.0f")

    # fallback text
    return st.text_input(label, value="")


def main():
    st.set_page_config(page_title="Prediksi Diabetes", page_icon="🩺", layout="centered")

    st.title("🩺 Prediksi Diabetes")
    st.caption("Aplikasi Streamlit untuk deployment model dari notebook (LogReg vs RF).")

    # Sidebar pengaturan
    with st.sidebar:
        st.subheader("Pengaturan")
        threshold = st.slider("Threshold klasifikasi", 0.05, 0.95, 0.50, 0.01)
        

        st.markdown(
            "Tips:\n"
            "- Threshold dapat diubah untuk menyeimbangkan precision/recall (dataset imbalanced)."
        )

    # Load model
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(str(e))
        st.stop()

    expected = get_expected_features(model)
    cat_options = extract_categorical_options(model)

    # Input form
    st.subheader("Input Fitur")
    with st.form("input_form"):
        inputs: Dict[str, Any] = {}
        cols = st.columns(2)
        for i, feat in enumerate(expected):
            with cols[i % 2]:
                inputs[feat] = widget_for_feature(feat, cat_options)

        submitted = st.form_submit_button("Prediksi")

    if submitted:
        X = pd.DataFrame([inputs], columns=expected)

        try:
            proba = predict_probability(model, X)
            pred = int(proba >= threshold)
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")
            st.stop()

        st.subheader("Hasil Prediksi")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Kelas", "Diabetes (1)" if pred == 1 else "Non-diabetes (0)")
        with c2:
            st.metric("Probabilitas diabetes", f"{proba*100:.2f}%")

        st.progress(min(max(proba, 0.0), 1.0))
        st.caption(f"Threshold: {threshold:.2f}")

        with st.expander("Lihat input (debug)"):
            st.json(inputs)

        # Info model + interpretasi (opsional)
        info = build_model_info(model)
        with st.expander("Info Model (opsional)"):
            st.write("Estimator:", info.get("estimator", "Unknown"))
            st.write("Jumlah fitur input:", len(expected))
            if "top_coef" in info:
                st.write("Top koefisien (Logistic Regression):")
                st.dataframe(info["top_coef"], use_container_width=True)
            if "top_importance" in info:
                st.write("Top feature importance (Random Forest):")
                st.dataframe(info["top_importance"], use_container_width=True)

        st.info("Catatan: ini alat bantu prediksi berbasis ML untuk tujuan akademik, bukan diagnosis medis.")


if __name__ == "__main__":
    main()
