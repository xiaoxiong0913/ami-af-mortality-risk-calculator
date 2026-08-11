from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


FEATURES = (
    "age",
    "hemoglobin_g_l",
    "creatinine_umol_l",
    "stemi",
    "wbc_10e9_l",
    "platelets_10e9_l",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified_model():
    assets = Path(__file__).resolve().parent / "model"
    model_path = assets / "mimic_ami_af_selected_six_stemi_model.joblib"
    freeze_path = assets / "mimic_ami_af_selected_six_stemi_model_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen" or tuple(freeze.get("features", [])) != FEATURES:
        raise RuntimeError("The frozen model record has an unexpected feature contract.")
    if sha256_file(model_path) != freeze.get("model_sha256"):
        raise RuntimeError("The packaged model does not match its freeze record.")
    model = joblib.load(model_path)
    if tuple(getattr(model, "feature_names_in_", ())) != FEATURES:
        raise RuntimeError("The packaged model has an unexpected feature contract.")
    return model, freeze


@st.cache_resource(show_spinner=False)
def cached_model():
    return load_verified_model()


def case_frame(age, hemoglobin, creatinine, stemi, wbc, platelets) -> pd.DataFrame:
    return pd.DataFrame(
        [[float(age), float(hemoglobin), float(creatinine), int(stemi == "Yes"), float(wbc), float(platelets)]],
        columns=list(FEATURES),
    )


def main() -> None:
    st.set_page_config(page_title="AMI-AF mortality risk", page_icon="♥", layout="wide")
    st.title("AMI-AF one-year mortality risk")
    st.write("Enter the six routinely available predictors to obtain the model-estimated one-year mortality risk.")
    try:
        model, freeze = cached_model()
    except Exception as exc:
        st.error(f"Model verification failed: {exc}")
        st.stop()

    entry, result = st.columns((1.2, 0.8), gap="large")
    with entry:
        with st.form("six_predictor_form"):
            age = st.number_input("Age (years)", min_value=18.0, max_value=110.0, value=65.0, step=1.0)
            hemoglobin = st.number_input("Hemoglobin (g/L)", min_value=40.0, max_value=220.0, value=135.0, step=1.0)
            creatinine = st.number_input("Creatinine (µmol/L)", min_value=20.0, max_value=1500.0, value=90.0, step=1.0)
            stemi = st.selectbox("STEMI", ("No", "Yes"))
            wbc = st.number_input("White blood cell count (10^9/L)", min_value=0.1, max_value=100.0, value=8.0, step=0.1)
            platelets = st.number_input("Platelet count (10^9/L)", min_value=1.0, max_value=1000.0, value=220.0, step=1.0)
            calculate = st.form_submit_button("Calculate predicted risk", type="primary")
    with result:
        st.subheader("Estimated risk")
        if calculate:
            probability = float(model.predict_proba(case_frame(age, hemoglobin, creatinine, stemi, wbc, platelets))[:, 1][0])
            st.metric("One-year mortality", f"{probability:.1%}")
        else:
            st.info("Enter values and select Calculate predicted risk.")
        st.caption("Model version: frozen seed-42 six-predictor model")
        st.caption(f"Model SHA-256: {freeze['model_sha256']}")

    st.divider()
    st.caption("Research-use tool. This estimate does not replace clinical assessment, guideline-directed treatment, or shared decision-making.")


if __name__ == "__main__":
    main()
