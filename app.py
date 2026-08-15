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
    reporting_path = assets / "reporting_manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    reporting = json.loads(reporting_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen" or tuple(freeze.get("features", [])) != FEATURES:
        raise RuntimeError("The frozen model record has an unexpected feature contract.")
    if sha256_file(model_path) != freeze.get("model_sha256"):
        raise RuntimeError("The packaged model does not match its freeze record.")
    if reporting.get("random_seed") != 42 or tuple(reporting.get("formal_features", [])) != FEATURES:
        raise RuntimeError("The reporting manifest has an unexpected model contract.")
    if reporting.get("model_sha256") != freeze.get("model_sha256"):
        raise RuntimeError("The reporting manifest does not match the packaged model.")
    if reporting.get("threshold", {}).get("value") != freeze.get("threshold", {}).get("value"):
        raise RuntimeError("The reporting manifest does not match the frozen threshold.")
    model = joblib.load(model_path)
    if tuple(getattr(model, "feature_names_in_", ())) != FEATURES:
        raise RuntimeError("The packaged model has an unexpected feature contract.")
    return model, freeze, reporting


@st.cache_resource(show_spinner=False)
def cached_model():
    return load_verified_model()


def case_frame(age, hemoglobin, creatinine, stemi, wbc, platelets) -> pd.DataFrame:
    return pd.DataFrame(
        [[float(age), float(hemoglobin), float(creatinine), int(stemi == "Yes"), float(wbc), float(platelets)]],
        columns=list(FEATURES),
    )


def numeric_input_label(spec: dict) -> str:
    return f"{spec['label']} ({spec['unit']}; allowed {spec['min']:g}–{spec['max']:g})"


def main() -> None:
    try:
        model, freeze, reporting = cached_model()
    except Exception as exc:
        st.error(f"Model verification failed: {exc}")
        st.stop()
    st.set_page_config(page_title=reporting["endpoint"]["calculator_title"], page_icon="♥", layout="wide")
    st.title(reporting["endpoint"]["calculator_title"])
    st.write("Enter the six routinely available predictors to obtain the model-estimated 1-year mortality risk.")
    inputs = reporting["feature_inputs"]

    entry, result = st.columns((1.2, 0.8), gap="large")
    with entry:
        with st.form("six_predictor_form"):
            age_spec = inputs["age"]
            age = st.number_input(numeric_input_label(age_spec), min_value=age_spec["min"], max_value=age_spec["max"], value=age_spec["default"], step=age_spec["step"])
            hemoglobin_spec = inputs["hemoglobin_g_l"]
            hemoglobin = st.number_input(numeric_input_label(hemoglobin_spec), min_value=hemoglobin_spec["min"], max_value=hemoglobin_spec["max"], value=hemoglobin_spec["default"], step=hemoglobin_spec["step"])
            creatinine_spec = inputs["creatinine_umol_l"]
            creatinine = st.number_input(numeric_input_label(creatinine_spec), min_value=creatinine_spec["min"], max_value=creatinine_spec["max"], value=creatinine_spec["default"], step=creatinine_spec["step"])
            stemi_spec = inputs["stemi"]
            stemi_options = tuple(stemi_spec["options"])
            stemi = st.selectbox(stemi_spec["label"], stemi_options, index=stemi_options.index(stemi_spec["default"]))
            wbc_spec = inputs["wbc_10e9_l"]
            wbc = st.number_input(numeric_input_label(wbc_spec), min_value=wbc_spec["min"], max_value=wbc_spec["max"], value=wbc_spec["default"], step=wbc_spec["step"])
            platelet_spec = inputs["platelets_10e9_l"]
            platelets = st.number_input(numeric_input_label(platelet_spec), min_value=platelet_spec["min"], max_value=platelet_spec["max"], value=platelet_spec["default"], step=platelet_spec["step"])
            calculate = st.form_submit_button("Calculate predicted risk", type="primary")
    with result:
        st.subheader("Estimated risk")
        if calculate:
            probability = float(model.predict_proba(case_frame(age, hemoglobin, creatinine, stemi, wbc, platelets))[:, 1][0])
            st.metric(reporting["endpoint"]["calculator_metric"], f"{probability:.1%}")
        else:
            st.info("Enter values and select Calculate predicted risk.")
        st.caption("Model version: frozen seed-42 six-predictor model")
        st.caption(f"Model SHA-256: {freeze['model_sha256']}")

    st.divider()
    st.caption("Research-use tool. This estimate does not replace clinical assessment, guideline-directed treatment, or shared decision-making.")


if __name__ == "__main__":
    main()
