#!/usr/bin/env python
# coding: utf-8

import os
import sys
import io
import base64
import joblib
import shap
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components


# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="H. pylori Genetic Variation-Driven Gastric Cancer Risk Prediction: A SHAP-Explained Online Platform",
    layout="wide"
)

st.title("H. pylori Genetic Variation-Driven Gastric Cancer Risk Prediction: A SHAP-Explained Online Platform")


# =========================
# absence / presence mapping
# =========================
OPTION_LIST = ["absence", "presence"]

OPTION_MAP = {
    "absence": 0,
    "presence": 1
}


# =========================
# Load model
# =========================
@st.cache_resource
def load_model():
    """
    Load GBDT.pkl normally.
    If model deserialization raises No module named '_loss',
    automatically apply the sklearn._loss._loss compatibility mapping.
    """
    try:
        model = joblib.load("GBDT.pkl")
        return model

    except ModuleNotFoundError as e:
        if str(e) == "No module named '_loss'" or getattr(e, "name", None) == "_loss":
            import sklearn._loss._loss as cy_loss
            sys.modules["_loss"] = cy_loss
            model = joblib.load("GBDT.pkl")
            return model
        else:
            raise e


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


try:
    clf = load_model()
except Exception as e:
    st.error("Model loading failed. Please check whether GBDT.pkl is in the same directory as app.py and whether the scikit-learn version is compatible.")
    st.exception(e)
    st.stop()


# =========================
# Get feature names
# =========================
if not hasattr(clf, "feature_names_in_"):
    st.error("The current model does not have the feature_names_in_ attribute, so input fields cannot be generated automatically.")
    st.stop()

feature_names = list(clf.feature_names_in_)

if len(feature_names) != 23:
    st.warning(f"The current model has {len(feature_names)} features, not 23. Please confirm that the correct model is loaded.")


# =========================
# Initialize input state
# =========================
def init_input_state():
    """
    Initialize the dropdown state for each variable.
    The default value is absence for all variables.
    """
    for i, name in enumerate(feature_names):
        key = f"input_{i}"

        if key not in st.session_state:
            st.session_state[key] = "absence"

        # Prevent old number_input states such as 0.0 / 1.0 from causing selectbox errors
        if st.session_state[key] not in OPTION_LIST:
            st.session_state[key] = "absence"


def reset_inputs():
    """
    Reset all inputs to absence.
    """
    for i, name in enumerate(feature_names):
        st.session_state[f"input_{i}"] = "absence"


init_input_state()


# =========================
# Input area: two columns
# The first column contains 12 features, and the second column contains 11 features
# =========================
left_features = feature_names[:12]
right_features = feature_names[12:]

col1, col2 = st.columns(2)

with col1:
    for i, name in enumerate(left_features):
        st.selectbox(
            label=name,
            options=OPTION_LIST,
            key=f"input_{i}"
        )

with col2:
    for j, name in enumerate(right_features):
        i = j + 12
        st.selectbox(
            label=name,
            options=OPTION_LIST,
            key=f"input_{i}"
        )


# =========================
# Button area
# =========================
st.markdown("---")

btn_col1, btn_col2 = st.columns([1, 1])

with btn_col1:
    predict_btn = st.button(
        "Predict",
        type="primary",
        use_container_width=True
    )

with btn_col2:
    st.button(
        "Reset Inputs",
        on_click=reset_inputs,
        use_container_width=True
    )


# =========================
# Prediction and SHAP explanation
# =========================
if predict_btn:

    # Original dropdown selections: absence / presence
    input_label_dict = {
        name: st.session_state[f"input_{i}"]
        for i, name in enumerate(feature_names)
    }

    # Numeric values passed to the model: absence -> 0, presence -> 1
    input_dict = {
        name: OPTION_MAP[st.session_state[f"input_{i}"]]
        for i, name in enumerate(feature_names)
    }

    X_label = pd.DataFrame([input_label_dict], columns=feature_names)
    X = pd.DataFrame([input_dict], columns=feature_names)

    st.subheader("Current Input Data")

    with st.expander("View original input: absence / presence", expanded=False):
        st.dataframe(X_label, use_container_width=True)

    with st.expander("View model input values: 0 / 1", expanded=True):
        st.dataframe(X, use_container_width=True)

    try:
        pred_class = clf.predict(X)[0]

        if hasattr(clf, "predict_proba"):
            pred_proba_all = clf.predict_proba(X)
            classes = list(clf.classes_)

            # By default, explain and display the probability of class 1.
            # If class 1 is not available, use the last class by default.
            if 1 in classes:
                target_class = 1
                target_index = classes.index(1)
            else:
                target_class = classes[-1]
                target_index = len(classes) - 1

            pred_proba = pred_proba_all[0][target_index]

            # Index corresponding to the predicted class, used as a fallback
            if pred_class in classes:
                pred_index = classes.index(pred_class)
            else:
                pred_index = int(np.argmax(pred_proba_all[0]))

        else:
            pred_proba = None
            target_index = 0
            pred_index = 0
            target_class = None

        st.subheader("Prediction Results")

        st.success(f"Predicted class: {pred_class}")

        if pred_proba is not None:
            st.success(f"Prediction probability: {pred_proba * 100:.2f}%")
        else:
            st.info("The current model does not support predict_proba, so prediction probability cannot be displayed.")

    except Exception as e:
        st.error("Model prediction failed.")
        st.exception(e)
        st.stop()


    # =========================
    # SHAP force plot
    # =========================
    st.subheader("SHAP Force Plot")

    try:
        explainer = load_explainer(clf)
        shap_values = explainer(X)

        values = shap_values.values
        base_values = shap_values.base_values

        # Compatible with binary classification, multiclass classification, and different SHAP versions
        if values.ndim == 3:
            # shape: [number of samples, number of features, number of classes]
            shap_value_single = values[0, :, target_index]

            if np.ndim(base_values) == 2:
                expected_value = base_values[0, target_index]
            elif np.ndim(base_values) == 1:
                expected_value = base_values[target_index]
            else:
                expected_value = base_values

        elif values.ndim == 2:
            # shape: [number of samples, number of features]
            shap_value_single = values[0]

            if np.ndim(base_values) == 1:
                expected_value = base_values[0]
            else:
                expected_value = base_values

        else:
            st.error(f"Unrecognized SHAP values dimension: {values.shape}")
            st.stop()

        # Ensure expected_value is a scalar
        expected_value = float(np.ravel(expected_value)[0])

        # Prevent old figures from remaining
        plt.close("all")

        shap.force_plot(
            expected_value,
            np.round(shap_value_single, 3),
            np.round(X.iloc[0], 3),
            feature_names=feature_names,
            figsize=(45, 4),
            matplotlib=True,
            show=False,
            text_rotation=0,
            contribution_threshold=0.03
        )

        fig = plt.gcf()

        # Save as SVG for clearer display on the webpage
        svg_buffer = io.StringIO()
        fig.savefig(
            svg_buffer,
            format="svg",
            bbox_inches="tight"
        )
        plt.close(fig)

        svg_data = svg_buffer.getvalue()
        b64 = base64.b64encode(svg_data.encode("utf-8")).decode("utf-8")

        components.html(
            f"""
            <div style="width:100%; overflow-x:auto; border:1px solid #e6e6e6; padding:10px;">
                <img src="data:image/svg+xml;base64,{b64}" style="width:1800px;">
            </div>
            """,
            height=360,
            scrolling=False
        )

    except Exception as e:
        st.error("Failed to generate SHAP force plot.")
        st.exception(e)