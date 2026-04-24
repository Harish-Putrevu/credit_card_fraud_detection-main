import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# ------------------- Load Model -------------------
MODEL_PATH = "credit_card_model.pkl"

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# ------------------- Infer Features -------------------
def get_features(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None

feature_names = get_features(model)

# ------------------ UI ------------------
st.title("💳 Credit Card Fraud Detection (CSV Only)")
st.write("Upload a CSV file containing credit card transactions.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # If model knows its features, align automatically
    if feature_names:
        missing = [c for c in feature_names if c not in df.columns]
        
        if missing:
            st.error(f"Your CSV is missing required columns: {missing}")
            st.stop()
        
        X = df[feature_names]
    else:
        st.warning("Model does not contain feature names. Trying to use entire CSV as input.")
        X = df.select_dtypes(include=[np.number])

    # ------------- Make Predictions -------------
    pred = model.predict(X)

    df_result = df.copy()
    df_result["predicted_label"] = pred

    st.subheader("Prediction Results")
    st.dataframe(df_result.head())

    # Download
    st.download_button(
        label="Download predictions",
        data=df_result.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )

    # ------------ Evaluation (Optional) ------------
    true_label_col = None
    for col in df.columns:
        if col.lower() in ["class", "label", "fraud", "is_fraud"]:
            true_label_col = col
            break

    if true_label_col:
        st.subheader("Model Performance (Using True Labels)")
        y_true = df[true_label_col]
        y_pred = pred

        st.write("✅ Accuracy:", accuracy_score(y_true, y_pred))
        st.write("✅ Precision:", precision_score(y_true, y_pred))
        st.write("✅ Recall:", recall_score(y_true, y_pred))
        st.write("✅ F1 Score:", f1_score(y_true, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        st.subheader("Confusion Matrix")
        st.write(cm)
