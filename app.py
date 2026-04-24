# app.py
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# -------------- Page setup --------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide",
    page_icon="💳",
)

PRIMARY = "#16a34a"   # green
DANGER  = "#dc2626"   # red
MUTED   = "#64748b"   # slate-500

st.markdown(
    f"""
    <style>
    .metric-good > div > div:nth-child(2) {{ color:{PRIMARY} !important; }}
    .metric-bad  > div > div:nth-child(2) {{ color:{DANGER}  !important; }}
    .stMetric label, .st-emotion-cache-15hul6a {{ color: #e2e8f0 !important; }}
    .stAlert > div {{ font-size: 1.05rem; }}
    .small-muted {{ color:{MUTED}; font-size:0.9rem; }}
    .highlight-fraud {{ background-color: rgba(220,38,38,0.12) !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------- Helpers --------------
@st.cache_resource
def load_model(path="credit_card_model.pkl"):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Could not load model '{path}': {e}")
        return None

def infer_features(model):
    # Try common places where training columns are stored
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    for attr in ("columns","feature_columns","features"):
        if hasattr(model, attr):
            cols = getattr(model, attr)
            if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
                return list(cols)
    return None

def align_features(df, features):
    """Ensure df has all required features in the right order; create missing columns as 0."""
    df = df.copy()
    missing = [c for c in features if c not in df.columns]
    for c in missing:
        df[c] = 0
    return df[features]

def predict(model, X):
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    yhat = model.predict(X)
    return yhat, proba

def style_fraud(df, label_col="predicted_label"):
    def _rowstyle(row):
        if row[label_col] == 1:
            # Light red background for fraud rows
            return ["background-color: rgba(255, 0, 0, 0.25);" for _ in row]
        else:
            # No style for normal rows
            return ["" for _ in row]
    return df.style.apply(_rowstyle, axis=1)



def kpi_card(col, label, value, delta=None, good=True):
    with col:
        css = "metric-good" if good else "metric-bad"
        st.markdown(f"<div class='{css}'>", unsafe_allow_html=True)
        st.metric(label, value, delta=delta)
        st.markdown("</div>", unsafe_allow_html=True)

def plot_bar_counts(df, label_col):
    counts = df[label_col].value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.bar(["Not Fraud","Fraud"], counts.values)
    ax.set_title("Prediction Count")
    st.pyplot(fig)

def plot_amount_box(df, label_col, amount_col="Amount"):
    if amount_col not in df.columns:
        return
    sample = df[[label_col, amount_col]].copy()
    sample[label_col] = sample[label_col].map({0:"Not Fraud", 1:"Fraud"})
    fig, ax = plt.subplots()
    groups = [sample[sample[label_col]=="Not Fraud"][amount_col],
              sample[sample[label_col]=="Fraud"][amount_col]]
    ax.boxplot(groups, labels=["Not Fraud","Fraud"], showfliers=False)
    ax.set_title("Amount Distribution by Predicted Class")
    st.pyplot(fig)

def plot_time_scatter(df, label_col, time_col="Time", max_points=10000):
    if time_col not in df.columns:
        return
    sample = df[[time_col, label_col]].copy()
    if len(sample) > max_points:
        sample = sample.sample(max_points, random_state=42)
    fig, ax = plt.subplots()
    ax.scatter(sample[time_col], sample[label_col], s=4, alpha=0.35)
    ax.set_xlabel("Time")
    ax.set_ylabel("Predicted Label")
    ax.set_title("Fraud vs Time (sample)")
    st.pyplot(fig)

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Not Fraud","Fraud"])
    ax.set_yticklabels(["Not Fraud","Fraud"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for (i,j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha='center', va='center', color='white', fontsize=14)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

def plot_roc(y_true, probs):
    fpr, tpr, _ = roc_curve(y_true, probs)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc(fpr,tpr):.3f}")
    ax.plot([0,1],[0,1],'--')
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(loc="lower right")
    ax.set_title("ROC Curve")
    st.pyplot(fig)

def fraud_category(row):
    if row["predicted_label"] == 0:
        return "Not Fraud"
    if row["Amount"] > 5000:
        return "High-Value Fraud"
    if row["Time"] < 10:
        return "Velocity Attack"
    if row["V13"] < -2 and row["V17"] > 3:
        return "CNP Fraud Pattern"
    return "General Fraud"


# -------------- Sidebar: model + upload --------------
st.title("💳 Credit Card Fraud Detection — Analytics Dashboard")

with st.sidebar:
    st.header("Model")
    st.caption("The app expects a scikit-learn model saved as `credit_card_model.pkl`.")
    model = load_model("credit_card_model.pkl")
    if model is None:
        st.stop()
    features = infer_features(model)
    if features:
        st.success("Model loaded ✓")
        st.caption("Detected feature columns from model.")
    else:
        st.warning("Model loaded, but feature names are not embedded. The app will use numeric columns from your CSV.")

    st.header("Upload Data")
    uploaded = st.file_uploader("Upload your CSV (e.g., Kaggle creditcard.csv)", type=["csv"])

# -------------- Main flow --------------
if uploaded is None:
    st.info("Upload a CSV to begin. The file never leaves your machine.")
    st.markdown("<span class='small-muted'>Expected columns typically include: Time, V1..V28, Amount, and optionally Class as ground truth.</span>", unsafe_allow_html=True)
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("Uploaded Data Preview")
st.dataframe(df.head(), use_container_width=True)

# Build X
if features:
    missing = [c for c in features if c not in df.columns]
    if missing:
        st.error(f"Your CSV is missing required model features: {missing}")
        st.stop()
    X = align_features(df, features)
else:
    # Fall back: use numeric columns (last resort)
    X = df.select_dtypes(include=[np.number])

# Predict
with st.spinner("Running predictions..."):
    y_pred, y_prob = predict(model, X)

result = df.copy()
result["predicted_label"] = y_pred.astype(int)
if y_prob is not None:
    result["fraud_probability"] = y_prob

# KPIs
total = len(result)
fraud_count = int((result["predicted_label"] == 1).sum())
legit_count = total - fraud_count
fraud_rate = (fraud_count / total * 100.0) if total else 0.0

st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
kpi_card(col1, "Total Transactions", f"{total:,}", good=True)
kpi_card(col2, "Fraud Detected", f"{fraud_count:,}", good=False)
kpi_card(col3, "Legitimate", f"{legit_count:,}", good=True)
kpi_card(col4, "Fraud Rate", f"{fraud_rate:.3f}%", good=(fraud_rate < 1.0))

# Status banner
if fraud_count > 0:
    st.error(f"🚨 Fraud detected in {fraud_count:,} transactions. Review the table below.")
else:
    st.success("✅ No fraud detected in the uploaded data.")

# Tabs: Predictions | Fraud Only | Analytics | Evaluation
tab_pred, tab_fraud, tab_analytics, tab_eval = st.tabs(
    ["📄 Predictions", "🚨 Fraud Only", "📊 Analytics", "✅ Evaluation (if labels present)"]
)

with tab_pred:
    st.subheader("Prediction Results")
    # Show only first 500 rows with styling
preview_size = 500
styled_preview = style_fraud(result.head(preview_size))

st.write(f"Showing styled preview of first {preview_size} rows:")
st.dataframe(styled_preview, use_container_width=True)

# Show full table without styling (for performance)
st.write("Full unstyled dataframe:")
st.dataframe(result, use_container_width=True)


st.download_button(
        "Download predictions CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv"
    )

with tab_fraud:
    st.subheader("Fraudulent Transactions (predicted)")
    fraud_df = result[result["predicted_label"] == 1]
    if fraud_df.empty:
        st.info("No fraud rows to show.")
    else:
        st.dataframe(style_fraud(fraud_df), use_container_width=True)

        st.download_button(
            "Download fraud-only CSV",
            data=fraud_df.to_csv(index=False).encode("utf-8"),
            file_name="fraud_only.csv"
        )

with tab_analytics:
    st.subheader("Distributions & Patterns")
    c1, c2 = st.columns(2)
    with c1:
        plot_bar_counts(result, "predicted_label")
    with c2:
        plot_amount_box(result, "predicted_label", amount_col="Amount")
    st.caption("Box plot hides outliers to keep scale readable.")

    st.subheader("Fraud vs Time (sample)")
    plot_time_scatter(result, "predicted_label", time_col="Time")

with tab_eval:
    # If real labels exist, show metrics
    label_col = None
    for name in ["Class","label","is_fraud","fraud","true_label"]:
        if name in df.columns:
            label_col = name
            break

    if label_col is None:
        st.info("No ground-truth label column found (e.g., 'Class'). Add one to see evaluation metrics.")
    else:
        st.subheader("Evaluation against Ground Truth")
        y_true = df[label_col].astype(int).values
        y_hat  = result["predicted_label"].astype(int).values

        c1, c2, c3, c4 = st.columns(4)
        kpi_card(c1, "Accuracy",  f"{accuracy_score(y_true,y_hat):.4f}")
        kpi_card(c2, "Precision", f"{precision_score(y_true,y_hat, zero_division=0):.4f}")
        kpi_card(c3, "Recall",    f"{recall_score(y_true,y_hat,  zero_division=0):.4f}")
        kpi_card(c4, "F1 Score",  f"{f1_score(y_true,y_hat,      zero_division=0):.4f}")

        st.subheader("Confusion Matrix")
        plot_confusion(y_true, y_hat)

        if "fraud_probability" in result.columns:
            st.subheader("ROC Curve")
            plot_roc(y_true, result["fraud_probability"].values)

st.markdown("---")
st.caption("Tip: The model expects the same preprocessing/columns used in training (Kaggle credit card dataset: Time, V1–V28, Amount). If you trained a pipeline, preprocessing is already embedded.")
