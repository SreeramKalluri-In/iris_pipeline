"""
Iris model web app — prediction + live drift monitoring panel.
Streamlit turns plain Python into a web page, so no HTML/JS needed.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Load the model once, when the page opens (not on every rerun).
model = joblib.load("iris_model.joblib")

# The exact column order the model was trained with.
FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

# Training reference (written by pipeline.py) — used for the drift check.
try:
    _training_ref = pd.read_csv("training_reference.csv")
    _ref_stats = _training_ref[FEATURES].describe().loc[["mean", "std"]]
    HAS_REF = True
except FileNotFoundError:
    HAS_REF = False

st.set_page_config(page_title="Iris Predictor & Monitor", page_icon="🌸")
st.title("Iris Species Predictor & Drift Monitor")
st.write(
    "Enter iris measurements below. The model predicts the species, "
    "and the monitoring panel shows how far your values drift from the "
    "training reference."
)

# Unlimited number inputs — no min/max, so any value is accepted.
col1, col2, col3, col4 = st.columns(4)
with col1:
    sepal_length = st.number_input("Sepal length (cm)", value=5.5, step=0.1)
with col2:
    sepal_width = st.number_input("Sepal width (cm)", value=3.0, step=0.1)
with col3:
    petal_length = st.number_input("Petal length (cm)", value=4.0, step=0.1)
with col4:
    petal_width = st.number_input("Petal width (cm)", value=1.0, step=0.1)

if st.button("Predict species"):
    flower = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=FEATURES,
    )
    prediction = model.predict(flower)
    probabilities = model.predict_proba(flower)
    top = probabilities[0].max()
    st.success(f"Predicted species: **{prediction[0]}** (confidence {top:.0%})")
    st.write(
        "Confidence = how likely the model thinks it is. "
        "Low confidence usually means the flower is unusual."
    )

# ── Monitoring section ────────────────────────────────────────────────
st.subheader("Monitoring: drift vs. training reference")

st.write(
    "In production, `monitor.py` runs this same check on every incoming "
    "batch of rows: it compares feature distributions to the training "
    "reference and writes `drift_report.html` with Evidently's "
    "`DataDriftPreset`. The panel below simulates that check on the four "
    "values you just entered, so you can see what monitoring is looking "
    "for — and why removing input limits matters."
)

if not HAS_REF:
    st.warning(
        "`training_reference.csv` not found. "
        "Run `uv run python pipeline.py` first to generate it."
    )
else:
    values   = np.array([sepal_length, sepal_width, petal_length, petal_width])
    ref_mean = _ref_stats.loc["mean"].values
    ref_std  = _ref_stats.loc["std"].values
    z_scores = (values - ref_mean) / ref_std

    monitor_rows = [
        "| Feature | Your val | Ref mean | Ref std | Z-score | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for feat, v, z, m, s in zip(FEATURES, values, z_scores, ref_mean, ref_std):
        status = "⚠️  > 2σ" if abs(z) > 2 else "✅"
        monitor_rows.append(
            f"| {feat} | {v:.1f} | {m:.2f} | {s:.2f} | {z:+.2f} | {status} |"
        )
    st.markdown("\n".join(monitor_rows))

    drifted = [f for f, z in zip(FEATURES, z_scores) if abs(z) > 2]

    if drifted:
        st.error(
            f"**Drift detected in:** {', '.join(drifted)} — more than 2 "
            "standard deviations from the training data. The model's "
            "prediction may be unreliable; in production this would "
            "trigger an alert."
        )
    else:
        st.info(
            "No drift detected. All values are within 2 standard deviations "
            "of the training reference — the prediction is operating in "
            "familiar territory."
        )
