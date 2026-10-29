"""
Tiny Streamlit web app for the iris model.
This is the whole "product" - 4 sliders, a predict button, an answer.
Streamlit turns plain Python into a web page, so no HTML/JS needed.
"""

import joblib
import pandas as pd
import streamlit as st

# Load the model once, when the page opens (not on every button click).
model = joblib.load("iris_model.joblib")

# The exact column order the model was trained with.
FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

st.title("Iris Species Predictor")
st.write(
    "Enter measurements of an iris flower (a real one or a made-up one). "
    "The model will guess which species it is."
)

# One slider per feature. Real iris measurements live in roughly
# these ranges, so this keeps users inside sensible territory.
sepal_length = st.slider("Sepal length (cm)", 4.0, 9.0, 5.5, 0.1)
sepal_width = st.slider("Sepal width (cm)", 2.0, 5.0, 3.0, 0.1)
petal_length = st.slider("Petal length (cm)", 1.0, 8.0, 4.0, 0.1)
petal_width = st.slider("Petal width (cm)", 0.1, 3.0, 1.0, 0.1)

if st.button("Predict species"):
    # Build the exact same shape of data the model was trained on.
    flower = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=FEATURES,
    )

    prediction = model.predict(flower)

    # Model probabilities - how CONFIDENT is it? (needs a raw frame
    # because the scaler's pandas output isn't guaranteed here).
    probabilities = model.predict_proba(flower)
    top = probabilities[0].max()

    st.success(f"Predicted species: **{prediction[0]}** "
               f"(confidence {top:.0%})")
    st.write(
        "Confidence = how likely the model thinks it is. "
        "Low confidence usually means the flower is unusual."
    )
