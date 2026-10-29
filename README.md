<<<<<<< HEAD
# iris_pipeline
ML pipeline deployed in HF Spaces and Monitored using evidently.ai
=======
# Iris Species Predictor - Hugging Face Space

A tiny Streamlit web app that predicts the species of an iris flower
from four measurements. Deployed on a free Hugging Face Space
(CPU basic tier).

## Run it

If you clone this folder, just:

```bash
pip install -r requirements.txt
streamlit run app.py
```

On Hugging Face Spaces, the `Dockerfile` + `iris_model.joblib` are
all the Space needs - it boots Streamlit on port 7860 automatically.

## Files

- `app.py` - the Streamlit UI
- `iris_model.joblib` - the trained model (save it next to `app.py`
  after running `pipeline.py`)
- `Dockerfile`, `requirements.txt` - how the Space builds the app
>>>>>>> ad927b8 (FirstCommit)
