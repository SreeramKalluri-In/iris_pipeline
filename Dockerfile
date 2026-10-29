# Dockerfile for a Hugging Face Space that runs our Streamlit app.
# (OPTIONAL - HF Spaces can also run app.py from requirements.txt alone.
#  This file just makes the build explicit and fast.)

FROM python:3.11-slim

# Install everything the app needs.
RUN pip install --no-cache-dir \
    streamlit \
    huggingface-hub \
    scikit-learn \
    pandas \
    joblib

# Drop the app + model into the container.
COPY app.py .
COPY iris_model.joblib .

WORKDIR /code

# Hugging Face Spaces expects the app to listen on port 7860.
EXPOSE 7860

# When the Space boots, run Streamlit and point it at the HF port.
ENTRYPOINT streamlit run app.py --server.port=7860 --server.address=0.0.0.0
