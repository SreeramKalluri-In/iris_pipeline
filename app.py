"""
Iris model web app — prediction + live drift monitoring panel.
Gradio version (drop-in replacement for app.py's Streamlit UI).

HF Spaces runs this on the free CPU tier (zero GPUs) — Gradio is the
Space type for that, and it listens on port 7860 by default.

Test locally first:
    uv run python app_gradio.py --self-test   # headless: runs two sample
                                               # predictions, no server
    uv run python app_gradio.py                # serves http://127.0.0.1:7860
                                               # and opens your browser
    uv run python app_gradio.py --port 7861    # custom local port

On a Space the same entry point automatically binds 0.0.0.0:7860
(detected via the SPACE_ID env var Gradio sets in Spaces).
"""

import joblib
import numpy as np
import pandas as pd
import gradio as gr

# Load the model once, when the module is imported (not on every inference).
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


def predict(sepal_length, sepal_width, petal_length, petal_width):
    """Run one prediction and the drift check; return markdown for both panels."""
    flower = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=FEATURES,
    )
    prediction = model.predict(flower)
    probabilities = model.predict_proba(flower)
    top = probabilities[0].max()
    result_md = (
        f"### Predicted species: **{prediction[0]}** (confidence {top:.0%})\n\n"
        "Confidence = how likely the model thinks it is. "
        "Low confidence usually means the flower is unusual."
    )

    # ── Monitoring panel ───────────────────────────────────────────────
    monitor_md = "## Monitoring: drift vs. training reference\n\n"

    if not HAS_REF:
        monitor_md += (
            "**`training_reference.csv` not found.** "
            "Run `uv run python pipeline.py` first to generate it."
        )
        return result_md, monitor_md

    values = np.array([sepal_length, sepal_width, petal_length, petal_width])
    ref_mean = _ref_stats.loc["mean"].values
    ref_std = _ref_stats.loc["std"].values
    z_scores = (values - ref_mean) / ref_std

    rows = [
        "| Feature | Your val | Ref mean | Ref std | Z-score | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for feat, v, z, m, s in zip(FEATURES, values, z_scores, ref_mean, ref_std):
        status = "⚠️  > 2σ" if abs(z) > 2 else "✅"
        rows.append(f"| {feat} | {v:.1f} | {m:.2f} | {s:.2f} | {z:+.2f} | {status} |")

    drifted = [f for f, z in zip(FEATURES, z_scores) if abs(z) > 2]
    if drifted:
        alert = (
            f"**Drift detected in:** {', '.join(drifted)} — more than 2 "
            "standard deviations from the training data. The model's "
            "prediction may be unreliable; in production this would "
            "trigger an alert."
        )
    else:
        alert = (
            "No drift detected. All values are within 2 standard deviations "
            "of the training reference — the prediction is operating in "
            "familiar territory."
        )

    monitor_md += "\n".join(rows) + f"\n\n{alert}\n"
    monitor_md += (
        "\n\n---\n\nIn production, `monitor.py` runs this same check on every "
        "incoming batch of rows: it compares feature distributions to the "
        "training reference and writes `drift_report.html` with Evidently's "
        "`DataDriftPreset`. The table above simulates that check on the four "
        "values you just entered, so you can see what monitoring is looking "
        "for."
    )
    return result_md, monitor_md


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Sepal length (cm)", value=5.5),
        gr.Number(label="Sepal width (cm)", value=3.0),
        gr.Number(label="Petal length (cm)", value=4.0),
        gr.Number(label="Petal width (cm)", value=1.0),
    ],
    outputs=[
        gr.Markdown(label="Prediction"),
        gr.Markdown(label="Monitoring"),
    ],
    title="Iris Species Predictor & Drift Monitor",
    description=(
        "Enter iris measurements. The model predicts the species, and the "
        "monitoring panel shows how far your values drift from the training "
        "reference."
    ),
)

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Iris Gradio app")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run two sample predictions headlessly and exit (no server)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="local server port (default 7860)",
    )
    args = parser.parse_args()

    if args.self_test:
        # Quick logic check without starting a server: one familiar flower,
        # one deliberately out-of-distribution set.
        for sample in ((5.5, 3.0, 4.0, 1.0), (15.0, 0.5, 8.0, 3.0)):
            result_md, monitor_md = predict(*sample)
            print(f"input={sample}\n")
            print(result_md)
            print()
            print(monitor_md)
            print("=" * 60)
        raise SystemExit(0)

    if os.environ.get("SPACE_ID"):
        # Hugging Face Space: bind on all interfaces, port 7860.
        demo.launch(server_name="0.0.0.0", server_port=7860)
    else:
        # Local dev: loopback only, browser opens automatically.
        demo.launch(server_name="127.0.0.1", server_port=args.port)
