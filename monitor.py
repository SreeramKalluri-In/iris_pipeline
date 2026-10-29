"""
Monitoring step - uses Evidently (free, open-source, Apache-2.0 license).

Requires (see requirements-monitoring.txt):  pip install -r requirements-monitoring.txt

What this does, in plain language:
  1. DATA DRIFT check  - "Are the flowers arriving at the greenhouse still
     the same KIND of flowers the model learned from?" We compare a fresh
     batch of input against the training baseline. In a real system this
     batch is your web app's live logs; here we SIMULATE one and nudge two
     measurements up, so the check has something to catch.
  2. PERFORMANCE check - "Is the model still right about the same fraction
     of answers?" We re-grade its saved test-set predictions.

Run it:  python monitor.py
It writes two standalone HTML dashboards you open in any browser:
     drift_report.html         - data quality + data drift
     performance_report.html   - model accuracy, precision, recall, F1
"""

import numpy as np
import pandas as pd

from evidently import ColumnMapping
from evidently.metrics import (
    ClassificationPerformanceMetric,
    DataDriftPresetMetric,
    DataQualityPresetMetric,
)
from evidently.report import Report

# ------------------------------------------------------------------
# 1. Load the pieces we need to compare
# ------------------------------------------------------------------
# Baseline = what the model learned from (4 measurements + true species).
reference = pd.read_csv("training_reference.csv")
reference_features = reference.drop(columns=["target"])

# "Incoming" data = a fresh batch of flowers that just arrived.
# We start from the baseline and NUDGE two measurements upward,
# simulating "the florists started sending us different flowers".
incoming = reference_features.copy()
incoming["petal length (cm)"] = np.clip(
    incoming["petal length (cm)"] + 1.2, 0.0, None)   # shifted up
incoming["petal width (cm)"] = np.clip(
    incoming["petal width (cm)"] + 0.4, 0.0, None)     # shifted up

# The model's saved answers on the test set (true label + its guess).
predictions = pd.read_csv("test_predictions.csv")

# ------------------------------------------------------------------
# 2. Report #1 - data quality + data drift
# ------------------------------------------------------------------
# No target column needed here: we're only looking at the measurements.
drift_report = Report(metrics=[
    DataQualityPresetMetric(),   # blanks, unexpected values, new columns
    DataDriftPresetMetric(),    # distribution shift vs. the reference
])
drift_report.run(
    reference_data=reference_features,
    current_data=incoming,
    column_mapping=ColumnMapping(),
)

# ------------------------------------------------------------------
# 3. Report #2 - model performance
# ------------------------------------------------------------------
# Tell Evidently which column is the truth ("target") and which column
# is the model's guess ("predicted").
perf_report = Report(metrics=[ClassificationPerformanceMetric()])
perf_report.run(
    predictions=predictions,
    column_mapping=ColumnMapping(target="target", prediction="predicted"),
)

# ------------------------------------------------------------------
# 4. Save both as HTML dashboards and open them in your browser
# ------------------------------------------------------------------
# save_html() has the same API in Evidently 0.x and 1.x, so this works
# either way; open_in_browser pops them up automatically.
drift_report.save_html("drift_report.html", open_in_browser=True)
perf_report.save_html("performance_report.html", open_in_browser=True)
print("Wrote drift_report.html and performance_report.html "
      "(open them in any browser).")

# ------------------------------------------------------------------
# 5. How to read them
# ------------------------------------------------------------------
print("""
HOW TO READ THE REPORTS
----------------------
drift_report.html
  * DATA QUALITY - red flags for new missing values, wrong types, or
    values outside known ranges.
  * DATA DRIFT   - for each measurement, compares "incoming" vs the
    training baseline. Green = still looks the same. The two columns
    we intentionally nudged (petal length & petal width) should be
    flagged here - that's the check doing its job.

performance_report.html
  * ACCURACY, PRECISION, RECALL, F1 on the saved predictions.
  * Compare the accuracy here with the number pipeline.py printed on
    the test set. If live numbers start falling well below it, the
    model is degrading - retrain it on fresh data.
""")
