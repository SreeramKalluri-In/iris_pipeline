"""
Iris model monitoring with Evidently 0.7.x.

Creates:
1. drift_report.html        - data quality + feature distribution drift
2. performance_report.html  - classification performance metrics

Evidently 0.7 changed the public API compared to 0.6:
- Report / Dataset / DataDefinition are imported from the top level
- presets live in `evidently.presets`, and the old DataQualityPreset
  was renamed DataSummaryPreset
- report.run() now takes (current_data, reference_data) and returns a
  Snapshot; the snapshot saves the HTML report
- column mapping is done with a DataDefinition (e.g. which columns are
  the target and the prediction) instead of a ColumnMapping object
"""

import numpy as np
import pandas as pd

from evidently import DataDefinition, Dataset, Report
from evidently.core.datasets import MulticlassClassification
from evidently.presets import (
    ClassificationPreset,
    DataDriftPreset,
    DataSummaryPreset,
)


# ------------------------------------------------------------------
# 1. Load reference data and create simulated incoming data
# ------------------------------------------------------------------
# Reference = training data used by the model.
# It contains four measurements plus the true species label.
reference = pd.read_csv("training_reference.csv")
reference_features = reference.drop(columns=["target"])

# Simulate a new batch of incoming flowers by shifting two features upward.
incoming = reference_features.copy()

incoming["petal length (cm)"] = np.clip(
    incoming["petal length (cm)"] + 1.2,
    0.0,
    None,
)

incoming["petal width (cm)"] = np.clip(
    incoming["petal width (cm)"] + 0.4,
    0.0,
    None,
)

# Load saved model predictions.
# Expected columns: target, predicted
predictions = pd.read_csv("test_predictions.csv")


# ------------------------------------------------------------------
# 2. Data quality and data drift report
# ------------------------------------------------------------------
# This report compares incoming feature data with the training baseline.
# It does not require model predictions.
#
# 0.7 API: wrap each DataFrame in a Dataset, and pass CURRENT data first,
# REFERENCE data second (the reverse of the 0.6 order). The presets are
# passed to the Report; DataSummaryPreset covers the old data-quality
# checks and DataDriftPreset compares distributions between the two.
drift_report = Report(
    [
        DataSummaryPreset(),
        DataDriftPreset(),
    ]
)

drift_snapshot = drift_report.run(
    Dataset.from_pandas(incoming),
    Dataset.from_pandas(reference_features),
)

drift_snapshot.save_html("drift_report.html")


# ------------------------------------------------------------------
# 3. Classification performance report
# ------------------------------------------------------------------
# This report evaluates the model using true labels and predicted labels.
# 0.7 API: tell Evidently which column is the target and which is the
# model's prediction with a DataDefinition (0.6 used ColumnMapping).
# MulticlassClassification fits our three iris species; it can also
# carry probability columns, but we only have plain label predictions.
classification_definition = DataDefinition(
    classification=[
        MulticlassClassification(
            target="target",
            prediction_labels="predicted",
        )
    ]
)

performance_report = Report([ClassificationPreset()])

performance_snapshot = performance_report.run(
    Dataset.from_pandas(predictions, data_definition=classification_definition),
    None,
)

performance_snapshot.save_html("performance_report.html")


# ------------------------------------------------------------------
# 4. Summary
# ------------------------------------------------------------------
print("Monitoring reports created successfully:")
print("- drift_report.html")
print("- performance_report.html")

print(
    """
HOW TO READ THE REPORTS
-----------------------
drift_report.html
- DATA SUMMARY shows missing values, feature statistics, and data issues.
- DATA DRIFT compares incoming measurements with the training baseline.
- petal length (cm) and petal width (cm) should be flagged because they
  were intentionally shifted upward.

performance_report.html
- Shows classification metrics such as accuracy, precision, recall, and F1.
- Compare these results with the test-set metrics from pipeline.py.
- A large drop in performance may indicate model degradation and the need
  to retrain with newer data.
"""
)
