"""
Iris Flower Classifier - Complete End-to-End ML Pipeline
=========================================================
Scenario: You're a greenhouse manager. Florists want to order the right
iris variety, but telling them apart by eye is tricky. Your job: build a
model that identifies the species from four simple measurements
(sepal/flower-leaf length and width), then ship it live on the web
and keep an eye on it.

Run this file once:  python pipeline.py
It will (1) load the data, (2) explore + prepare it, (3) train a model,
(4) evaluate it on held-out data, and (5) save everything the
deployment and monitoring steps need:

    iris_model.joblib    - the trained model + its preprocessing
    training_reference.csv - a sample of the data the model learned from
                             (our "known-good" baseline for drift checks)
    test_predictions.csv   - the model's answers on the test set, used
                             to monitor live model performance
"""

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Tell matplotlib to draw charts inline (e.g., in Jupyter).
# Harmless when running in a plain terminal.
try:
    matplotlib.use("inline")
except Exception:
    pass


# ------------------------------------------------------------------
# STEP 1 - LOAD THE DATA
# ------------------------------------------------------------------
# A "dataset" is just a table: rows are individual things we measured
# (here: 150 iris flowers), columns are facts about each one.
# as_frame=True asks for pandas objects; scikit-learn bundles this
# classic dataset, so no download needed.
iris = load_iris(as_frame=True)
data = iris.data.copy()          # the measurements
data["species"] = iris.target    # the answer we want to predict

# The target column is stored as numbers 0, 1, 2 - we turn them into
# real names so everything is human-friendly from here on.
species_names = dict(enumerate(iris.target_names))  # {0: 'setosa', ...}
data["species"] = data["species"].map(species_names)

print("=== STEP 1: First look at our data ===")
print(data.head())
print("Rows (flowers):", data.shape[0], "| Columns:", data.shape[1])

# The 4 "features" are the measurement columns; "species" is the target.
feature_columns = [c for c in data.columns if c != "species"]
print("Feature columns:", feature_columns)
print("Species counts (how many of each kind):")
print(data["species"].value_counts())

# ------------------------------------------------------------------
# STEP 2 - EXPLORE & PREPARE THE DATA
# ------------------------------------------------------------------
# "Exploratory data analysis" (EDA) = curiosity-driven checking:
# are there blanks? duplicates? weird values?
print("\n=== STEP 2: Checking for problems ===")
print("Missing values per column:")
print(data.isna().sum())
print("Duplicate rows:", data.duplicated().sum())

# The Iris dataset is famously clean, but a REAL project always runs
# these checks. If missing values ever appear, you'd fill them in
# (e.g. data.fillna(data.median())) or drop those rows. Here there's
# nothing to fix - the pipeline simply survives a messy dataset too.

# Let's SEE the data - charts beat print() for spotting patterns.
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, col in zip(axes.flat, feature_columns):
    sns.boxplot(data = data, y = col, hue= "species", ax=ax)
    ax.set_title(col, fontsize=9)
plt.suptitle("Each measurement by species - see how the species separate?")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(data[feature_columns].corr(), annot=True, fmt=".2f",
            cmap="coolwarm")
ax.set_title("Correlation: petal size is the most informative feature")
plt.show()

# ------------------------------------------------------------------
# STEP 3 - SPLIT INTO TRAINING AND TEST DATA
# ------------------------------------------------------------------
# We must hold out some flowers the model has NEVER seen, otherwise
# we'd be "cheating" - grading itself on a test it already studied.
#
#   training data: the model studies these (70%)
#   test data:     the model gets graded on these (30%) - one time only
#
# stratify keeps the species mix balanced in both halves.

X = data[feature_columns]  # the measurements (the "input")
y = data["species"]        # the species labels  (the "answer")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\n=== STEP 3: Split ===\n"
      f"Training set: {X_train.shape[0]} flowers | "
      f"Test set: {X_test.shape[0]} flowers")

# ------------------------------------------------------------------
# STEP 4 - TRAIN THE MODEL
# ------------------------------------------------------------------
# A "model" is a mathematical formula the computer fills in by
# studying the training data. We use Logistic Regression - a simple,
# fast, popular starting point for classification.
#
# StandardScaler squishes every measurement to a similar scale
# (mean 0, std 1) so one big-numbered feature can't bully the others.
# We build it into a "pipeline" object so scaling + model stay glued
# together - this guarantees the web app scales new input the exact
# same way the model was trained.

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000),  # tiny models may need more iterations
)
model.fit(X_train, y_train)

# --- Optional sanity check: cross-validation ---
# Split the training data into 5 folds, train on 4, grade on 1,
# rotate, average. It's like 5 mini mock exams - a good estimate of
# how well the model will generalize to new flowers.
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-validation accuracy: {cv_scores.mean():.2f} "
      f"(varies by {cv_scores.std():.2f} between folds)")

# Save the model + its scaler so the web app (app.py) can load it.
joblib.dump(model, "iris_model.joblib")
print("Saved model to iris_model.joblib")

# Save a "known-good" sample (features + the true species) - this is the
# baseline the monitoring step will compare live data against for drift.
train_reference = pd.concat([X_train, y_train], axis=1).rename(
    columns={"species": "target"})
train_reference.sample(100, random_state=42).to_csv(
    "training_reference.csv", index=False)
print("Saved training_reference.csv (baseline for drift checks)")

# ------------------------------------------------------------------
# STEP 5 - PREDICT & EVALUATE ON THE TEST SET
# ------------------------------------------------------------------
# "Prediction" = the model guessing species for flowers it has never
# seen. "Evaluation" = comparing those guesses to the true species.
y_pred = model.predict(X_test)  # NOTE: test data, not training data

print(f"\n=== STEP 5: Evaluation on the held-out test set ===")
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
print(classification_report(y_test, y_pred, digits=2))

# A "confusion matrix" shows WHERE it's wrong:
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=y.unique(), cmap="Blues", ax=ax
)
ax.set_title("Confusion matrix: what did we get right / mix up?")
plt.show()

# Save the model's test-set answers - we'll feed them to the monitoring
# report later as a "recent performance" sample.
# Evidently expects the true label in a column called "target" and the
# model's answer in a column called "predicted".
pd.DataFrame({"target": y_test.values,
              "predicted": y_pred}).to_csv(
    "test_predictions.csv", index=False)
print("Saved test_predictions.csv (sample performance data)")


# ------------------------------------------------------------------
# DONE - the model is ready. Next up:
#   • app.py      -> a tiny Streamlit web app you host on Hugging Face
#   • monitor.py  -> an Evidently report that keeps it healthy
# ------------------------------------------------------------------
print("\nAll local steps complete! "
      "Files created: iris_model.joblib, training_reference.csv, "
      "test_predictions.csv")
