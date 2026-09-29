import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# --------------------------------------------------
# 1. CREATE SAMPLE DATA
# --------------------------------------------------

np.random.seed(42)

n = 5000

age = np.random.randint(18, 70, n)

income = np.random.normal(
    loc=50000,
    scale=20000,
    size=n
).clip(10000, 150000)

loan_amount = np.random.normal(
    loc=20000,
    scale=12000,
    size=n
).clip(1000, 100000)

credit_score = np.random.normal(
    loc=650,
    scale=80,
    size=n
).clip(300, 850)


# --------------------------------------------------
# 2. CREATE SYNTHETIC DEFAULT PROBABILITY
# --------------------------------------------------

logit = (
    -0.00003 * income
    + 0.00006 * loan_amount
    - 0.008 * (credit_score - 600)
    + 0.01 * (age - 40)
)

probability = 1 / (1 + np.exp(-logit))

default = np.random.binomial(
    1,
    probability
)


# --------------------------------------------------
# 3. CREATE DATAFRAME
# --------------------------------------------------

df = pd.DataFrame({
    "age": age,
    "income": income,
    "loan_amount": loan_amount,
    "credit_score": credit_score,
    "default": default,
})

print(df.head())


# --------------------------------------------------
# 4. FEATURES AND TARGET
# --------------------------------------------------

X = df[
    [
        "age",
        "income",
        "loan_amount",
        "credit_score",
    ]
]

y = df["default"]


# --------------------------------------------------
# 5. TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# --------------------------------------------------
# 6. CREATE ML PIPELINE
# --------------------------------------------------

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression()
    ),
])


# --------------------------------------------------
# 7. TRAIN
# --------------------------------------------------

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 8. TEST
# --------------------------------------------------

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(
    y_test,
    predictions
)

auc = roc_auc_score(
    y_test,
    probabilities
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nModel Results")
print("----------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)


# --------------------------------------------------
# 9. SAVE MODEL
# --------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/credit_model.joblib"
)

print(
    "\nModel saved to "
    "models/credit_model.joblib"
)