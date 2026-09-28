import joblib
import pandas as pd


MODEL_PATH = "models/credit_model.joblib"


model = joblib.load(MODEL_PATH)


def predict_default(
    age,
    income,
    loan_amount,
    credit_score,
):

    data = pd.DataFrame([
        {
            "age": age,
            "income": income,
            "loan_amount": loan_amount,
            "credit_score": credit_score,
        }
    ])

    probability = model.predict_proba(data)[0][1]

    prediction = int(
        probability >= 0.5
    )

    return {
        "prediction": prediction,
        "default_probability": float(probability),
    }