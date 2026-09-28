import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000/predict"


st.title(
    "Credit Risk Prediction"
)

st.write(
    "Enter customer information below."
)


age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

income = st.number_input(
    "Annual Income",
    min_value=0.0,
    value=50000.0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=20000.0
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=650
)


if st.button("Predict"):

    customer = {
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
    }

    try:

        response = requests.post(
            API_URL,
            json=customer,
            timeout=10,
        )

        response.raise_for_status()

        result = response.json()

        probability = result[
            "default_probability"
        ]

        prediction = result[
            "prediction"
        ]

        st.subheader(
            "Prediction Result"
        )

        st.metric(
            "Default Probability",
            f"{probability:.2%}"
        )

        if prediction == 1:

            st.warning(
                "Model prediction: Higher default risk"
            )

        else:

            st.success(
                "Model prediction: Lower default risk"
            )

    except requests.RequestException:

        st.error(
            "Could not connect to the prediction API. "
            "Make sure FastAPI is running."
        )