import streamlit as st
import requests


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


API_URL = "http://127.0.0.1:8000/predict"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Main title */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-top: 0.25rem;
        margin-bottom: 2rem;
    }

    /* Cards */
    .card {
        padding: 1.5rem;
        border-radius: 14px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 1rem;
    }

    .card-title {
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }

    .card-description {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    /* Risk badges */
    .risk-low {
        padding: 0.6rem 1rem;
        border-radius: 10px;
        background-color: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.4);
        font-weight: 600;
    }

    .risk-medium {
        padding: 0.6rem 1rem;
        border-radius: 10px;
        background-color: rgba(234, 179, 8, 0.15);
        border: 1px solid rgba(234, 179, 8, 0.4);
        font-weight: 600;
    }

    .risk-high {
        padding: 0.6rem 1rem;
        border-radius: 10px;
        background-color: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        font-weight: 600;
    }

    /* Streamlit buttons */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.8rem;
        margin-top: 3rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("📊 Credit Risk")

    st.caption(
        "Machine Learning Credit Decision Support"
    )

    st.divider()

    st.subheader("About")

    st.write(
        """
        This application estimates the probability
        that a borrower may default on a loan using
        a trained machine learning model.
        """
    )

    st.divider()

    st.subheader("Model Output")

    st.write(
        """
        **Default Probability**

        Estimated probability that the customer
        belongs to the default class.

        **Prediction**

        - `0` → Lower default risk
        - `1` → Higher default risk
        """
    )

    st.divider()

    st.caption(
        "For demonstration and portfolio purposes."
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="main-title">
        Credit Risk Intelligence
    </div>

    <div class="subtitle">
        ML-powered borrower risk assessment and probability estimation
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MAIN LAYOUT
# =========================================================

input_col, result_col = st.columns(
    [1.1, 0.9],
    gap="large",
)


# =========================================================
# INPUT SECTION
# =========================================================

with input_col:

    st.subheader("👤 Borrower Information")

    st.caption(
        "Enter the applicant's financial and demographic information."
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1,
            help="Applicant's age in years.",
        )

        income = st.number_input(
            "Annual Income",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            format="%.2f",
            help="Applicant's annual income.",
        )

    with col2:

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=20000.0,
            step=1000.0,
            format="%.2f",
            help="Requested loan amount.",
        )

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=650,
            step=1,
            help="Applicant's credit score.",
        )

    st.divider()

    # -----------------------------------------------------
    # Quick derived indicators
    # -----------------------------------------------------

    if income > 0:
        loan_to_income = loan_amount / income
    else:
        loan_to_income = 0

    metric1, metric2 = st.columns(2)

    metric1.metric(
        "Loan-to-Income Ratio",
        f"{loan_to_income:.1%}",
    )

    metric2.metric(
        "Credit Score",
        f"{credit_score}",
    )

    st.caption(
        "These metrics are shown for context and may not "
        "necessarily be direct inputs to the trained model."
    )

    st.divider()

    predict_button = st.button(
        "Run Risk Assessment",
        type="primary",
        use_container_width=True,
    )


# =========================================================
# RESULT SECTION
# =========================================================

with result_col:

    st.subheader("📈 Risk Assessment")

    st.caption(
        "The prediction result will appear here."
    )

    if not predict_button:

        st.info(
            "Enter borrower information and click "
            "**Run Risk Assessment**."
        )


# =========================================================
# API REQUEST
# =========================================================

if predict_button:

    customer = {
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
    }

    with result_col:

        with st.spinner(
            "Analyzing borrower risk..."
        ):

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


                # =========================================
                # RISK CATEGORY
                # =========================================

                if probability < 0.30:

                    risk_level = "Low Risk"
                    risk_class = "risk-low"

                elif probability < 0.60:

                    risk_level = "Moderate Risk"
                    risk_class = "risk-medium"

                else:

                    risk_level = "High Risk"
                    risk_class = "risk-high"


                # =========================================
                # DISPLAY RESULT
                # =========================================

                st.success(
                    "Assessment completed successfully."
                )

                st.metric(
                    "Default Probability",
                    f"{probability:.2%}",
                )

                st.progress(
                    min(
                        max(probability, 0.0),
                        1.0,
                    )
                )

                st.markdown(
                    f"""
                    <div class="{risk_class}">
                        Risk Classification:
                        {risk_level}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.write("")

                if prediction == 1:

                    st.warning(
                        "The model predicts that this "
                        "borrower belongs to the "
                        "**higher default-risk class**."
                    )

                else:

                    st.success(
                        "The model predicts that this "
                        "borrower belongs to the "
                        "**lower default-risk class**."
                    )


                # =========================================
                # SUMMARY
                # =========================================

                st.divider()

                st.subheader(
                    "Applicant Summary"
                )

                summary_col1, summary_col2 = st.columns(2)

                summary_col1.metric(
                    "Age",
                    f"{age}",
                )

                summary_col2.metric(
                    "Credit Score",
                    f"{credit_score}",
                )

                summary_col1.metric(
                    "Annual Income",
                    f"${income:,.0f}",
                )

                summary_col2.metric(
                    "Loan Amount",
                    f"${loan_amount:,.0f}",
                )


                # =========================================
                # RAW API RESPONSE
                # =========================================

                with st.expander(
                    "View API Response"
                ):

                    st.json(result)


            except requests.ConnectionError:

                st.error(
                    """
                    Unable to connect to the prediction API.

                    Make sure FastAPI is running on:

                    `http://127.0.0.1:8000`
                    """
                )


            except requests.Timeout:

                st.error(
                    "The prediction API took too long "
                    "to respond."
                )


            except requests.HTTPError as error:

                st.error(
                    f"API returned an error: {error}"
                )


            except KeyError:

                st.error(
                    "The API response does not contain "
                    "the expected prediction fields."
                )


            except Exception as error:

                st.error(
                    f"Unexpected error: {error}"
                )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Credit Risk ML Platform • Streamlit + FastAPI
    </div>
    """,
    unsafe_allow_html=True,
)