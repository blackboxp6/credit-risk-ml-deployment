from fastapi import FastAPI
from pydantic import BaseModel

from src.model import predict_default


app = FastAPI(
    title="Credit Risk ML API",
    description="Simple machine learning model deployment",
    version="1.0"
)


class Customer(BaseModel):

    age: int

    income: float

    loan_amount: float

    credit_score: float


@app.get("/")
def home():

    return {
        "message": "Credit Risk API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(customer: Customer):

    result = predict_default(
        age=customer.age,
        income=customer.income,
        loan_amount=customer.loan_amount,
        credit_score=customer.credit_score,
    )

    return result