# Credit Risk ML Deployment

An end-to-end machine learning project for predicting credit default risk, comparing multiple classification models, and explaining predictions using SHAP.

## Features

- Credit default probability prediction
- 5-model comparison:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Extra Trees
  - HistGradientBoosting
- Stratified 5-fold cross-validation
- Automatic best-model selection using ROC-AUC
- Model evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
- SHAP global and local explainability
- FastAPI prediction API
- Streamlit web interface
- Git/GitHub feature-branch workflow

## Architecture

```text
Data
 ↓
Preprocessing
 ↓
5-Fold Cross-Validation
 ↓
Model Comparison
 ↓
Best Model
 ↓
Final Test Evaluation
 ↓
Model Serialization
 ↓
SHAP Explainability
 ↓
FastAPI
 ↓
Streamlit
```

## Project Structure

```text
ml-model-deployment/
├── api/
│   └── main.py
├── models/
├── results/
│   └── shap/
├── src/
│   ├── data.py
│   ├── train.py
│   ├── model.py
│   └── explain.py
├── app.py
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python src/train.py
```

## Generate SHAP Explanations

```bash
python src/explain.py
```

## Run FastAPI

```bash
uvicorn api.main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## Run Streamlit

In another terminal:

```bash
streamlit run app.py
```

## Tech Stack

Python • scikit-learn • SHAP • pandas • FastAPI • Streamlit • Git • GitHub

## Roadmap

- Real-world credit risk dataset
- Hyperparameter tuning across candidate models
- PR-AUC, KS and calibration analysis
- Automated testing with pytest
- Docker
- GitHub Actions CI/CD
- Model monitoring

## Disclaimer

This project is for educational and portfolio purposes. Predictions should not be used for real lending or credit decisions without appropriate validation, governance, and regulatory review.