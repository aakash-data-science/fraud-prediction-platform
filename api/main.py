


# STEP 1 - IMPORT FASTAPI


from fastapi import FastAPI

from logger import logger

from pydantic import BaseModel

import pandas as pd

from src.database import insert_prediction , fetch_predictions

import joblib

# STEP 2 - CREATE FASTAPI APP

app = FastAPI()

logger.info("Fraud Detection API Started Successfully")

# LOAD TRAINED MODEL & PREPROCESSOR


model = joblib.load("models/fraud_detection_model.joblib")
preprocessor = joblib.load("models/preprocessor.joblib")

# ==========================================================
# REQUEST MODEL
# ==========================================================

class TransactionInput(BaseModel):

    TransactionTime: int
    TransactionAmount: float
    TransactionType: str
    MerchantCategory: str
    DeviceType: str
    PaymentMethod: str
    CardType: str
    CustomerAge: int
    CustomerGender: str
    CustomerRegion: str
    IsInternational: str
    DistanceFromHomeKM: float
    HourOfDay: int
    DayOfWeek: str
    PreviousTransactions24h: int
    AverageTransactionAmount: float
    FailedLoginAttempts: int
    RiskScore: float
    IsNewDevice: str
    IPReputation: str
    AccountAgeMonths: int
    BalanceBefore: float
    BalanceAfter: float
    
# STEP 3 - HOME ROUTE

@app.get("/")
def home():

    return {
        "message": "Welcome to Fraud Detection Platform API"
    }



# PREDICTION ENDPOINT


@app.post("/predict")
def predict(transaction: TransactionInput):

    logger.info("Prediction request received")

    # Convert input to dictionary
    transaction_dict = transaction.model_dump()

    # Convert dictionary to DataFrame
    transaction_df = pd.DataFrame([transaction_dict])

    # Preprocess the data
    processed_data = preprocessor.transform(transaction_df)

    # Make prediction
    prediction = model.predict(processed_data)[0]

    # Predict probability
    probability = model.predict_proba(processed_data)[0][1]

    # Convert prediction to label
    result = "Fraud" if prediction == 1 else "Legitimate"

    logger.info(f"Prediction Result: {result}")
    logger.info(f"Fraud Probability: {float(probability):.4f}")

    insert_prediction(
    transaction_time=transaction.TransactionTime,
    transaction_amount=transaction.TransactionAmount,
    transaction_type=transaction.TransactionType,
    merchant_category=transaction.MerchantCategory,
    device_type=transaction.DeviceType,
    payment_method=transaction.PaymentMethod,
    risk_score=transaction.RiskScore,
    fraud_probability=float(probability),
    prediction=result
    )
    
    return {
        "Prediction": result,
        "Fraud Probability": round(float(probability), 4)
    }

@app.get("/predictions")
def get_predictions():

    predictions = fetch_predictions()

    return predictions