
# predict


# Step 1 — Import Libraries


import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd

from database import insert_prediction


# Step 2 — Load Saved Model

print("\n" + "=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = joblib.load(

    "models/fraud_detection_model.joblib"

)

print("Model Loaded Successfully!")

# Step 3 — Load Saved Preprocessor


print("\n" + "=" * 70)
print("LOADING PREPROCESSOR")
print("=" * 70)

preprocessor = joblib.load(

    "models/preprocessor.joblib"

)

print("Preprocessor Loaded Successfully!")

# 🚀 Step 4 — Create a Sample Transaction

# SAMPLE TRANSACTION


print("\n" + "=" * 70)
print("CREATING SAMPLE TRANSACTION")
print("=" * 70)

sample_transaction = {

    "TransactionTime": 45231,
    "TransactionAmount": 8500.75,
    "TransactionType": "Online",
    "MerchantCategory": "Electronics",
    "DeviceType": "Mobile",
    "PaymentMethod": "Credit Card",
    "CardType": "Visa",
    "CustomerAge": 32,
    "CustomerGender": "Male",
    "CustomerRegion": "South",
    "IsInternational": "Yes",
    "DistanceFromHomeKM": 950.5,
    "HourOfDay": 2,
    "DayOfWeek": "Sunday",
    "PreviousTransactions24h": 1,
    "AverageTransactionAmount": 320.50,
    "FailedLoginAttempts": 5,
    "RiskScore": 89.5,
    "IsNewDevice": "Yes",
    "IPReputation": "High",
    "AccountAgeMonths": 12,
    "BalanceBefore": 250000.00,
    "BalanceAfter": 241499.25
}

print("Sample Transaction Created Successfully!")


# 🚀 Step 5 — Convert to DataFrame

print("\n" + "=" * 70)
print("CONVERTING TO DATAFRAME")
print("=" * 70)

input_data = pd.DataFrame([sample_transaction])

print(input_data)



# 🚀 Phase 19 — Continue predict.py


# Step 6 — Preprocess the Input

# PREPROCESS INPUT

print("\n" + "=" * 70)
print("PREPROCESSING INPUT")
print("=" * 70)

processed_input = preprocessor.transform(input_data)

print("Input Preprocessed Successfully!")

print("Processed Shape :", processed_input.shape)


# Step 7 — Predict Fraud Class

#  PREDICT CLASS

print("\n" + "=" * 70)
print("PREDICTING TRANSACTION")
print("=" * 70)

prediction = model.predict(processed_input)

print("Prediction Completed!")


# Step 8 — Predict Fraud Probability

# PREDICT PROBABILITY

probability = model.predict_proba(processed_input)

fraud_probability = probability[0][1]

genuine_probability = probability[0][0]

# 🚀 Step 8 — Save the Prediction from database

insert_prediction(

    transaction_time=sample_transaction["TransactionTime"],

    transaction_amount=sample_transaction["TransactionAmount"],

    transaction_type=sample_transaction["TransactionType"],

    merchant_category=sample_transaction["MerchantCategory"],

    device_type=sample_transaction["DeviceType"],

    payment_method=sample_transaction["PaymentMethod"],

    risk_score=sample_transaction["RiskScore"],

    fraud_probability=float(fraud_probability),

    prediction="Fraud" if prediction[0] == 1 else "Genuine"

)

# Step 9 — Display Result

print("\n" + "=" * 70)
print("PREDICTION RESULT")
print("=" * 70)

if prediction[0] == 1:

    print("🚨 Fraud Transaction Detected")

else:

    print("✅ Genuine Transaction")

print(f"\nFraud Probability    : {fraud_probability:.4f}")

print(f"Genuine Probability  : {genuine_probability:.4f}")