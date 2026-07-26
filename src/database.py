
# database


# Step 1 — Import Library

# DATABASE OPERATIONS

# STEP 1 - IMPORT LIBRARY

import sqlite3


# Step 2 — Create Database Connection

#  CREATE DATABASE CONNECTION

def create_connection():

    connection = sqlite3.connect("database/fraud_detection.db")

    return connection

# Step 3 — Create Cursor

def create_cursor(connection):

    return connection.cursor()


# STEP 4 - # CREATE TABLE


def create_table():

    connection = create_connection()

    cursor = create_cursor(connection)

    cursor.execute("""

    CREATE TABLE IF NOT EXISTS fraud_predictions(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        transaction_time INTEGER,

        transaction_amount REAL,

        transaction_type TEXT,

        merchant_category TEXT,

        device_type TEXT,

        payment_method TEXT,

        risk_score REAL,

        fraud_probability REAL,

        prediction TEXT,

        prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP

    )

    """)

    connection.commit()

    connection.close()

    print("Fraud Predictions Table Ready!")


# ==========================================================
# STEP 6 - INSERT PREDICTION
# ==========================================================

def insert_prediction(

    transaction_time,
    transaction_amount,
    transaction_type,
    merchant_category,
    device_type,
    payment_method,
    risk_score,
    fraud_probability,
    prediction

):

    connection = create_connection()

    cursor = create_cursor(connection)

    cursor.execute("""

        INSERT INTO fraud_predictions(

            transaction_time,
            transaction_amount,
            transaction_type,
            merchant_category,
            device_type,
            payment_method,
            risk_score,
            fraud_probability,
            prediction

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

    """, (

        transaction_time,
        transaction_amount,
        transaction_type,
        merchant_category,
        device_type,
        payment_method,
        risk_score,
        fraud_probability,
        prediction

    ))

    connection.commit()

    connection.close()

    print("Prediction Saved Successfully!")

# ==========================================================
# STEP 9 - FETCH PREDICTIONS
# ==========================================================

def fetch_predictions():

    connection = create_connection()

    cursor = create_cursor(connection)

    cursor.execute("""

        SELECT * FROM fraud_predictions

    """)

    records = cursor.fetchall()

    connection.close()

    return records


#  Step 5 — Run the Script

# # MAIN FUNCTION

if __name__ == "__main__":

    predictions = fetch_predictions()

    for prediction in predictions:

        print(prediction)

    #create_table()