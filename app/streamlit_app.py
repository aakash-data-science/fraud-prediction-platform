# streamlit_app.py


# 1 . imoprt libraries


import streamlit as st

import requests

import sqlite3

import joblib

import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

import json
import numpy as np
import os

import seaborn as sns

import plotly.express as px

# Page Configuration


st.set_page_config(
    page_title="Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# LOAD TRAINED MODEL RESULTS
# ============================================================

MODELS_DIR = "models"

model = joblib.load(os.path.join(MODELS_DIR, "fraud_detection_model.joblib"))
preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))

try:

    with open(os.path.join(MODELS_DIR, "metrics.json"), "r") as f:
        metrics = json.load(f)

    model_comparison_df = pd.read_csv(
        os.path.join(MODELS_DIR, "model_comparison.csv")
    )

    classification_df = pd.read_csv(
        os.path.join(MODELS_DIR, "classification_report.csv")
    )

    feature_importance_df = pd.read_csv(
        os.path.join(MODELS_DIR, "feature_importance.csv")
    )

    confusion_matrix = np.load(
        os.path.join(MODELS_DIR, "confusion_matrix.npy")
    )

    roc_curve_df = pd.read_csv(
        os.path.join(MODELS_DIR, "roc_curve.csv")
    )

    pr_curve_df = pd.read_csv(
        os.path.join(MODELS_DIR, "precision_recall_curve.csv")
    )

except Exception as e:

    st.error(f"Unable to load model artifacts: {e}")

# Title


st.title("🛡️ Fraud Detection Platform")

st.write(
    "This application predicts whether a transaction is **Fraud** or **Legitimate** using a Machine Learning model."
)


# Sidebar


st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a Page",
    [
        "Prediction",
        "Prediction History",
        "Model Performance"
    ]
)

# --------------------------------------------------
# Prediction Page
# --------------------------------------------------

if page == "Prediction":

    st.header("Predict Fraud Transaction")

    # ============================================================
    # TWO COLUMN LAYOUT
    # ============================================================

    left_col, right_col = st.columns(2)

    

    # --------------------------------------------------
    # Customer Information
    # --------------------------------------------------

    
    with left_col:

        st.subheader("👤 Customer Information")

        customer_age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=30
        )

        customer_gender = st.selectbox(
            "Customer Gender",
            ["Male", "Female"]
        )

        customer_region = st.selectbox(
            "Customer Region",
            ["North", "South", "East", "West"]
        )

        is_international = st.selectbox(
            "International Transaction",
            ["Yes", "No"]
        )

    # ============================================================
    # TRANSACTION DETAILS
    # ============================================================

    with left_col:

        st.subheader("📍 Transaction Details")

        distance_from_home = st.number_input(
            "Distance From Home (KM)",
            min_value=0.0,
            value=10.0
        )

        hour_of_day = st.slider(
            "Hour of Day",
            0,
            23,
            12
        )

        day_of_week = st.selectbox(
            "Day of Week",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ]
        )

        previous_transactions = st.number_input(
            "Previous Transactions (24 Hours)",
            min_value=0,
            value=5
        )


    # ============================================================
    # RISK INFORMATION
    # ============================================================

    with right_col:

        st.subheader("⚠️ Risk Information")

        average_transaction_amount = st.number_input(
            "Average Transaction Amount",
            min_value=0.0,
            value=500.0
        )

        failed_login_attempts = st.number_input(
            "Failed Login Attempts",
            min_value=0,
            value=0
        )

        risk_score = st.slider(
            "Risk Score",
            0.0,
            100.0,
            50.0
        )

        is_new_device = st.selectbox(
            "New Device",
            [
                "Yes",
                "No"
            ]
        )

        ip_reputation = st.selectbox(
            "IP Reputation",
            [
                "Low",
                "Medium",
                "High"
            ]
        )

    # 🎯 Step 5 – Account Information

    with right_col:

        st.subheader("💳 Transaction Information")

        

    
        transaction_time = st.number_input(
        "Transaction Time",
        min_value=0,
        value=45231
        )

        transaction_amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=1000.0
        )

        transaction_type = st.selectbox(
            "Transaction Type",
            ["Online", "POS", "ATM"]
        )

        merchant_category = st.selectbox(
            "Merchant Category",
         [
            "Electronics",
            "Grocery",
            "Travel",
            "Restaurant",
            "Fashion",
            "Healthcare"
            ]
        )

        device_type = st.selectbox(
         "Device Type",
        [
            "Mobile",
            "Desktop",
            "Tablet"
        ]
        )

        payment_method = st.selectbox(
            "Payment Method",
        [
            "Credit Card",
            "Debit Card",
            "UPI",
            "Net Banking"
        ]
        )

        card_type = st.selectbox(
            "Card Type",
        [
            "Visa",
            "MasterCard",
            "RuPay"
        ]
        )

        account_age_months = st.number_input(
        "Account Age (Months)",
        min_value=0,
        value=24
        )

        balance_before = st.number_input(
        "Balance Before Transaction",
        min_value=0.0,
        value=10000.0
        )

        balance_after = st.number_input(
        "Balance After Transaction",
         min_value=0.0,
        value=9000.0
        )

    # 🎯 Step 6 – Add the Predict Button

    predict_button = st.button("🔍 Predict Fraud")

    if predict_button:

        transaction_data = {

            "TransactionTime": transaction_time,
            "TransactionAmount": transaction_amount,
            "TransactionType": transaction_type,
            "MerchantCategory": merchant_category,
            "DeviceType": device_type,
            "PaymentMethod": payment_method,
            "CardType": card_type,
            
            "CustomerAge": customer_age,
            "CustomerGender": customer_gender,
            "CustomerRegion": customer_region,
            "IsInternational": is_international,

            "DistanceFromHomeKM": distance_from_home,
            "HourOfDay": hour_of_day,
            "DayOfWeek": day_of_week,
            "PreviousTransactions24h": previous_transactions,

            "AverageTransactionAmount": average_transaction_amount,
            "FailedLoginAttempts": failed_login_attempts,
            "RiskScore": risk_score,
            "IsNewDevice": is_new_device,
            "IPReputation": ip_reputation,

            "AccountAgeMonths": account_age_months,
            "BalanceBefore": balance_before,
            "BalanceAfter": balance_after
        }
        
        try:

            with st.spinner("Analyzing transaction..."):

                input_df = pd.DataFrame([transaction_data])

                processed_data = preprocessor.transform(input_df)

                prediction_value = model.predict(processed_data)[0]

                probability = model.predict_proba(processed_data)[0][1]

                prediction = "Fraud" if prediction_value == 1 else "Legitimate"

                probability *= 100

               

                st.divider()

            if prediction == "Fraud":

                st.error("🚨 Fraud Transaction Detected")

                st.metric(
                label="Fraud Probability",
                value=f"{probability:.2f}%"
                )

                st.warning(
                "This transaction appears to be highly suspicious."
                )

            else:

                st.success("✅ Legitimate Transaction")

                st.metric(
                label="Fraud Probability",
                value=f"{probability:.2f}%"
                )

                st.info(
                "This transaction appears to be safe."
                )

        except Exception as e:

            st.error(f"Error: {e}")


    # Prediction History Page
    

elif page == "Prediction History":

    st.header("Prediction History")

    try:

        with st.spinner("Loading prediction history..."):

            db_path = "database/fraud_detection.db"

            if not os.path.exists(db_path):
                st.info("No prediction history available yet.")
                st.stop()

            conn = sqlite3.connect(db_path)

            cursor = conn.cursor()

            cursor.execute("""
                SELECT *
                FROM fraud_predictions
                ORDER BY id DESC
            """)

            history = cursor.fetchall()

            conn.close()

        if len(history) == 0:

            st.info("No prediction history found.")

        else:

            columns = [
                "ID",
                "Transaction Time",
                "Transaction Amount",
                "Transaction Type",
                "Merchant Category",
                "Device Type",
                "Payment Method",
                "Risk Score",
                "Fraud Probability",
                "Prediction",
                "Created At"
            ]

            history_df = pd.DataFrame(
                history,
                columns=columns
            )

            history_df["Transaction Amount"] = history_df["Transaction Amount"].map(
                lambda x: f"₹{x:,.2f}"
                )

            history_df["Fraud Probability"] = history_df["Fraud Probability"].map(
                lambda x: f"{x*100:.2f}%"
                )

            history_df = history_df.reset_index(drop=True)


            # ----------------------------------------
            # Search Prediction History
            # ----------------------------------------

            search = st.text_input(
            "🔍 Search Prediction History",
            placeholder="Search by Prediction, Payment Method, Merchant Category..."
            )

            if search:

                history_df = history_df[
                    history_df.astype(str)
                    .apply(
                        lambda row: row.str.contains(
                            search, 
                            case=False
                            ).any(),
                              axis=1)
            ]
            # ----------------------------------------
            # Prediction Filter
            # ----------------------------------------

            prediction_filter = st.selectbox(
                "Filter by Prediction",
                [
                        "All",
                        "Fraud",
                        "Legitimate"
                ]
            )

            if prediction_filter != "All":
                    history_df = history_df[
                        history_df["Prediction"] == prediction_filter
                ]

                    
            # ----------------------------------------
            # Dashboard Metrics
            # ----------------------------------------

            total_predictions = len(history_df)

            fraud_count = len(
            history_df[history_df["Prediction"] == "Fraud"]
            )

            legitimate_count = len(
            history_df[history_df["Prediction"] == "Legitimate"]
            )

            fraud_rate = (
            (fraud_count / total_predictions) * 100
            if total_predictions > 0
            else 0
            )

            
            st.markdown("""
            <style>

            .metric-card{
                background:#ffffff;
                border-radius:14px;
                padding:20px;
                text-align:center;
                box-shadow:0 2px 10px rgba(0,0,0,0.08);
                border:1px solid #E6E6E6;
            }

            .metric-title{
                font-size:15px;
                color:#666;
                font-weight:600;
            }   

            .metric-value{
                font-size:34px;
                font-weight:bold;
                margin-top:10px;
            }

            </style>
            """, unsafe_allow_html=True)

            c1,c2,c3,c4 = st.columns(4)

            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">📊 Total Predictions</div>
                    <div class="metric-value">{total_predictions}</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">🚨 Fraud</div>
                    <div class="metric-value" style="color:#E53935;">
                        {fraud_count}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">✅ Legitimate</div>
                    <div class="metric-value" style="color:#2E7D32;">
                        {legitimate_count}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">⚠ Fraud Rate</div>
                    <div class="metric-value" style="color:#FB8C00;">
                        {fraud_rate:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            # ----------------------------------------
            # Download Prediction History
            # ----------------------------------------

            csv = history_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Prediction History",
                data=csv,
                file_name="prediction_history.csv",
                mime="text/csv"
                )
            
            
            # ============================================================
            # DASHBOARD
            # ============================================================

            st.markdown("""
            <div style="
            padding:18px;
            border-radius:15px;
            background:linear-gradient(90deg,#0f172a,#1e3a8a);
            color:white;
            margin-top:15px;
            margin-bottom:20px;
            ">

            <h2 style="margin:0;">
            🛡 Fraud Detection Analytics Dashboard
            </h2>

            <p style="margin-top:8px;font-size:16px;">
            Interactive visualization of fraud detection predictions and transaction analytics.
            </p>

            </div>
            """,
            unsafe_allow_html=True)

            left_col, right_col = st.columns(2)

            # ============================================================
            # LEFT COLUMN
            # ============================================================

            with left_col:

                st.markdown("### 🚨 Fraud Distribution")

                prediction_counts = (
                    history_df["Prediction"]
                    .value_counts()
                    .reset_index()
                )

                prediction_counts.columns = [
                    "Prediction",
                    "Count"
                ]

                pie_fig = px.pie(
                    prediction_counts,
                    names="Prediction",
                    values="Count",
                    hole=0.55,
                    color="Prediction",
                    color_discrete_map={
                        "Fraud": "#E53935",
                        "Legitimate": "#43A047"
                    }
                )

                pie_fig.update_traces(
                    textposition="inside",
                    textinfo="percent+label"
                )

                pie_fig.update_layout(
                    showlegend=True,
                    height=420,
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(size=14)
                )

                st.plotly_chart(
                    pie_fig,
                    width="stretch"
                )

            # ============================================================
            # RIGHT COLUMN
            # ============================================================

            with right_col:

                st.markdown("### 💳 Payment Methods")



                payment_counts = (
                    history_df["Payment Method"]
                    .value_counts()
                    .reset_index()
                )

                payment_counts.columns = [
                        "Payment Method",
                        "Transactions"
                ]

              

            # ============================================================
            # SECOND ROW OF DASHBOARD
            # ============================================================

            trend_col, merchant_col = st.columns(2)

            # ============================================================
            # LEFT : Prediction Trend
            # ============================================================

            with trend_col:

                st.markdown("### 📈 Prediction Trend")

                trend_df = history_df.copy()

                trend_df["Created At"] = pd.to_datetime(
                trend_df["Created At"]
                )

                trend_df["Date"] = trend_df["Created At"].dt.date

                daily_predictions = (
                trend_df
                .groupby("Date")
                .size()
                .reset_index(name="Transactions")
                )

                trend_fig = px.line(
                daily_predictions,
                x="Date",
                y="Transactions",
                markers=True
                )

                trend_fig.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=20, b=10)
                )

                st.plotly_chart(
                trend_fig,
                width="stretch"
                )

            # ============================================================
            # RIGHT : Merchant Category ANALYTICS
            # ============================================================

            with merchant_col:

                st.markdown("### 🏪 Merchant Categories")

                merchant_counts = (
                    history_df["Merchant Category"]
                    .value_counts()
                    .reset_index()
                )

                merchant_counts.columns = [
                    "Merchant Category",
                    "Transactions"
                ]

                merchant_fig = px.bar(
                    merchant_counts,
                    x="Merchant Category",
                    y="Transactions",
                    text_auto=True,
                    color="Transactions",
                    color_continuous_scale="Purples"
                )

                

                merchant_fig.update_layout(
                    height=420,
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_title="",
                    yaxis_title="Transactions",
                    coloraxis_showscale=False
                )

                st.plotly_chart(
                    merchant_fig,
                    width="stretch"
                )

                payment_fig = px.bar(
                    payment_counts,
                    x="Payment Method",
                    y="Transactions",
                    text_auto=True,
                    color="Transactions",
                    color_continuous_scale="Blues"
                )

                payment_fig.update_layout(
                    height=420,
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_title="",
                    yaxis_title="Transactions",
                    coloraxis_showscale=False
                )

                st.plotly_chart(
                    payment_fig,
                    width="stretch"
                )
                
            
                
            st.dataframe(
                history_df.style.map(
                    lambda value:
                    "background-color:#ffcccc;color:red;font-weight:bold"
                    if value == "Fraud"
                    else (
                        "background-color:#d4edda;color:green;font-weight:bold"
                        if value == "Legitimate"
                        else ""
                    ),
                    subset=["Prediction"]
                ),
                 width="stretch",
                 hide_index=True
                 )
            

    except Exception as e:

        st.error(f"Error: {e}")


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    #✅ Step 4: Add Model Information Card

    st.divider()

    st.subheader("🤖 Model Information")

    info1, info2, info3 = st.columns(3)

    with info1:
        st.info("""
    **Algorithm**

    • Logistic Regression

    • Scikit-Learn Pipeline

    • Binary Classification
    """)

    with info2:
        st.info("""
    **Training Dataset**

    • Synthetic Fraud Detection Dataset

    • 250,000 Transactions

    • SMOTE Applied for Class Balancing
    """)

    with info3:
        st.info("""
    **Model Details**

    • 23 Input Features

    • Probability-Based Prediction

    • Binary Fraud Classification
    """)

    st.header("📊 Model Performance Dashboard")

    st.write(
        "Performance metrics of the trained Fraud Detection model."
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "Accuracy",
            f"{metrics['Accuracy'] * 100:.2f}%"
        )

    with c2:
        st.metric(
        "Precision",
        f"{metrics['Precision'] * 100:.2f}%"
        )

    with c3:
        st.metric(
        "Recall",
        f"{metrics['Recall'] * 100:.2f}%"
        )

    with c4:
        st.metric(
        "F1 Score",
        f"{metrics['F1 Score'] * 100:.2f}%"
        )

    with c5:
        st.metric(
        "ROC-AUC",
        f"{metrics['ROC-AUC'] * 100:.2f}%"
        )


    # ✅ Step 5: Add Dataset Statistics

    st.divider()

    st.subheader("📊 Dataset Statistics")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Training Samples",
            "200,000"
        )

    with c2:
        st.metric(
            "Testing Samples",
            "50,000"
        )

    with c3:
        st.metric(
            "Fraud Cases",
            "23"
        )

    with c4:
        st.metric(
            "Input Features",
            "23"
        )

    # ✅ Step 6: Add Performance Summary

    st.divider()

    st.subheader("📝 Performance Summary")

    st.info(f"""
    ### Model Performance Summary

    ✔ Accuracy: **{metrics['Accuracy'] * 100:.2f}%**

    ✔ Precision: **{metrics['Precision'] * 100:.2f}%**

    ✔ Recall: **{metrics['Recall'] * 100:.2f}%**

    ✔ F1 Score: **{metrics['F1 Score'] * 100:.2f}%**

    ✔ ROC-AUC Score: **{metrics['ROC-AUC'] * 100:.2f}%**

    **Interpretation**

    • The model achieves high overall accuracy and a strong ROC-AUC score.

    • The recall indicates that the model detects 60% of fraud cases in the test set.

    • The low precision suggests that many transactions flagged as fraud are actually legitimate, indicating room for improvement in reducing false positives.

    • These metrics provide a realistic assessment of the current model rather than relying on hardcoded statements.
    """)   

    # ⭐ Step 7: Add Classification Report Table

    st.divider()

    st.subheader("📋 Classification Report")

    classification_display = classification_df.copy()

    classification_display = classification_display.rename(
        columns={"Unnamed: 0": "Class"}
    )

    classification_display["Class"] = classification_display["Class"].replace({
        "0": "Legitimate",
        "1": "Fraud"
    })

    st.dataframe(
        classification_display,
        width="stretch",
        hide_index=True
    )


    # ⭐ Step 8: Confusion Matrix Heatmap

    st.divider()

    st.subheader("📊 Confusion Matrix")

    confusion_df = pd.DataFrame(
        confusion_matrix,
        columns=[
            "Predicted Legitimate",
            "Predicted Fraud"
        ],
        index=[
            "Actual Legitimate",
            "Actual Fraud"
        ]
    )

    heatmap = px.imshow(

        confusion_df,

        text_auto=True,

        color_continuous_scale="Blues",

        aspect="auto"

    )   

    heatmap.update_layout(

        height=420,

        xaxis_title="Predicted",

        yaxis_title="Actual",

        margin=dict(
            l=20,
            r=20,
            t=40,
            b=20
        ),

        coloraxis_colorbar=dict(
        title="Count"
        )
    )

    st.plotly_chart(
        heatmap,
        width="stretch"
)
    
    # ⭐ Step 9: Feature Importance

    st.divider()

    st.subheader("⭐ Top 15 Most Feature Importance")

    # to show all feauture -> feature_df = feature_importance_df.copy()

    feature_df = (
        feature_importance_df
        .sort_values(
            by="Absolute Coefficient",
            ascending=False
        )
        .head(15)
        .sort_values(
            by="Absolute Coefficient",
            ascending=True
        )
    )

    feature_df["Feature"] = (
    feature_df["Feature"]
        .str.replace("cat__", "", regex=False)
        .str.replace("num__", "", regex=False)
        .str.replace("_", " ", regex=False)
    )

    # ============================================================
    # MAKE FEATURE NAMES PROFESSIONAL
    # ============================================================

    feature_df["Feature"] = feature_df["Feature"].replace({

        "IPReputation Suspicious": "IP Reputation (Suspicious)",
        "IPReputation Blacklisted": "IP Reputation (Blacklisted)",

        "PaymentMethod UPI": "Payment Method (UPI)",
        "PaymentMethod Credit Card": "Payment Method (Credit Card)",
        "PaymentMethod Debit Card": "Payment Method (Debit Card)",
        "PaymentMethod Net Banking": "Payment Method (Net Banking)",

        "MerchantCategory Restaurant": "Merchant Category (Restaurant)",
        "MerchantCategory Electronics": "Merchant Category (Electronics)",
        "MerchantCategory Grocery": "Merchant Category (Grocery)",
        "MerchantCategory Fashion": "Merchant Category (Fashion)",
        "MerchantCategory Travel": "Merchant Category (Travel)",

        "CustomerRegion West": "Customer Region (West)",
        "CustomerRegion East": "Customer Region (East)",
        "CustomerRegion South": "Customer Region (South)",
        "CustomerRegion North": "Customer Region (North)",

        "DeviceType Mobile": "Device Type (Mobile)",
        "DeviceType Desktop": "Device Type (Desktop)",
        "DeviceType Tablet": "Device Type (Tablet)",

        "CustomerGender Male": "Customer Gender (Male)",
        "CustomerGender Female": "Customer Gender (Female)",

        "TransactionType POS": "Transaction Type (POS)",
        "TransactionType ATM": "Transaction Type (ATM)",
        "TransactionType Online": "Transaction Type (Online)",
        "TransactionType Transfer": "Transaction Type (Transfer)",

        "DayOfWeek Mon": "Monday",
        "DayOfWeek Tue": "Tuesday",
        "DayOfWeek Wed": "Wednesday",
        "DayOfWeek Thu": "Thursday",
        "DayOfWeek Fri": "Friday",
        "DayOfWeek Sat": "Saturday",
        "DayOfWeek Sun": "Sunday",

        "IsInternational Yes": "International Transaction",
        "IsInternational No": "Domestic Transaction",

        "IsNewDevice Yes": "New Device",
        "IsNewDevice No": "Known Device",

        "RiskScore": "Risk Score",

        "AverageTransactionAmount": "Average Transaction Amount",

        "DistanceFromHomeKM": "Distance From Home (KM)",

        "PreviousTransactions24h": "Previous Transactions (24h)",

        "FailedLoginAttempts": "Failed Login Attempts",

        "BalanceBefore": "Balance Before",

        "BalanceAfter": "Balance After"
    })
    
    
    feature_fig = px.bar(
    feature_df,
    x="Absolute Coefficient",
    y="Feature",
    orientation="h",
    text_auto=".3f",
    color="Absolute Coefficient",
    color_continuous_scale="Viridis"
    )

    feature_fig.update_traces(
    textposition="outside"
    )


    feature_fig.update_layout(

        title={

            "text":"Top 15 Most Important Features",

            "x":0.5,

            "xanchor":"center"

        },

        height=650,

        xaxis_title="Importance Score",

        yaxis_title="",

        font=dict(

            size=13

        ),

        margin=dict(

            l=40,

            r=20,

            t=70,

            b=30

        ),

        coloraxis_showscale=False
    )
    
    #feature_fig.update_layout(
     #   height=600,
      #  title="Top 15 Most Important Features",
       # xaxis_title="Importance",
        #yaxis_title="Features",
        #coloraxis_showscale=False,
        #margin=dict(
         #   l=20,
          #  r=20,
           # t=60,
            #b=20
        #)
    #)

    st.plotly_chart(
        feature_fig,
        width="stretch"
    )

    # ⭐ Step 10: Prediction Distribution

    st.divider()

    st.subheader("📈 Prediction Distribution")

    legitimate_count = int(confusion_matrix[0][0] + confusion_matrix[1][0])

    fraud_count = int(confusion_matrix[0][1] + confusion_matrix[1][1])

    distribution_df = pd.DataFrame(
        {
            "Prediction": [
                "Legitimate",
                "Fraud"
            ],
            "Count": [
                legitimate_count,
                fraud_count
            ]
        }
    )

    distribution_fig = px.pie(
        distribution_df,
        names="Prediction",
        values="Count",
        hole=0.55,
        color="Prediction",
        color_discrete_map={
            "Fraud":"red",
            "Legitimate":"green"
        }
    )

    #distribution_fig.update_layout(height=450)

    distribution_fig.update_layout(
        height=450,
        showlegend=True,
        legend_title="Prediction",
        margin=dict(
            l=20,
            r=20,
            t=40,
            b=20
        )
    )

    st.plotly_chart(
        distribution_fig,
        width="stretch"
    )

    # ============================================================
    # STEP 11 : ROC CURVE
    # ============================================================

    st.divider()

    st.subheader("📈 ROC Curve")

    roc_fig = px.line(
        roc_curve_df,
        x="False Positive Rate",
        y="True Positive Rate",
        title=f"ROC Curve (AUC = {metrics['ROC-AUC']:.4f})"
    )

    roc_fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Guess",
        line=dict(
            dash="dash",
            color="red"
        )
    )

    roc_fig.update_layout(
        height=500,
        xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    st.plotly_chart(
        roc_fig,
        width="stretch"
    )

    # ============================================================
    # STEP 12 : PRECISION RECALL CURVE
    # ============================================================

    st.divider()

    st.subheader("📈 Precision-Recall Curve")

    

    pr_fig = px.line(
        pr_curve_df,
        x="Recall",
        y="Precision",
        title="Precision-Recall Curve"
    )

    pr_fig.update_layout(
        height=500,
        xaxis_title="Recall",
        yaxis_title="Precision",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    st.plotly_chart(
        pr_fig,
        width="stretch"
    )

    # ============================================================
    # STEP 13 : MODEL COMPARISON
    # ============================================================

    st.divider()

    st.subheader("🏆 Model Comparison")

    st.dataframe(
        model_comparison_df,
        width="stretch",
        hide_index=True
    )

    comparison_fig = px.bar(
        model_comparison_df,
        x="Model",
        y="ROC-AUC",
        color="Model",
        text_auto=".3f"
    )

    comparison_fig.update_layout(
        height=450,
        xaxis_title="Model",
        yaxis_title="ROC-AUC Score",
        coloraxis_showscale=False
    )

    st.plotly_chart(
        comparison_fig,
        width="stretch"
    )
