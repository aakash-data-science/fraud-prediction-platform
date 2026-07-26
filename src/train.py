# train.py

# Fraud Detection Platform

# 🚀 Next Step

# step 1 - Import Libraries

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import joblib

import json

import os

from sklearn.model_selection import train_test_split


#from sklearn.preprocessing import LabelEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier 

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)   

from imblearn.over_sampling import SMOTE

import shap

# step 2 -> LOAD DATASET

df = pd.read_csv("data/raw/synthetic_fraud_detection_250000.csv")

print("\nDataset Loaded Successfully!\n")



# STEP 3 -> DISPLAY FIRST 5 ROWS

print("=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)

print(df.head())


# STEP 4 -> DISPLAY LAST 5 ROWS


print("\n" + "=" * 60)
print("LAST 5 ROWS")
print("=" * 60)

print(df.tail())


# STEP 5 -> SHAPE OF DATASET

print("\n" + "=" * 60)
print("SHAPE OF DATASET")
print("=" * 60)

print(df.shape)


# STEP 6 -> COLUMN NAMES

print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)

print(df.columns.tolist())



# Step 7 — Dataset Information

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

df.info()


# Step 8 — Missing Values

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())



# Step 9 — Duplicate Records

print("\n" + "=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)

duplicates = df.duplicated().sum()

print(f"Duplicate Records : {duplicates}")



# Step 10 — Statistical Summary

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe())

# 📌 Phase 2 — Exploratory Data Analysis (EDA)


#  💻 Step 11 — Class Distribution

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

class_distribution = df["Class"].value_counts()

print(class_distribution)


# Step 12 — Fraud Percentage

fraud_percentage = (df["Class"].sum() / len(df)) * 100

print("\nFraud Percentage : {:.4f}%".format(fraud_percentage))


# 13 — Genuine Percentage


genuine_percentage = 100 - fraud_percentage

print("Genuine Percentage : {:.4f}%".format(genuine_percentage))


# Step 14 — Bar Chart


plt.figure(figsize=(6,5))

sns.countplot(x="Class", data=df)

plt.title("Fraud vs Genuine Transactions")

plt.xlabel("Transaction Class")

plt.ylabel("Count")

plt.show()


# Bar chart → sns.countplot()
# Histogram → sns.histplot()
# Box plot → sns.boxplot()
# Scatter plot → sns.scatterplot()
# Line chart → sns.lineplot()
# Pie chart → plt.pie() (from Matplotlib)



# Step 15 — Pie Chart

plt.figure(figsize=(6,6))

df["Class"].value_counts().plot(
    kind="pie",
    autopct="%1.2f%%",
    labels=["Genuine", "Fraud"]
)

plt.title("Transaction Distribution")

plt.ylabel("")

plt.show()


# 🚀 Next Phase: Data Quality & Preprocessing


# Step 16: Handle Duplicate Records


print("\n" + "=" * 60)
print("REMOVING DUPLICATE RECORDS")
print("=" * 60)

before_rows = df.shape[0]

df = df.drop_duplicates()

after_rows = df.shape[0]

print(f"Rows Before Removing Duplicates : {before_rows}")
print(f"Rows After Removing Duplicates  : {after_rows}")
print(f"Duplicates Removed              : {before_rows - after_rows}")


# Step 20 – Analyze Every FEATURE ANALYSIS


print("\n" + "=" * 70)
print("FEATURE ANALYSIS")
print("=" * 70)

feature_summary = pd.DataFrame({

    "Data Type": df.dtypes,
    "Missing Values": df.isnull().sum(),
    "Unique Values": df.nunique()

})

print(feature_summary)







# 🚀 Phase 3 – Feature Analysis & Preprocessing



# Step 21 – Separate Columns by Type

# IDENTIFY COLUMN TYPES



identifier_columns = [

    "TransactionID",
    "CustomerID",
    "MerchantID"

]

# CATEGORICAL FEATURES

categorical_features = [

    "TransactionType",
    "MerchantCategory",
    "DeviceType",
    "PaymentMethod",
    "CardType",
    "CustomerGender",
    "CustomerRegion",
    "IsInternational",
    "DayOfWeek",
    "IsNewDevice",
    "IPReputation"

]

# NUMERICAL FEATURES

numerical_features = [

    "TransactionTime",
    "TransactionAmount",
    "CustomerAge",
    "DistanceFromHomeKM",
    "HourOfDay",
    "PreviousTransactions24h",
    "AverageTransactionAmount",
    "FailedLoginAttempts",
    "RiskScore",
    "AccountAgeMonths",
    "BalanceBefore",
    "BalanceAfter"

]



# Step 22 - DISPLAY COLUMN GROUPS

print("\nIdentifier Columns")
print(identifier_columns)

print("\nCategorical Columns")
print(categorical_features)

print("\nNumerical Columns")
print(numerical_features)


# CREATE FEATURES & TARGET
# Step 17: Define Features and Target

X = df.drop(columns=["Class"])

y = df["Class"]

print("\nFeatures Shape :", X.shape)
print("Target Shape   :", y.shape)


# Step 23 – Drop Identifier Columns

print("\n" + "=" * 70)
print("DROPPING IDENTIFIER COLUMNS")
print("=" * 70)

X = X.drop(columns=identifier_columns)

print("Remaining Features :", X.shape[1])
print(X.columns.tolist())


# Step 18: Train-Test Split (Very Important)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])


# Step 19: Verify Class Distribution

# CLASS DISTRIBUTION AFTER SPLIT


print("\nTraining Class Distribution")
print(y_train.value_counts(normalize=True))

print("\nTesting Class Distribution")
print(y_test.value_counts(normalize=True))


# 📌 Phase 4 – Data Preprocessing

print("\n" + "=" * 70)
print("PREPROCESSING PIPELINE")
print("=" * 70)

# Step 28 —PREPROCESSING PIPELINE

preprocessor = ColumnTransformer(

    transformers=[
    (
        "num",
        StandardScaler(),
        numerical_features
    ),
    (
        "cat",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    )
])

print("\nPreprocessing Pipeline Created Successfully!")

# LOGISTIC REGRESSION PIPELINE

print("\n" + "=" * 70)
print("BUILDING LOGISTIC REGRESSION PIPELINE")
print("=" * 70)

balanced_logistic_pipeline = Pipeline(

    steps = [
(
    "preprocessor",
    preprocessor
),
(
    "classifier",
    LogisticRegression(
        random_state = 42,
        max_iter=1000,
         class_weight="balanced"

    )
)
    ]
)


print("Balanced Logistic Regression Pipeline Created Successfully!")

# Step 31 — Train the Pipeline

# TRAIN LOGISTIC REGRESSION MODEL

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION MODEL")
print("=" * 70)

balanced_logistic_pipeline.fit(X_train , y_train)

print("Balanced Model Trained Successfully")

# Step 32 — Predictions

print("\n" + "=" * 70)
print("MAKING PREDICTIONS")
print("=" * 70)

y_pred = balanced_logistic_pipeline.predict(X_test)

print("predictions Completed")

# Step 33 – Calculate metrics

# accuracy

accuracy = accuracy_score(y_test , y_pred)

print(f"Accuracy : {accuracy:.4f}")

# precision

precision = precision_score( y_test,y_pred,zero_division=0)

print(f"Precision : {precision:.4f}")

# # RECALL

recall = recall_score(y_test , y_pred , zero_division = 0)

print(f"Recall : {recall:.4f}")

# F1 score

f1 = f1_score(y_test , y_pred , zero_division =0)

print(f"f1 Score : {f1:.4f}")

# ROC AUC SCORE

y_prob = balanced_logistic_pipeline.predict_proba(X_test)[:,1]

roc_auc = roc_auc_score( y_test , y_prob)

print(f"ROC-AUC Score : {roc_auc:.4f}")

# CONFUSION MATRIX

cm = confusion_matrix( y_test , y_pred)

print("\n Confusion Matrix")

print(cm)

# CLASSIFICATION REPORT

print("\n Classification Report")

classification = classification_report( y_test , y_pred , zero_division =0)
print(classification)

# STEP 44 -  RANDOM FOREST PIPELINE


print("\n" + "=" * 70)
print("BUILDING RANDOM FOREST PIPELINE")
print("=" * 70)

balanced_random_forest_pipeline = Pipeline(
    steps = [
        (

        "preprocessor",
        preprocessor
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators = 100,
                class_weight="balanced",
                random_state = 42,
                n_jobs=-1
            )
        )
    ]
)

print("Random Forest Pipeline Created Successfully")

# Step 45 – Train Random Forest

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)


balanced_random_forest_pipeline.fit(X_train , y_train)

print("Balanced Random Forest Trained Successfully!")

# Step 46 – Predictions


print("\n" + "=" * 70)
print("RANDOM FOREST PREDICTIONS")
print("=" * 70)

 
rf_pred = balanced_random_forest_pipeline.predict(X_test)

rf_prob = balanced_random_forest_pipeline.predict_proba(X_test)[:,1]

print("Predictions Completed!")

# Step 47 – Evaluate Random Forest


print("\n" + "=" * 70)
print("RANDOM FOREST RESULTS")
print("=" * 70)


print(f" Accuracy : {accuracy_score(y_test , rf_pred):.4f}")
print(f"Precision : {precision_score(y_test, rf_pred, zero_division=0):.4f}")

print(f"Recall : {recall_score(y_test, rf_pred, zero_division=0):.4f}")

print(f"F1 Score : {f1_score(y_test, rf_pred, zero_division=0):.4f}")

print(f"ROC-AUC Score : {roc_auc_score(y_test, rf_prob):.4f}")


# Step 48 – Confusion Matrix

rf_cm = confusion_matrix(y_test , rf_pred)

print("\n Random Forest Confusion Matrix")

print(rf_cm)


# Step 49 – Classification Report

print("\nRandom Forest Classification Report")

cr = classification_report(y_test , rf_pred , zero_division =0)

print(cr)


# XG BOOST

# ==========================================================
# CALCULATE SCALE_POS_WEIGHT  FOR XGBOOST
# ==========================================================

print("\n" + "=" * 70)
print("CALCULATING SCALE_POS_WEIGHT")
print("=" * 70)

negative_class = (y_train == 0).sum()

positive_class = (y_train == 1).sum()

scale_pos_weight = negative_class / positive_class

print(f"Negative Samples : {negative_class}")

print(f"Positive Samples : {positive_class}")

print(f"Scale Pos Weight : {scale_pos_weight:.2f}")


# STEP 57 - XGBOOST PIPELINE



print("\n" + "=" * 70)
print("BUILDING XGBOOST PIPELINE")
print("=" * 70)

balanced_xgboost_pipeline = Pipeline(

    steps = [
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            XGBClassifier(

                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False
            )
        )
    ]
)

print("XGBoost BALANCED  Pipeline Created Successfully")

# Step 58 – Train XGBoost


print("\n" + "=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)


balanced_xgboost_pipeline.fit(X_train,y_train)

print("XGBoost Trained Successfully")

# Step 59 – Predictions

print("\n" + "=" * 70)
print("XGBOOST PREDICTIONS")
print("=" * 70)

xgb_pred = balanced_xgboost_pipeline.predict(X_test)

xgb_prob = balanced_xgboost_pipeline.predict_proba(X_test)[:,1]

print("Predictions Completed")

# Step 60 – Evaluation

print("\n" + "=" * 70)
print("XGBOOST RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy_score(y_test, xgb_pred):.4f}")

print(f"Precision : {precision_score(y_test, xgb_pred, zero_division=0):.4f}")

print(f"Recall : {recall_score(y_test, xgb_pred, zero_division=0):.4f}")

print(f"F1 Score : {f1_score(y_test, xgb_pred, zero_division=0):.4f}")

print(f"ROC-AUC Score : {roc_auc_score(y_test, xgb_prob):.4f}")

# Step 61 – Confusion Matrix

xgb_cm = confusion_matrix(

    y_test,

    xgb_pred

)

print("\nXGBoost Confusion Matrix")

print(xgb_cm)


# Step 62 – Classification Report

print("\nXGBoost Classification Report")

print(

    classification_report(

        y_test,

        xgb_pred,

        zero_division=0

    )

)


# 🚀 Phase 11 – SMOTE (Synthetic Minority Oversampling Technique)

# Step 72 – Create SMOTE Object


print("\n" + "=" * 70)
print("APPLYING SMOTE")
print("=" * 70)


smote = SMOTE(random_state=42)

print("SMOTE Object Created Successfully")


# Step 74 — Fit the Preprocessor on Training Data

# FIT PREPROCESSOR ON TRAINING DATA

print("\n" + "=" * 70)
print("FITTING PREPROCESSOR")
print("=" * 70)


X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("Training Data Shape :",X_train_processed.shape)

print("Testing Data Shape :",X_test_processed.shape)


# Step 75 — Apply SMOTE

# APPLY SMOTE TO PREPROCESSED TRAINING DATA

print("\n" + "=" * 70)
print("APPLYING SMOTE")
print("=" * 70)

X_train_balanced , y_train_balanced = smote.fit_resample( X_train_processed ,y_train)

print("SMOTE Applied Successfully")

print("\n Balanced Training Shape :",X_train_balanced.shape)

print("Balanced Target Distribution")

print(y_train_balanced.value_counts())


# 📌 Step 76 — Train Logistic Regression on SMOTE Data

# LOGISTIC REGRESSION AFTER SMOTE

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION ON SMOTE DATA")
print("=" * 70)

smote_logistic = LogisticRegression(

    random_state=42,

    max_iter=1000

)

smote_logistic.fit(

    X_train_balanced,

    y_train_balanced

)

print("Model Trained Successfully!")

# 📌 Step 77 — Predictions

# SMOTE LOGISTIC PREDICTIONS

print("\n" + "=" * 70)
print("SMOTE LOGISTIC PREDICTIONS")
print("=" * 70)

smote_lr_pred = smote_logistic.predict(

    X_test_processed

)

smote_lr_prob = smote_logistic.predict_proba(

    X_test_processed

)[:,1]

print("Predictions Completed!")



# 📌 Step 78 — Evaluation

# # SMOTE LOGISTIC RESULTS


print("\n" + "=" * 70)
print("SMOTE LOGISTIC RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy_score(y_test, smote_lr_pred):.4f}")

print(f"Precision : {precision_score(y_test, smote_lr_pred, zero_division=0):.4f}")

print(f"Recall : {recall_score(y_test, smote_lr_pred, zero_division=0):.4f}")

print(f"F1 Score : {f1_score(y_test, smote_lr_pred, zero_division=0):.4f}")

print(f"ROC-AUC Score : {roc_auc_score(y_test, smote_lr_prob):.4f}")


# 📌 Step 79 — Confusion Matrix

#  SMOTE LOGISTIC CONFUSION MATRIX


smote_lr_cm = confusion_matrix(

    y_test,

    smote_lr_pred

)

print("\nSMOTE Logistic Confusion Matrix")

print(smote_lr_cm)

# Step 79A – Save Confusion Matrix

np.save(

    "models/confusion_matrix.npy",

    smote_lr_cm

)

print("\nConfusion Matrix Saved Successfully!")

# 📌 Step 80 — Classification Report


# SMOTE LOGISTIC CLASSIFICATION REPORT

# 📌 Step 80 — Classification Report

# SMOTE LOGISTIC CLASSIFICATION REPORT

print("\nSMOTE Logistic Classification Report")

classification = classification_report(

    y_test,

    smote_lr_pred,

    zero_division=0,

    output_dict=True

)

print(pd.DataFrame(classification).transpose())

# Step 80A – Save Classification Report

classification_df = pd.DataFrame(classification).transpose()

classification_df.to_csv(

    "models/classification_report.csv"

)

print("\nClassification Report Saved Successfully!")


# STEP 81 - # MODEL COMPARISON

# STEP 81 - MODEL COMPARISON

results = {

    "Model": [

        "Balanced Logistic Regression",

        "Balanced Random Forest",

        "Balanced XGBoost",

        "SMOTE + Logistic Regression"

    ],

    "Accuracy": [

        accuracy,

        accuracy_score(y_test, rf_pred),

        accuracy_score(y_test, xgb_pred),

        accuracy_score(y_test, smote_lr_pred)

    ],

    "Precision": [

        precision,

        precision_score(y_test, rf_pred, zero_division=0),

        precision_score(y_test, xgb_pred, zero_division=0),

        precision_score(y_test, smote_lr_pred, zero_division=0)

    ],

    "Recall": [

        recall,

        recall_score(y_test, rf_pred, zero_division=0),

        recall_score(y_test, xgb_pred, zero_division=0),

        recall_score(y_test, smote_lr_pred, zero_division=0)

    ],

    "F1 Score": [

        f1,

        f1_score(y_test, rf_pred, zero_division=0),

        f1_score(y_test, xgb_pred, zero_division=0),

        f1_score(y_test, smote_lr_pred, zero_division=0)

    ],

    "ROC-AUC": [

        roc_auc_score(y_test, y_prob),

        roc_auc_score(y_test, rf_prob),

        roc_auc_score(y_test, xgb_prob),

        roc_auc_score(y_test, smote_lr_prob)

    ]

}

# Step 82 – Create DataFrame

comparison_df = pd.DataFrame(

    results

)

# Step 83 – Display Comparison


print("\n" + "=" * 80)

print("MODEL COMPARISON")

print("=" * 80)

print(comparison_df)

# Step 83A – Save Model Comparison

comparison_df.to_csv(

    "models/model_comparison.csv",

    index=False

)

print("\nModel Comparison Saved Successfully!")

# Step 84 – Best Model


print("\n" + "=" * 80)

print("BEST MODEL")

print("=" * 80)

best_model = comparison_df.loc[

    comparison_df["Recall"].idxmax()

]

print(best_model)


# Step 81 — Extract Feature Names


print("\n" + "=" * 70)
print("EXTRACTING FEATURE NAMES")
print("=" * 70)

feature_names = preprocessor.get_feature_names_out()

print("Total Features :", len(feature_names))

print(feature_names)


# Step 82 EXTRACT FEATURE IMPORTANCE


print("\n" + "=" * 70)
print("EXTRACTING FEATURE IMPORTANCE")
print("=" * 70)

coefficients = smote_logistic.coef_[0]

print("Total Coefficients :", len(coefficients))


# STEP 83 - # FEATURE IMPORTANCE DATAFRAME

feature_importance = pd.DataFrame({

    "Feature": feature_names,

    "Coefficient": coefficients

})

feature_importance["Absolute Coefficient"] = (

    feature_importance["Coefficient"].abs()

)

feature_importance = feature_importance.sort_values(

    by="Absolute Coefficient",

    ascending=False

)

print(feature_importance.head(20))


# Step 83A – Save Feature Importance

feature_importance.to_csv(

    "models/feature_importance.csv",

    index=False

)

print("\nFeature Importance Saved Successfully!")

# Step 84 TOP 20 IMPORTANT FEATURES


plt.figure(figsize=(12,8))

top_features = feature_importance.head(20)

plt.barh(

    top_features["Feature"],

    top_features["Absolute Coefficient"]

)

plt.title("Top 20 Important Features")

plt.xlabel("Absolute Coefficient")

plt.ylabel("Feature")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()



# 🚀 Phase 14 — ROC Curve


# STEP 90 - # ROC CURVE

print("\n" + "=" * 70)
print("CALCULATING ROC CURVE")
print("=" * 70)

fpr, tpr, thresholds = roc_curve(

    y_test,

    smote_lr_prob

)

roc_auc = auc(

    fpr,

    tpr

)

print(f"ROC-AUC : {roc_auc:.4f}")

# Step 90A – Save ROC Curve Data

roc_curve_df = pd.DataFrame({

    "False Positive Rate": fpr,

    "True Positive Rate": tpr,

    "Threshold": thresholds

})

roc_curve_df.to_csv(

    "models/roc_curve.csv",

    index=False

)

print("\nROC Curve Data Saved Successfully!")

# STEP 91 - ROC CURVE GRAPH


plt.figure(figsize=(8,6))

plt.plot(

    fpr,

    tpr,

    label=f"ROC Curve (AUC = {roc_auc:.4f})"

)

plt.plot(

    [0,1],

    [0,1],

    linestyle="--"

)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.show()



# Step 93 –  Precision–Recall Curve


print("\n" + "=" * 70)
print("CALCULATING PRECISION-RECALL CURVE")
print("=" * 70)


pr_precision, pr_recall, pr_thresholds = precision_recall_curve(

    y_test,

    smote_lr_prob

)

average_precision = average_precision_score(

    y_test,

    smote_lr_prob

)

print(f"Average Precision Score : {average_precision:.4f}")



# Step 93A – Save Precision-Recall Curve Data

pr_curve_df = pd.DataFrame({

    "Recall": pr_recall,

    "Precision": pr_precision,

    "Threshold": np.append(pr_thresholds, np.nan)

})

pr_curve_df.to_csv(

    "models/precision_recall_curve.csv",

    index=False

)

print("\nPrecision-Recall Curve Data Saved Successfully!")


# STEP 94 - PRECISION-RECALL CURVE GRAPH


plt.plot(

    pr_recall,

    pr_precision,

    label=f"PR Curve (AP = {average_precision:.4f})"

)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision-Recall Curve")

plt.legend()

plt.grid(True)

plt.show()


# 🚀 Phase 17 — SHAP Explainability


# Step 97 - SHAP EXPLAINER

print("\n" + "=" * 70)
print("CREATING SHAP EXPLAINER")
print("=" * 70)

explainer = shap.LinearExplainer(

    smote_logistic,

    X_train_balanced

)

print("SHAP Explainer Created Successfully!")


# Step 98 — Calculate SHAP Values


print("\n" + "=" * 70)
print("CALCULATING SHAP VALUES")
print("=" * 70)

shap_values = explainer.shap_values(

    X_test_processed

)

print("SHAP Values Calculated Successfully!")


# Step 99 — SHAP Summary Plot

# SHAP SUMMARY PLOT

print("\n" + "=" * 70)
print("SHAP SUMMARY PLOT")
print("=" * 70)

shap.summary_plot(

    shap_values,

    X_test_processed,

    feature_names=feature_names

)


#  🚀 Phase 13 – Save the Best Model

# EXTRACT FEATURE NAMES




# Step 85 – Import Joblib

# Step 86 – Save the Best Model

#  # SAVE BEST MODEL


# Step 84A – Save Model Metrics

print("\n" + "=" * 70)
print("SAVING MODEL METRICS")
print("=" * 70)

metrics = {

    "Accuracy": float(accuracy_score(y_test, smote_lr_pred)),
    "Precision": float(precision_score(y_test, smote_lr_pred, zero_division=0)),
    "Recall": float(recall_score(y_test, smote_lr_pred, zero_division=0)),
    "F1 Score": float(f1_score(y_test, smote_lr_pred, zero_division=0)),
    "ROC-AUC": float(roc_auc_score(y_test, smote_lr_prob)),
    "Average Precision": float(average_precision)

}

with open("models/metrics.json", "w") as f:

    json.dump(metrics, f, indent=4)

print("Model Metrics Saved Successfully!")




print("\n" + "=" * 70)
print("SAVING BEST MODEL")
print("=" * 70)

joblib.dump(

    smote_logistic,

    "models/fraud_detection_model.joblib"

)

print("Best Model Saved Successfully!")


# Step 87 – Save the Preprocessor


# # SAVE PREPROCESSOR


print("\n" + "=" * 70)
print("SAVING PREPROCESSOR")
print("=" * 70)

joblib.dump(

    preprocessor,

    "models/preprocessor.joblib"

)

print("Preprocessor Saved Successfully!")


# Step 88 – Verify Files


# # VERIFY SAVED FILES

print("\n" + "=" * 70)
print("VERIFYING SAVED FILES")
print("=" * 70)

saved_model = joblib.load(

    "models/fraud_detection_model.joblib"

)

saved_preprocessor = joblib.load(

    "models/preprocessor.joblib"

)

print("Model Loaded Successfully!")

print("Preprocessor Loaded Successfully!")