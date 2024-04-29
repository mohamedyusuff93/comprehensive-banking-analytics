import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
import pickle

df = pd.read_csv(r"C:/Users/srira.SRIRAM-SD/OneDrive/Desktop/Data science/Comprehensive banking analytics/final.csv")

model_credit_score = pickle.load(r'credit_score_model.pkl')
imputer_kmeans = pickle.load(r'imputer.pkl')
scaler_kmeans = pickle.load(r'scaler.pkl')
kmeans_model = pickle.load(r'kmeans_model.pkl')
model_credit_risk = pickle.load(r'rf_model.pkl')
scaler_credit_risk = pickle.load(r'scaler1.pkl')


columns_for_clustering = ['Age', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
                           'Credit_Utilization_Ratio', 'Credit_History_Age', 'Total_EMI_per_month',
                           'Monthly_Balance', 'Credit_Score', 'Annual_Income_log']

def predict_credit_score(input_values):
    features = np.array([input_values])
    return model_credit_score.predict(features)[0]

def predict_clustering(input_values):
    features = np.array(input_values).reshape(1, -1)

    sample_input = pd.DataFrame(features, columns=columns_for_clustering)

    sample_input[columns_for_clustering] = imputer_kmeans.transform(sample_input[columns_for_clustering])

    sample_input_scaled = scaler_kmeans.transform(sample_input[columns_for_clustering])

    predicted_cluster = kmeans_model.predict(sample_input_scaled)[0]

    return predicted_cluster

def predict_credit_risk(input_values):
    features = np.array([input_values])
    sample_input_scaled = scaler_credit_risk.transform(features)
    predicted_credit_risk = model_credit_risk.predict(sample_input_scaled)[0]
    return predicted_credit_risk

selected_option = st.sidebar.selectbox(
    "Main Menu",
    ["About Project", "Predictions"],
    format_func=lambda x: "🏠 About Project" if x == "About Project" else "🔮 Predictions"
)