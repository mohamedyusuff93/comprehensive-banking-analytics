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
file_path = "C:/Users/srira.SRIRAM-SD/OneDrive/Desktop/Data science/Copper modeling/bank/Credit_score_model.pkl"
with open(file_path, 'rb') as f:
    model_credit_score = pickle.load(f)

file_path1 = "C:/Users/srira.SRIRAM-SD/OneDrive/Desktop/Data science/Copper modeling/bank/imputer.pkl"
with open(file_path1, 'rb') as f:
    imputer_kmeans = pickle.load(f)

file_path2 = "C:/Users/srira.SRIRAM-SD/OneDrive/Desktop/Data science/Copper modeling/bank/Scaler.pkl"
with open(file_path2, 'rb') as f:
    scaler_kmeans = pickle.load(f)

file_path3 = "C:/Users/srira.SRIRAM-SD/OneDrive/Desktop/Data science/Copper modeling/bank/Kmeans.pkl"
with open(file_path3, 'rb') as f:
    kmeans_model = pickle.load(f)

file_path4 = "C:/Users/srira.SRIRAM-SD/OneDrive/Desktop/Data science/Copper modeling/bank/rf_model.pkl"
with open(file_path4, 'rb') as f:
    model_credit_risk = pickle.load(f)

file_path5 = "C:/Users/srira.SRIRAM-SD/OneDrive/Desktop/Data science/Copper modeling/bank/Scaler1.pkl"
with open(file_path5, 'rb') as f:
    scaler_credit_risk = pickle.load(f)



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

if selected_option == "About Project":
    st.title("About Project")
    st.markdown("""
        ## Overview:
        This prediction app utilizes machine learning models for various financial predictions. The app is designed to provide insights into credit scores, clustering, credit risk, and investment amounts.

        ## Technologies Used:
        - Python
        - Streamlit
        - Scikit-learn
        - Pandas

        ## Features:
        - **Credit Score Prediction:**
          Users can input financial data, and the app predicts their credit score using a RandomForestClassifier model.

        - **Clustering Prediction:**
          Utilizes KMeans clustering to categorize users into clusters based on their financial attributes.

        - **Credit Risk Prediction:**
          Predicts whether a user is at credit risk or not using a RandomForestClassifier model.

        - **Investment Prediction:**
          Predicts the monthly amount invested based on financial features using a Linear Regression model.

        ## How to Use:
        - Select "Predictions" in the sidebar to access the prediction functionalities.
        - Input relevant financial information.
        - Click the "Predict" button for the desired prediction.

       
    """)
elif selected_option == "Predictions":
    st.title("Predictions")
    prediction_tab = st.sidebar.selectbox(
        "Select Prediction",
        ["Credit Score", "Clustering", "Credit Risk"]
    )
    def get_user_input(features):
        input_values = {}
        for feature in features:
            input_values[feature] = st.number_input(f"Enter value for {feature}:", value=0.0)
        return input_values
    if prediction_tab=="Credit Score":
        st.header('Credit Score Prediction')
        input_values_credit_score = get_user_input(['Log_Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
                                                    'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
                                                    'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Credit_Mix',
                                                    'Outstanding_Debt', 'Credit_History_Age', 'Monthly_Balance'])
        if st.button('Predict Credit Score'):
            predicted_credit_score = predict_credit_score(list(input_values_credit_score.values()))
            st.success(f'Predicted Credit Score: {predicted_credit_score}')
    elif prediction_tab == 'Clustering':
        st.header('Clustering Prediction')
        input_values_clustering = get_user_input(columns_for_clustering)
        if st.button('Predict Clustering'):
            predicted_cluster = predict_clustering(list(input_values_clustering.values()))
            st.success(f'Predicted Cluster: {predicted_cluster}')

    elif prediction_tab == 'Credit Risk':
        st.header('Credit Risk Prediction')
        input_values_credit_risk = get_user_input(['Age', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
                                                'Num_Credit_Card', 'Credit_Utilization_Ratio', 'Credit_History_Age',
                                                'Total_EMI_per_month', 'Monthly_Balance', 'Credit_Score',
                                                'Log_Annual_Income'])
        if st.button('Predict Credit Risk'):
            predicted_credit_risk = predict_credit_risk(list(input_values_credit_risk.values()))
            st.success(f'Predicted Credit Risk: {predicted_credit_risk}')