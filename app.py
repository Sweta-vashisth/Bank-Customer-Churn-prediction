import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

sclr = StandardScaler()

# loading models
df = pickle.load(open('df.pkl', 'rb'))
rfc = pickle.load(open('rfc.pkl', 'rb'))

def prediction(CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary):
    # Check for empty strings and handle accordingly
    if CreditScore == '':
        st.error("Please provide a valid credit score.")
        return None
    if Geography == '':
        st.error("Please provide a valid country.")
        return None
    if Gender == '':
        st.error("Please provide a valid gender.")
        return None
    if Age == '':
        st.error("Please provide a valid age.")
        return None
    if Tenure == '':
        st.error("Please provide a valid tenure.")
        return None
    if Balance == '':
        st.error("Please provide a valid balance.")
        return None
    if NumOfProducts == '':
        st.error("Please provide a valid number of products.")
        return None
    if HasCrCard == '':
        st.error("Please provide a valid credit card status.")
        return None
    if IsActiveMember == '':
        st.error("Please provide a valid active member status.")
        return None
    if EstimatedSalary == '':
        st.error("Please provide a valid estimated salary.")
        return None

    features = np.array([[float(CreditScore), Geography, Gender, float(Age), float(Tenure), float(Balance), float(NumOfProducts), float(HasCrCard), float(IsActiveMember), float(EstimatedSalary)]])
    features = sclr.fit_transform(features)  # Scale the country column
    prediction = rfc.predict(features).reshape(1, -1)
    return prediction[0]

# web app
st.title('Bank Customer Churn Prediction')
CreditScore = st.number_input('Credit Score')
Geography = st.text_input('Country')
Gender = st.text_input('Gender')
Age = st.number_input('Age')
Tenure = st.number_input('Tenure')
Balance = st.number_input('Balance')
NumOfProducts = st.number_input('Products Number')
HasCrCard= st.number_input('Credit Card')
IsActiveMember = st.number_input('Active Member')
EstimatedSalary = st.number_input('Estimated Salary')

if st.button('Predict'):
    pred = prediction(CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary)

    if pred is not None:
        if pred == 1:
            st.write("The customer has left.")
        else:
            st.write("The customer is still active.")