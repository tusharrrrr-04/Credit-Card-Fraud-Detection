from pyexpat import features

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Load the Data set
data = pd.read_csv('creditcard 3.csv')

#Seperate legitimate and fraud transaction
legit = data[data.Class==0]
fraud = data[data['Class'] == 1]

#undersample legitimate transaction to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample,fraud],axis=0)

#split data into training and testing sets
X = data.drop('Class',axis=1)
Y = data['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

#Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#train logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train_scaled, Y_train)

#evaluate model performance
train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test), Y_test)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(Y_test, y_pred)

#Web app
st.title("Credit Card Fraud Detection")

#Display accuracy
st.subheader("Model Accuracy")
st.write(f"Train Accuracy: **{accuracy:.2f}**")

# Confusion Matrix Visualization
if st.checkbox("Show Confusion Matrix"):
    cm = confusion_matrix(Y_test, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    st.pyplot(fig)

# Input prediction
st.subheader("Check Transaction Status Manually")
input_text = st.text_input("Enter transaction details:")

if st.button("Check"):
    try:
        input_array = np.array([float(i) for i in input_text.split(",")])
        input_scaled = scaler.transform(input_array.reshape(1, -1))
        prediction = model.predict(input_scaled)[0]
        st.success("Legit Transaction" if prediction == 0 else "Fraud Transaction")
    except:
        st.error("Invalid input. Please ensure you’ve entered the correct transaction details.")