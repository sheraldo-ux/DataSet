import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Admission Chance.csv')
X = df.drop(['Chance of Admit ', 'Serial No'], axis=1)
scaler = StandardScaler().fit(X)

model_dict = {
    "Logistic Regression": joblib.load('model_lr'),
    "SVM": joblib.load('model_svm'),
    "KNN": joblib.load('model_knn'),
    "Random Forest": joblib.load('model_rf'),
    "Gradient Boosting": joblib.load('model_gr')
}

# Streamlit app title
st.title("Graduate Admission Prediction")

# Input fields
p1 = st.number_input("GRE Score (260-340):", min_value=260, max_value=340, step=1)
p2 = st.number_input("TOEFL Score (0-120):", min_value=0, max_value=120, step=1)
p3 = st.number_input("University Rating (1-5):", min_value=1.0, max_value=5.0, step=0.1)
p4 = st.number_input("SOP (1-5):", min_value=1.0, max_value=5.0, step=0.1)
p5 = st.number_input("LOR (1-5):", min_value=1.0, max_value=5.0, step=0.1)
p6 = st.number_input("CGPA (0-10):", min_value=0.0, max_value=10.0, step=0.1)
p7 = st.number_input("Research (0/1):", min_value=0, max_value=1, step=1)

# Model selection
selected_model = st.selectbox("Choose Model", ["Logistic Regression", "SVM", "KNN", "Random Forest", "Gradient Boosting"])

# Prediction
if st.button("Predict"):
    try:
        inputs = np.array([[p1, p2, p3, p4, p5, p6, p7]])
        scaled_inputs = scaler.transform(inputs)
        model = model_dict[selected_model]
        result = model.predict(scaled_inputs)

        if result[0] == 1:
            st.success("High Chance of getting admission!")
        else:
            st.error("Low Chance of Admission!")
    except Exception as e:
        st.error(f"Error: {str(e)}")
