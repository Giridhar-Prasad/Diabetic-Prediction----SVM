import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd

# Assuming that 'scaler' and 'classifier' are pre-trained and loaded
# scaler = ... (Load your StandardScaler)
# classifier = ... (Load your SVM classifier model)

# Mock-up standard scaler and classifier for demonstration
# scaler = StandardScaler()  # Replace with your trained scaler
# classifier = SVC()         # Replace with your trained classifier

diabetes_dataset = pd.read_csv("C:/Users/girid/OneDrive/Desktop/Diabetic Prediction_project/diabetes2.csv")
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
Y = diabetes_dataset['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Set page configuration
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# Title of the app
st.title("Diabetes Prediction System")

# Create input fields for the user to provide data
st.header("Input Patient Data")

pregnancies = st.number_input("Number of pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose level", min_value=0.0, value=0.0, step=0.1)
blood_pressure = st.number_input("Blood pressure", min_value=0.0, value=0.0, step=0.1)
skin_thickness = st.number_input("Skin thickness", min_value=0.0, value=0.0, step=0.1)
insulin = st.number_input("Insulin level", min_value=0.0, value=0.0, step=0.1)
bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, value=0.0, step=0.1)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.0, step=0.001)
age = st.number_input("Age", min_value=0, max_value=120, value=0)

# Prediction function
def predict_diabetes():
    try:
        # Collect user input from the input fields
        input_data = [
            pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age
        ]
        
        # Convert input data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)

        # Reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # Standardize the input data
        std_data = scaler.transform(input_data_reshaped)

        # Make prediction
        prediction = classifier.predict(std_data)

        # Display result
        if prediction[0] == 0:
            st.success("The person is not diabetic.")
        else:
            st.warning("The person is diabetic.")
    
    except ValueError:
        st.error("Please enter valid input values.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Create a submit button
if st.button("Predict Diabetes"):
    predict_diabetes()

# Create an exit button
if st.button("Exit"):
    st.stop()  # Stop the streamlit script
