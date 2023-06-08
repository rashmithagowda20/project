import numpy as np
import pickle
import streamlit as st

# Load ML model
model = pickle.load(open('model.pkl', 'rb'))

# Create a function for the home page
def home():
    st.markdown('<h1>Heart Disease Classifier</h1>', unsafe_allow_html=True)

# Create a function for the prediction page
def predict():
    # Get user input using Streamlit widgets
    age = st.number_input("Age")
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure")
    # ... add more input fields as needed

    # Preprocess the features
    features = [age, sex, cp, trestbps]  # Modify this based on your input fields and feature order
    array_features = np.array(features).reshape(1, -1)

    # Predict the output
    prediction = model.predict(array_features)

    # Display the prediction result
    if prediction == 1:
        result = "The patient is not likely to have heart disease!"
    else:
        result = "The patient is likely to have heart disease!"
    st.markdown(f'<h3>{result}</h3>', unsafe_allow_html=True)

# Create the Streamlit app
def main():
    st.set_page_config(page_title="Heart Disease Classifier")
    home()  # Display the home page

    # Add a navigation menu
    pages = {
        "Home": home,
        "Predict": predict
    }
    page = st.sidebar.radio("Navigation", list(pages.keys()))
    pages[page]()  # Display the selected page

if __name__ == '__main__':
    main()
