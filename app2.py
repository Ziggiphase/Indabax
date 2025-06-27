import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from featurize import FeatureInteraction, SND_Feature
from itertools import combinations

st.set_page_config(page_title="CO2 Prediction App", layout="centered")

st.title("🌍 CO2 Prediction with Sensor Data")
MODEL_FILENAME = "model.pkl"
if "retrained" not in st.session_state:
    st.session_state.retrained = False

def encode_device(df):
    enc_df = pd.get_dummies(df, columns=["device_name"], dtype=np.int64)
    return enc_df

def log_trans(enc_df):
    log_trans_df = enc_df.copy()
    for i in enc_df.columns[:-4]:
        log_trans_df[i] = np.log10(1 + log_trans_df[i])
    return log_trans_df

def train_model(df):
    X = df.drop(['CO2', 'ID'], axis=1)
    y = df['CO2']
    model = ExtraTreesRegressor(random_state=42)
    model.fit(X, y)
    return model

def predicted(features, model):
    input_df = pd.DataFrame([features])
    if input_df['device_name'][0] == 'alpha':
        input_df['device_name_alpha'] = 1
        input_df['device_name_beta'] = 0
        input_df['device_name_charlie'] = 0
    elif input_df["device_name"][0] == 'beta':
        input_df['device_name_alpha'] = 0
        input_df['device_name_beta'] = 1
        input_df['device_name_charlie'] = 0
    elif input_df['device_name'][0] == 'charlie':
        input_df['device_name_alpha'] = 0
        input_df['device_name_beta'] = 0
        input_df['device_name_charlie'] = 1
    else:
        input_df = input_df
        
    input_df = input_df.drop('device_name', axis=1)
    input_df = log_trans(input_df)
    input_df = FeatureInteraction(input_df, ["MQ7_analog","MQ9_analog","MG811_analog","MQ135_analog"])
    input_df = SND_Feature(input_df, ["Temperature", "Humidity", "MQ7_analog", "MQ9_analog", "MG811_analog", "MQ135_analog"])
    
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted CO2: {prediction:.2f}")

# --------------------------
# Sidebar: Select section
# --------------------------
section = st.sidebar.selectbox(
    "Select App Section",
    ["Home", "Predict CO2", "Add New Dataset & Retrain Model", "About Project"]
)

# --------------------------
# New Page 1: Home Page
# --------------------------
if section == "Home":
    st.header("🏡 Welcome to CO2 Prediction App")
    st.markdown("""
    ### 🌿 Turning Sensor Data into Clean Air Insights
    This app uses sensor readings and machine learning to estimate **CO₂ concentration levels** in the air.
    
    Predict air quality using advanced regression models built from:
    - Temperature & Humidity
    - Gas Sensors (MQ7, MQ9, MG811, MQ135)
    - Device-based differentiation

    ---
    """)
    st.markdown("Learn more in the **About** section.")

# --------------------------
# Section 1: Predict CO2
# --------------------------
elif section == "Predict CO2":
    st.header("🔍 Section 1: Predict CO2 using Existing Model")

    if os.path.exists(MODEL_FILENAME):
        with open(MODEL_FILENAME, "rb") as f:
            model = pickle.load(f)
        #st.success(f"Model loaded from `{MODEL_FILENAME}`.")

        st.subheader("Enter Input Features")
        with st.form("predict_form"):
            features = {
                "Temperature": st.number_input("Temperature"),
                "Humidity": st.number_input("Humidity"),
                "MQ7_analog": st.number_input("MQ7_analog"),
                "MQ9_analog": st.number_input("MQ9_analog"),
                "MG811_analog": st.number_input("MG811_analog"),
                "MQ135_analog": st.number_input("MQ135_analog"),
                "device_name": st.selectbox("Device", ["alpha", "beta", "charlie"])
            }
            submitted = st.form_submit_button("Predict CO2")
            if submitted:
                if features["Temperature"] == 0.0 and features["Humidity"] == 0.0:
                    st.warning("Please enter valid feature values before predicting.")
                else:
                    predicted(features, model)
    else:
        st.error(f"Model file `{MODEL_FILENAME}` not found in the current folder.")

# --------------------------
# Section 2: Merge, Retrain, Download
# --------------------------
elif section == "Add New Dataset & Retrain Model":
    st.header("🔄 Section 2: Merge Data, Retrain Model, and Download")
    existing_df = pd.read_csv("Train.csv")
    new_file = st.file_uploader("Upload new dataset to merge", type=["csv"], key="new")

    if new_file is not None:
        new_df = pd.read_csv(new_file)

        st.write("📊 Existing Dataset Preview")
        st.dataframe(existing_df.head())

        st.write("📊 New Dataset Preview")
        st.dataframe(new_df.head())


        if st.button("Merge & Retrain Model"):
            merged_df = pd.concat([existing_df, new_df], ignore_index=True)
            encoded_df = encode_device(merged_df)
            encoded_df.drop('ID', axis=1, inplace=True)
            log_trans_df = log_trans(encoded_df)
            input_df = FeatureInteraction(log_trans_df, ["MQ7_analog","MQ9_analog","MG811_analog","MQ135_analog"])
            input_df = SND_Feature(input_df, ["Temperature", "Humidity", "MQ7_analog", "MQ9_analog", "MG811_analog", "MQ135_analog"])
            input_df['ID'] = merged_df['ID']
            model = train_model(input_df)

            with open(MODEL_FILENAME, "wb") as f:
                pickle.dump(model, f)
            merged_df.to_csv("merged_dataset.csv", index=False)
            st.session_state.retrained = True
            st.success("✅ Model retrained and saved as `model.pkl`.")
            st.download_button("📥 Download Merged Dataset", data=merged_df.to_csv(index=False), file_name="merged_dataset.csv")
            with open(MODEL_FILENAME, "rb") as f:
                st.download_button("📥 Download Trained Model", data=f, file_name="new_model.pkl")

# --------------------------
# New Page 4: About Project
# --------------------------
elif section == "About Project":
    st.header("ℹ️ About This Project")
    st.markdown("""
    ### Project Summary
    This project focuses on **CO₂ level prediction using sensor data** from multiple air quality devices.
    Carbon emissions significantly contribute to climate change, and monitoring these emissions is essential for mitigating environmental impact. However, existing high-accuracy reference meters are prohibitively expensive. Chemotronix, a company developing low-cost sensors, seeks to build machine learning models that can accurately map sensor readings to CO2 levels measured by reference meters. This will enable affordable and scalable solutions for tracking carbon emissions globally.

    The objective is to develop a machine learning model that accurately predicts CO2 levels using data from Chemotronix’s low-cost sensors. By achieving this, participants will help bridge the gap between affordability and precision in carbon emission tracking, enabling widespread adoption of low-cost monitoring technologies.

    An accurate prediction model will aid to manufacture low-cost sensors that rival the performance of expensive reference meters. This breakthrough has the potential to:

    Democratize access to environmental monitoring tools.
    Assist governments and organizations in implementing data-driven policies to curb carbon emissions.
    Promote sustainability by making emission tracking affordable for communities and industries worldwide.
    
    #### 🔧 Technologies Used
    - Python, Pandas, Scikit-learn, ExtraTreesRegressor
    - Streamlit Web Framework
    - Feature Engineering (Interaction Terms, Standardization)
    
    #### 📊 Dataset Info
    The dataset includes:
    - Sensor readings from MQ7, MQ9, MG811, MQ135
    - Environmental data: temperature and humidity
    - Device identity (alpha, beta, charlie)
    
    #### 👨‍💻 Developer
    - Developed as part of a DLI Project 2025
    - Contact: bellobasit790@gmail.com, akimuodunola@gmail.com, yusufagboola@gmail.com
    """)
