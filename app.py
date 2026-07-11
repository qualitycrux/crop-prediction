import streamlit as st
import numpy as np
import pandas as pd
import pickle

#1. load model
model=pickle.load(open("model-crop-prediction.pkl","rb"))
#load scaler
scaler=pickle.load(open("scaler-crop-prediction.pkl","rb"))
#load encoder
le=pickle.load(open("encoder-crop-prediction.pkl","rb"))

def predict(N,P,K,temperature,humidity,ph,rainfall):
    # get input into dataframes
    input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    # scale the data
    input_scaled=scaler.transform(input_df)
    # predict the model results
    prediction=model.predict(input_scaled)
    result=le.inverse_transform(prediction)
    return result

st.title("Crop Prediction Application")


col1,col2= st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0, value=50.0)
    P = st.number_input("Phosphorous (P)", min_value=0.0, value=50.0)
    K = st.number_input("Potassium (K)", min_value=0.0, value=50.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)

with col2:
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, value=80.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=100.0)

if st.button("Predict the Crop"):
    output=predict(N,P,K,temperature, humidity, ph, rainfall)
    st.success(f"Your predicted crop is ={output}")

st.divider()
st.subheader("Sample data")
st.write("N - P - K - Temperature - Humidity - Soil Ph - Rainfall")
st.write("90	42	43	20.879744	82.002744	6.502985	202.935536	(rice)")
st.write("107	34	32	26.774637	66.413269	6.780064	177.774507	(coffee)")
st.write("89	58	35	23.986517	82.090534	6.096839	167.057646	(jute)")
st.write("98	44	21	25.771751	74.089114	6.524478	107.493192	(maize)")
st.divider()

st.markdown("Developed By Muhammad AKhtar")
