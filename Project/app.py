import streamlit as st
import joblib
import numpy as np

model = joblib.load("model/rf_model.pkl")

st.title("Prediksi Lonjakan Kasus Covid-19 Berdasarkan Mobilitas")

retail = st.number_input("Retail & Recreation (%)", -100, 100, 0)
grocery = st.number_input("Grocery & Pharmacy (%)", -100, 100, 0)
transit = st.number_input("Transit Stations (%)", -100, 100, 0)
work = st.number_input("Workplaces (%)", -100, 100, 0)
residence = st.number_input("Residential (%)", -100, 100, 0)

if st.button("Prediksi"):
    data = np.array([[retail, grocery, transit, work, residence]])
    pred = model.predict(data)[0]
    st.success(f"Prediksi Lonjakan Kasus Baru: {int(pred)} kasus")