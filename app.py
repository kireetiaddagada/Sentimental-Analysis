import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title('Mental Health ANALYSIS')

st.header('USING NAIVE BAYES WE CAN TELL THE PARTICULAR STRESS THE PERSON IS FACING')


Posts = st.text_input("Enter how you are you feeling now")

if st.button("Predict"):
    vc = joblib.load('Vector_model.h5')
    Posts = vc.transform([Posts])
    model = joblib.load('reddit_model.h5')
    prediction = model.predict(Posts)
    st.success(prediction)
