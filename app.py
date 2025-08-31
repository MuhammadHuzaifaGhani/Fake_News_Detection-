# app.py
import streamlit as st
import joblib

# Load saved model & vectorizer
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# App title & description
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.write("This tool uses a machine learning model to classify news articles as **Fake** or **Real**.")

# Text input
news_input = st.text_area(" Enter a News Article:", height=200, placeholder="Paste or type your news content here...")

# Prediction
if st.button(" Check News"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)
        prediction_prob = model.predict_proba(transform_input)[0]

        if prediction[0] == 1:
            st.success(f"✅ The News is **Real**! (Confidence: {prediction_prob[1]*100:.2f}%)")
        else:
            st.error(f"❌ The News is **Fake**! (Confidence: {prediction_prob[0]*100:.2f}%)")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Built with  using Streamlit & Machine Learning</p>", unsafe_allow_html=True)

