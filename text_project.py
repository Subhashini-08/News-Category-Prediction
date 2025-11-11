import pickle
import pandas as pd
import streamlit as st
import numpy as np

with open("Bestmodel.pkl","rb") as files:
     model = pickle.load(files)
     
with open("tfidf_vector.pkl","rb") as vector_files:
    vector = pickle.load(vector_files)

label_map = {
    0: "World News",
    1: "Sports News",
    2: "Business News",
    3: "Science and Technology News"
}


st.set_page_config(page_title="News Category Classifier", layout="centered")
st.title("📰 News Category Prediction App")
st.write("Enter one or more news headlines and find out their category!")

# Text input (single or multiple)
user_input = st.text_area(
    "Enter news headlines (one per line):",
    placeholder="Example:\nApple launches new iPhone with advanced features\nIndia wins cricket world cup",
    height=150
)


if st.button("Predict Category"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter at least one news headline.")
    else:
        # Split input by line to handle multiple texts
        texts = [line.strip() for line in user_input.split("\n") if line.strip()]

        # Transform with TF-IDF
        text_tfidf = vector.transform(texts)

        # Predict
        predictions = model.predict(text_tfidf)

        # Display results
        st.subheader("🧾 Predictions:")
        for txt, p in zip(texts, predictions):
            st.markdown(f"**Headline:** {txt}")
            st.markdown(f"**Predicted Category:** {label_map.get(p, 'Unknown')}")
            st.markdown("---")

