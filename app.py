
import streamlit as st
import joblib

vectorizer = joblib.load("bow_vectorizer.pkl")
model = joblib.load("sentiment_model.pkl")

st.set_page_config(
    page_title="Social Media Sentiment Analysis",
    page_icon="💬"
)

st.title("Social Media Sentiment Analysis")
st.write("Analyze the sentiment of a tweet about ChatGPT or AI.")

tweet = st.text_area(
    "Enter a tweet:",
    placeholder="Type your tweet here..."
)

if st.button("Analyze Sentiment"):
    if tweet.strip():
        tweet_vector = vectorizer.transform([tweet])
        prediction = model.predict(tweet_vector)[0]

        st.subheader("Prediction Result")
        st.success(f"Sentiment: {str(prediction).capitalize()}")
    else:
        st.warning("Please enter a tweet first.")
