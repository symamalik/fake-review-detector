import streamlit as st
import pickle

# Load saved model and TF-IDF
with open("fake_review_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

st.title("🛒 Fake Review Detector")

review = st.text_area("Enter a review to check if it's Fake or Real:")

if st.button("Check Review"):
    if review.strip() == "":
        st.warning("Please enter some text first!")
    else:
        review_tfidf = tfidf.transform([review])
        prediction = model.predict(review_tfidf)[0]
        if prediction == 0:
            st.error("❌ This looks like a FAKE review")
        else:
            st.success("✅ This looks like a REAL review")
