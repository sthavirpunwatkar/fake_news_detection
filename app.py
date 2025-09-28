import streamlit as st
import joblib
import string
import pandas as pd
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")

# Load stopwords
stop_words = set(stopwords.words("english"))

# Load saved models
log_model = joblib.load("fake_news_log_model.pkl")
nb_model = joblib.load("fake_news_nb_model.pkl")

# Load full dataset for vectorizer
news_df = pd.read_csv("data/combined_news.csv")

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Fit TF-IDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
news_df["clean_title"] = news_df["title"].apply(clean_text)
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(news_df["clean_title"])

# Prediction function
def predict_news(text, model):
    clean = clean_text(text)
    vect = vectorizer.transform([clean])
    pred = model.predict(vect)[0]
    return "Real" if pred == 1 else "Fake"

# Streamlit App
st.title("📰 Fake News Classifier")
st.write("Type a news headline and see if it's real or fake.")

headline = st.text_input("Enter a news headline here:")

if st.button("Predict"):
    pred_log = predict_news(headline, log_model)
    pred_nb = predict_news(headline, nb_model)
    st.write(f"**Logistic Regression Prediction:** {pred_log}")
    st.write(f"**Naive Bayes Prediction:** {pred_nb}")
