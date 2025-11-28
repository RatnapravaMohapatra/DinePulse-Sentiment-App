import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# --------------------------------------------------
# INITIAL SETUP
# --------------------------------------------------
st.set_page_config(page_title="DinePulse Sentiment App", layout="wide")
st.title("🍽️ DinePulse Sentiment Analysis App")
st.write("Developed by **Ratna**")
st.markdown("---")

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
st.subheader("📂 Loading Dataset")

dataset_path = "Restaurant_Reviews.tsv"

try:
    dataset = pd.read_csv(dataset_path, delimiter="\t", quoting=3)
    st.success("Dataset loaded successfully!")
    st.write(dataset.head())
except FileNotFoundError:
    st.error("❌ ERROR: Dataset not found. Ensure Restaurant_Reviews.tsv is in the project root.")
    st.stop()

# --------------------------------------------------
# PREPROCESSING FUNCTION
# --------------------------------------------------
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_review(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    return " ".join(review)

# Clean entire dataset
corpus = [clean_review(review) for review in dataset['Review']]

# --------------------------------------------------
# TF-IDF VECTORIZATION
# --------------------------------------------------
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------
st.subheader("🤖 Model Training")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# --------------------------------------------------
# METRICS
# --------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
bias = classifier.score(X_train, y_train)
variance = classifier.score(X_test, y_test)

st.write("### 📊 Confusion Matrix")
st.write(cm)

st.write("### 🎯 Accuracy")
st.success(f"Model Accuracy: **{acc:.2f}**")

st.write("### 📈 Bias & Variance")
st.write(f"Bias (Train Accuracy): **{bias:.2f}**")
st.write(f"Variance (Test Accuracy): **{variance:.2f}**")

st.markdown("---")

# --------------------------------------------------
# LIVE PREDICTION
# --------------------------------------------------
st.header("🔮 Live Sentiment Prediction")

user_input = st.text_area("Enter a restaurant review:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = clean_review(user_input)
        vectorized = cv.transform([cleaned]).toarray()
        prediction = classifier.predict(vectorized)[0]

        if prediction == 1:
            st.success("🍀 **Positive Review**")
        else:
            st.error("⚠️ **Negative Review**")
