import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Title
st.title("Restaurant Review Sentiment Analysis (Simple Streamlit App)")

# Load dataset
st.subheader("Dataset Loaded Successfully")

dataset_path = r"C:\Users\mohap\PycharmProjects\pythonProject\4.CUSTOMERS REVIEW DATASET\Restaurant_Reviews.tsv"

dataset = pd.read_csv(dataset_path, delimiter="\t", quoting=3)

st.write("Dataset Preview:")
st.dataframe(dataset.head())

# Text Cleaning
st.subheader("Cleaning Text...")

corpus = []
nltk.download('stopwords')

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

st.success("Text cleaning completed!")

# TF-IDF Vectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

# Train model
st.subheader("Training Decision Tree Model...")
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Predict
y_pred = classifier.predict(X_test)

# Metrics
st.subheader("Model Performance")

cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
bias = classifier.score(X_train, y_train)
variance = classifier.score(X_test, y_test)

st.write("Confusion Matrix:")
st.write(cm)

st.write(f"Accuracy: **{ac:.2f}**")
st.write(f"Bias (Train Accuracy): **{bias:.2f}**")
st.write(f"Variance (Test Accuracy): **{variance:.2f}**")

# Prediction Section
st.subheader("Try Your Own Review")

user_review = st.text_area("Enter a restaurant review:")

if st.button("Predict"):
    # Clean text same as training
    new_review = re.sub('[^a-zA-Z]', ' ', user_review)
    new_review = new_review.lower()
    new_review = new_review.split()
    ps = PorterStemmer()
    new_review = [ps.stem(word) for word in new_review if word not in set(stopwords.words('english'))]
    new_review = ' '.join(new_review)

    # Vectorize
    new_vector = cv.transform([new_review]).toarray()

    # Predict
    prediction = classifier.predict(new_vector)[0]

    if prediction == 1:
        st.success("Sentiment: **Positive** 😊")
    else:
        st.error("Sentiment: **Negative** 😠")
