# DinePulse-Sentiment-App(LIVE_https://dinepulse-sentiment-app-by-ratnaprava.streamlit.app/)
AI app that predicts restaurant review sentiment using NLP + ML. Includes TF-IDF, preprocessing, model training, and a clean Streamlit UI.

# 🍽️ DinePulse Sentiment App  
### **AI-Powered Restaurant Review Sentiment Analysis**  
**Developed by Ratna**

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-App-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/NLP-Project-green" />
  <img src="https://img.shields.io/badge/Status-Production Ready-brightgreen" />
</p>


## 📝 Overview  
**DinePulse Sentiment App** is a complete end-to-end **Natural Language Processing (NLP)** project that classifies restaurant reviews as **Positive** or **Negative**.

The project includes:

✔ Text cleaning & preprocessing  
✔ TF-IDF vectorization  
✔ Machine Learning model training  
✔ Performance evaluation  
✔ Streamlit Web Application  
✔ Real-time sentiment prediction  

This project is **recruiter-friendly**, **MNC-ready**, and ideal for **resume portfolios**.

---

## 🎯 Why This Project Stands Out (Benefits)

### ✔ Industry-standard NLP pipeline  
Includes preprocessing → training → evaluation → deployment.

### ✔ Recruiter-friendly  
Clean structure, proper documentation, modular code.

### ✔ Real-world dataset  
Uses **Restaurant_Reviews.tsv (1000 labeled reviews)**.

### ✔ Deployable  
Run on Streamlit Cloud / Render / Locally.

### ✔ ML fundamentals covered  
- Confusion matrix  
- Accuracy  
- Bias & Variance  
- TF-IDF  
- Stemming  
- Stopword removal  

### ✔ Easy to upgrade  
Integrate XGBoost, LSTM, BERT, HuggingFace later.

---

## 🧰 Tech Stack

### **Languages**
- Python 3.8+

### **Libraries / Packages**
- Streamlit  
- NLTK  
- NumPy  
- Pandas  
- Scikit-Learn  
- Matplotlib  

### **NLP Techniques**
- Text cleaning  
- Regular expressions  
- Stemming (PorterStemmer)  
- Stopword removal  
- TF-IDF vectorization  

### **Machine Learning Models**
- Decision Tree (default)  
- Logistic Regression (optional)  
- Random Forest (optional)  
- SVM (optional)  


## 🧠 How the App Works (Step-by-Step)

### **1️⃣ Load Dataset**
- Loads `Restaurant_Reviews.tsv` containing 1000 reviews.

### **2️⃣ NLP Preprocessing**
- Remove special characters  
- Lowercase conversion  
- Tokenization  
- Stopword removal  
- Stemming  
- Combine processed words  

### **3️⃣ Convert Text to Numbers**
Using **TfidfVectorizer**, the text is transformed into numeric feature vectors.

### **4️⃣ Train ML Model**
- Model used: **DecisionTreeClassifier**  
- Data split: **80% training / 20% testing**

### **5️⃣ Model Evaluation**
Outputs:
- Confusion Matrix  
- Accuracy  
- Bias (Train Score)  
- Variance (Test Score)

### **6️⃣ Streamlit UI**
Enter review → click Predict → get result:  
🎉 **Positive** OR ❗ **Negative**

---

## ✨ Features

### 🔹 Real-time Sentiment Prediction  
Classifies any typed restaurant review.

### 🔹 Simple & Clean Streamlit Interface  
Easy to navigate for recruiters & users.

### 🔹 Complete ML Pipeline  
Includes preprocessing → training → evaluation → prediction.

### 🔹 Works Locally & Online  
Compatible with:
- Local machine  
- Streamlit Cloud  
- Render  
- HuggingFace Spaces  

### 🔹 Highly Customizable  
Switch models (SVM, RF, LR, NB) with minimal changes.


### DEVELOPED BY RATNAPRAVA MOHAPATRA



