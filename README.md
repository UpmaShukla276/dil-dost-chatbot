Dil Dost – AI Mental Health Companion

Dil Dost is an AI-powered mental health chatbot built using **Streamlit** and **Machine Learning**.  
It detects emotional states from user text and responds with empathetic, supportive messages — especially designed as a safe space for students.

---

 Project Overview

Dil Dost uses Natural Language Processing (NLP) techniques to classify user input into emotional categories and provide mood-based supportive responses.

It combines:
- TF-IDF (Trigram model)
- Logistic Regression
- Keyword-based mood guard system
- Streamlit UI for interactive experience

---

## ✨ Features

 Mood detection using TF-IDF + Logistic Regression
 🎯 Detects 6 moods:
  - Sad
  - Happy
  - Anxious
  - Angry
  - Lonely
  - Neutral
- Keyword-based override system for improved emotional detection
-  Non-repeating empathetic responses
-  Relaxation tips based on detected mood
-  Custom styled Streamlit interface
-  Model classification report view
-  Built-in fallback dataset (if Kaggle dataset not available)

---

## 📊 Dataset

Based on:

**Sentiment Analysis for Mental Health**  
Author: suchintikasarkar (Kaggle)

- 53,000+ labeled statements
- Categories:
  - Normal
  - Depression
  - Anxiety
  - Stress
  - Bipolar
  - Personality Disorder
  - Suicidal

If the Kaggle dataset CSV is not found locally, the system uses representative built-in samples.

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- TF-IDF Vectorizer
- Logistic Regression
- Regular Expressions (Text Cleaning)

---

How to Run the Project

### 1️⃣ Clone the repository
```
git clone https://github.com/your-username/dil-dost-chatbot.git
cd dil-dost-chatbot
```

### 2️⃣ Install dependencies
```
pip install -r requirements.txt
```

Run the app
```
streamlit run dil_dost_chatbot.py
```

The app will open in your browser.

---

 Model Details

- Vectorizer: TF-IDF (1–3 grams, max_features=10000)
- Classifier: Multinomial Logistic Regression
- Train-Test Split: 80/20 (Stratified)
- Confidence threshold applied for prediction fallback

---

Disclaimer

This chatbot is designed for educational and supportive purposes only.  
It is **not a substitute for professional medical advice, diagnosis, or treatment**.

If you are experiencing severe emotional distress or suicidal thoughts, please contact a mental health professional or your local emergency services immediately.

---

Author

**Upma Shukla**  
Dil Dost – Mental Health Companion for Students


## 💛 Purpose

The goal of Dil Dost is to create a safe, judgment-free space where students can express emotions and receive supportive responses powered by AI.
