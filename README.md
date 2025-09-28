# 📰 Fake News Classifier MVP

A **Machine Learning-based interactive web app** to classify news headlines as **Real** or **Fake** with confidence scores. Built using **Python, Scikit-learn, NLTK, and Streamlit**.

---

## **Table of Contents**
- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Enhancements](#enhancements)
- [License](#license)

---

## **Overview**
This project provides a simple, interactive tool to detect fake news. It uses **Logistic Regression** and **Naive Bayes** models trained on a labeled news dataset. Users can input any headline to see predictions along with confidence scores. Color-coded results make it easy to interpret.

---

## **Features**
- Real-time prediction for user-input headlines  
- Predictions from **Logistic Regression** & **Naive Bayes**  
- Confidence scores displayed (%)  
- Color-coded labels: **Green = Real, Red = Fake**  
- TF-IDF preprocessing: lowercase, remove punctuation & stopwords  
- Easy-to-run **Streamlit web interface**  

---

## **Demo**
![Demo Screenshot](/Screenshot.png)  
*Working Demo*

---

## **Installation**

1. **Clone the repository**
```bash
git clone https://github.com/sthavirpunwatkar/fake_news_classifier.git
cd fake_news_classifier
```

2. **Create a virtual environment**
```bash
python -m venv fake_news_env
```

3. **Activate the virtual environment**
- Windows:
```bash
fake_news_env\Scripts\activate
```
- Mac/Linux:
```bash
source fake_news_env/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## **Usage**

1. **Run the Streamlit app**
```bash
streamlit run app.py
```

2. **Enter a news headline** in the text box

3. **Click "Predict"** to see:
- Prediction from Logistic Regression & Naive Bayes  
- Confidence scores (%)  
- Color-coded results  

---

## **Project Structure**
```
fake_news_classifier/
│
├── data/
│   ├── True.csv
│   ├── Fake.csv
│   └── combined_news.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_inference_demo.ipynb
│
├── models/
│   ├── fake_news_log_model.pkl
│   └── fake_news_nb_model.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

---

## **Technologies**
- Python 3.x  
- Pandas, Numpy  
- Scikit-learn (Logistic Regression, Naive Bayes)  
- NLTK (Stopwords)  
- Streamlit (Interactive Web App)  

---

## **Enhancements**
- Batch prediction from CSV  
- Model selector (choose Logistic Regression / Naive Bayes)  
- Confidence-based UI enhancements  
- Deployment on **Streamlit Cloud**  

---

## **License**
This project is licensed under the MIT License.
