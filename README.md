# Phishing Email Detection using Machine Learning

## Overview

Phishing Email Detection is a Machine Learning-based cybersecurity project developed using Python and Scikit-learn. The system is designed to classify emails as **Phishing** or **Safe** by analyzing textual content, suspicious keywords, and URL-based patterns commonly found in phishing attacks.

This project demonstrates the practical application of Natural Language Processing (NLP) and Machine Learning techniques for improving email security and detecting malicious emails efficiently.

---

## Features

* Machine Learning-based phishing detection
* TF-IDF text feature extraction
* URL and suspicious keyword analysis
* Real-time email classification
* Accuracy score and confusion matrix visualization
* Interactive Streamlit web application

---

## Technologies Used

* Python
* Scikit-learn
* Pandas
* NumPy
* Streamlit
* Matplotlib
* Seaborn

---

## Project Workflow

1. Load phishing and legitimate email dataset
2. Extract textual and URL-based features
3. Convert text into numerical vectors using TF-IDF
4. Train Logistic Regression classifier
5. Predict whether emails are Phishing or Safe
6. Display model accuracy and confusion matrix
7. Provide real-time predictions using Streamlit UI

---

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

---

## Run the Application

```bash
streamlit run app.py
```

---

## Sample Prediction

* ⚠️ This Email is PHISHING
* ✅ This Email is SAFE

---

## Future Improvements

* Advanced NLP models (BERT, LSTM)
* Large-scale datasets
* Email attachment analysis
* Gmail API integration
* Cloud deployment

---

## Conclusion

This project highlights how Machine Learning and NLP can be effectively used in cybersecurity applications to detect phishing emails and improve digital communication security.
