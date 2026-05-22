import streamlit as st
import pandas as pd
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("emails.csv")

# Convert labels
df['label'] = df['label'].map({
    'safe': 0,
    'phishing': 1
})

# Custom feature extractor
class URLFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        features = []

        for text in X:

            # Find URLs
            urls = re.findall(r'http[s]?://\\S+', text)

            url_count = len(urls)

            # Suspicious words
            suspicious_words = [
                'verify',
                'bank',
                'password',
                'click',
                'urgent',
                'free',
                'winner',
                'account'
            ]

            suspicious_count = sum(
                word in text.lower()
                for word in suspicious_words
            )

            features.append([
                url_count,
                suspicious_count
            ])

        return np.array(features)

# Text processing
text_features = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english'
    ))
])

# Combine all features
combined_features = FeatureUnion([
    ('text', text_features),
    ('url', URLFeatures())
])

# Final ML model
model = Pipeline([
    ('features', combined_features),
    ('classifier', LogisticRegression())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['label'],
    test_size=0.2,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# ---------------- STREAMLIT UI ----------------

st.title("Phishing Email Detector")

st.write("Enter email content below:")

email_input = st.text_area("Email Text")

if st.button("Check Email"):

    prediction = model.predict([email_input])[0]

    if prediction == 1:
        st.error("⚠️ This Email is PHISHING")
    else:
        st.success("✅ This Email is SAFE")