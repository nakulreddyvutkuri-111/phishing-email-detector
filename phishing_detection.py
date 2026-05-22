import re
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("emails.csv")

# Convert labels
df['label'] = df['label'].map({
    'safe': 0,
    'phishing': 1
})

# URL Feature Extractor
class URLFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        features = []

        for text in X:

            urls = re.findall(r'http[s]?://\\S+', text)

            url_count = len(urls)

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

# Text Features
text_features = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english'
    ))
])

# Combine Features
combined_features = FeatureUnion([
    ('text', text_features),
    ('url', URLFeatures())
])

# Final Model
model = Pipeline([
    ('features', combined_features),
    ('classifier', LogisticRegression())
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['label'],
    test_size=0.2,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\\nAccuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\\nConfusion Matrix:")
print(cm)

# Plot Matrix
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Safe', 'Phishing'],
    yticklabels=['Safe', 'Phishing']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

# Test Email
sample_email = '''
URGENT:
Verify your bank account immediately.
Click here: http://fake-login.com
'''

prediction = model.predict([sample_email])[0]

if prediction == 1:
    print("\\nPrediction: PHISHING")
else:
    print("\\nPrediction: SAFE")