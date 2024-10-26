import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

from preprocess_data import load_data

# Load data
train_df, test_df = load_data()

# Split train_df into training and validation sets
def split_data(df):
    return train_test_split(df, test_size=0.2, random_state=42)

train_data, val_data = split_data(train_df)

# Feature Extraction using Sentiment Analysis
class SentimentFeatureExtractor:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def extract_features(self, df):
        features = []
        i, N = 0, len(df['Summary'])
        for summary, text in zip(df['Summary'], df['Text']):
            combined_text = (summary if summary else "") + " " + (text if text else "")
            sentiment = self.analyzer.polarity_scores(combined_text)
            features.append(sentiment['compound'])  # Extract the 'compound' score for simplicity
            i += 1
            if i % 1_000_000:
                print(f'{100*i/N:.1f}% complete')
            if i >= N:
                break
        return np.array(features).reshape(-1, 1)  # Return as a column vector

# Model: Simple classifier to map sentiment score to one of [1, 2, 3, 4, 5]
class SentimentToScoreModel:
    def __init__(self):
        self.boundaries = [-1.0, -0.5, 0.0, 0.5, 1.0]

    def predict(self, features):
        return np.digitize(features, bins=self.boundaries)

# Pipeline Orchestration
def pipeline(train_data, val_data):
    print('Extracting features. . .')
    extractor = SentimentFeatureExtractor()
    X_train = extractor.extract_features(train_data)
    X_val = extractor.extract_features(val_data)

    print('Getting labels. . .')
    y_train = train_data['Score'].values
    y_val = val_data['Score'].values

    print('Making predictions. . .')
    model = SentimentToScoreModel()
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    print('Evaluating. . .')
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
    print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Run pipeline
pipeline(train_data, val_data)
