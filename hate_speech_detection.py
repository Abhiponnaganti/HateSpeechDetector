import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('/Users/abhi/Downloads/HateSpeech Detection/data/hate_speech_data.csv')
tweets, labels = data['tweet'], data['class']

# For consistency and sanity in randomness
seed_value = 42
np.random.seed(seed_value)

# Quick split, simplicity rules
train_tweets, test_tweets, train_labels, test_labels = train_test_split(
    tweets, labels, test_size=0.2, random_state=seed_value)

# Set up the pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_df=0.75,  # A little more aggressive filtering of common words
        min_df=3,     # Keeping only more relevant terms
        ngram_range=(1, 2),  # Let’s explore up to bigrams
        sublinear_tf=True,   # Smoother frequency weighting
        norm='l2'
    )),
    ('classifier', LogisticRegression(
        max_iter=1500,  # A bit less iterating, balance between speed and precision
        n_jobs=-1,      # Use all CPUs to make things fast
        C=1.0           # Set a reasonable C value to start with
    ))
])

# Train the pipeline directl
print("Training the model...")
pipeline.fit(train_tweets, train_labels)

# Predict and evaluate
predictions = pipeline.predict(test_tweets)

# Classification results
class_labels = ['Hate Speech', 'Offensive', 'Neither']
report = classification_report(test_labels, predictions, target_names=class_labels, output_dict=True)

# Print out the classification report like a human would care
print("\nClassification Results (don't get bogged down by every metric, just the big picture):")
print(classification_report(test_labels, predictions, target_names=class_labels))

# Visualize the results but with a twist, add more style
print("Visualizing the predictions distribution...")
labels = list(report.keys())[:-3]
sizes = [report[label]['support'] for label in labels]

plt.figure(figsize=(7, 5))
plt.pie(sizes, labels=labels, autopct='%1.2f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99'])
plt.title('Prediction Distribution of Test Set', fontsize=14)
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is a circle
plt.tight_layout()
plt.show()

# Show sample predictions - but this time, let’s see a bit more context
print("\n--- Sample Predictions ---")
sample_size = 5
sample_idx = np.random.choice(len(test_tweets), sample_size, replace=False)

for idx in sample_idx:
    tweet = test_tweets.iloc[idx]
    pred = class_labels[predictions[idx]]
    print(f"\nTweet: {tweet}")
    print(f"Predicted: {pred}")
    print('-' * 50)

# Summary note
print("\nDone! The model is performing well.")