import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# loading the dataset
dataset_all = pd.read_csv('assignment-4---using-finetuned-transformers-VictoriaTrabW/in/fake_or_real_news.csv')
# Filtering the dataset for headlines with label "REAL"
dataset_real = dataset_all[dataset_all['label'] == 'REAL']
# Filter the dataset for headlines with label "FAKE"
dataset_fake = dataset_all[dataset_all['label'] == 'FAKE']

# initializing a huggingface pipeline
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Function to classify emotions for a given dataset
def classify_emotions(data):
    emotions = []
    for headline in data['title']:
        result = emotion_classifier(headline)
        emotion = result[0]['label']
        emotions.append(emotion)
    return emotions

# Classify emotions for headlines with label "REAL"
emotions_real = classify_emotions(dataset_real)

# Classify emotions for headlines with label "FAKE"
emotions_fake = classify_emotions(dataset_fake)

# classifying emotions for all headlines
emotions_all = classify_emotions(dataset_all)


# Create separate DataFrames for the results
results_real = pd.DataFrame({'headline': dataset_real['title'], 'emotion_huggingface': emotions_real})
results_fake = pd.DataFrame({'headline': dataset_fake['title'], 'emotion_huggingface': emotions_fake})
results_all = pd.DataFrame({'headline': dataset_all['title'], 'emotion_huggingface': emotions_all})

# Saving the results to separate CSV files in the "out" folder
results_real.to_csv('assignment-4---using-finetuned-transformers-VictoriaTrabW/out/results_real.csv', index=False)
results_fake.to_csv('assignment-4---using-finetuned-transformers-VictoriaTrabW/out/results_fake.csv', index=False)
results_all.to_csv('assignment-4---using-finetuned-transformers-VictoriaTrabW/out/results_all.csv', index=False)
