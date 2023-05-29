# imports
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# loading the dataset
dataset = pd.read_csv('assignment-4---using-finetuned-transformers-VictoriaTrabW/in/fake_or_real_news.csv')


# initializing a huggingface pipeline
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# iterating over the headlines in the dataset and appending 
emotions = []
for headline in dataset['title']:
    result = emotion_classifier(headline)
    emotion = result[0]['label']
    emotions.append(emotion)