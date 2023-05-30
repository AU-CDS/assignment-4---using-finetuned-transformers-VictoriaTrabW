#imports
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

# Now i want to count and visualise the distribution of emotions in each of the datasets

# Counting  the occurrences of each emotion for each dataset
emotion_counts_real = results_real['emotion_huggingface'].value_counts()
emotion_counts_fake = results_fake['emotion_huggingface'].value_counts()
emotion_counts_all = results_all['emotion_huggingface'].value_counts()

# Plotting the emotion distribution for each dataset
sns.set(style="darkgrid")
plt.figure(figsize=(20, 8))

# real headlines dataset
plt.subplot(1, 3, 1)
sns.barplot(x=emotion_counts_real.index, y=emotion_counts_real.values)
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.title("Emotion Distribution for REAL Headlines")
plt.ylim(0, 1700)  # Setting the same maximum count for both subplots to easier compare

# fake headlines dataset
plt.subplot(1, 3, 2)
sns.barplot(x=emotion_counts_fake.index, y=emotion_counts_fake.values)
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.title("Emotion Distribution for FAKE Headlines")
plt.ylim(0, 1700)  # Setting the same maximum count for both subplots to easier compare

#saving only the fake and real visualisation in out folder
plt.savefig('assignment-4---using-finetuned-transformers-VictoriaTrabW/out/emotion_distribution_fake_real.png')

plt.subplot(1, 3, 3)
sns.barplot(x=emotion_counts_all.index, y=emotion_counts_all.values)
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.title("Emotion Distribution for All Headlines")

plt.tight_layout(pad=2.0) 

# saving all the plots side by side in out folder
plt.savefig('assignment-4---using-finetuned-transformers-VictoriaTrabW/out/emotion_distribution_all_datasets.png')