[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/BhnScEmU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10921401&assignment_repo_type=AssignmentRepo)
# Assignment 4 - Using finetuned transformers via HuggingFace

In previous assignments, you've done a lot of model training of various kinds of complexity, such as training document classifiers or RNN language models. This assignment is more like Assignment 1, in that it's about *feature extraction*.

For this assignment, you should use ```HuggingFace``` to extract information from the *Fake or Real News* dataset that we've worked with previously.

You should write code and documentation which addresses the following tasks:

- Initalize a ```HuggingFace``` pipeline for emotion classification
- Perform emotion classification for every *headline* in the data
- Assuming the most likely prediction is the correct label, create tables and visualisations which show the following:
  - Distribution of emotions across all of the data
  - Distribution of emotions across *only* the real news
  - Distribution of emotions across *only* the fake news
- Comparing the results, discuss if there are any key differences between the two sets of headlines


## Tips
- I recommend using ```j-hartmann/emotion-english-distilroberta-base``` like we used in class.
- Spend some time thinking about how best to present you results, and how to make your visualisations appealing and readable.

## Repository structure
-	In folder: contains the dataset fake_or_real_news.csv.
-	out folder: Contains the outputs generated during the execution of the scripts. This includes .csv files with classified emotions and some images.
-	src folder: Contains the main script to run the assignment. This includes the pipeline for the classifier and the visualisation of the results.
-	Setup and reproducibility files:
o	Requirements.txt file: Lists the required programs and packages to run the code. 
o	Can be installed with: pip install -r requirements.txt
-	README.md file: Contains the assignment details, dependencies, additional notes, and reflections on the output. 

## Dependencies and data
The project has been run through the Coder Python app (1.78.2) via UCloud. To run the code effectively, the programs in the requirements.txt are neccesary. 

The dataset is obtained from Kaggle.com at https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news
More info on the huggingface pipeline can be found at https://huggingface.co/docs/transformers/main_classes/pipelines#natural-language-processing

## Reflections and methods
In this assignment, the following methods were employed. First, the dataset was loaded and divided into two categories: real and fake headlines. Next, a pre-trained emotion classification model was utilized to classify the emotions associated with each headline. Finally, the emotion distributions were visualized using bar plots, allowing for a comparison between the emotions evoked by real and fake headlines.

Based on the output, the FAKE and REAL headlines evoked similar emotions, with neutral, fear and anger being the most prevalent. However, there were some differences. The REAL headlines had a slightly greater occurrence of fear, but a lower occurrence of headlines categorized as anger. When observing the FAKE headlines visualisation, these headlines are more often categorized as containing the emotion disgust. The FAKE headlines also have more instances of joy and surprise.
NOTE: the visualisation shows the counts in descending order, so be aware that it is not the colour that represents the emotion but look at the name on the x-axis instead.
