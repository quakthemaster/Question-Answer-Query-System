# IIITH-AIML Capstone Project by Group 3

This repo is for the code files used by Group-3 of Cohort-20 of PGCP program at IIITH organized by TalentSprint.

Two tasks are carried out under the project. Task 1 is generating subject line for given Email body, while Task 2 is generating answers for queries related to AIML domain.

## Email Subject Line Generation

The dataset used for this task is Annotated Enron Subject Line Corpus (AESLC) available at https://github.com/ryanzhumich/AESLC/tree/7afc087a1cc234d07121f0d4a2d87102ddceabc2/enron_subject_line

### Data Preprocessing and EDA
The preprocessing and exploratory data analysis is carried out using  available AESLC Data preprocessing and EDA.ipynb
at https://github.com/quakthemaster/Question-Answer-Query-System/blob/5bab7f5a4464c49ccb9972dde08ef7afde43b2bc/Dataset%20for%20Email%20Subject%20Line%20Generation/AESLC%20Data%20preprocessing%20and%20EDA.ipynb

After preprocessing, the dataset is added to CSV files as follows-

Training data- https://github.com/quakthemaster/Question-Answer-Query-System/blob/5bab7f5a4464c49ccb9972dde08ef7afde43b2bc/Dataset%20for%20Email%20Subject%20Line%20Generation/Task1Train.csv

Validation data- https://github.com/quakthemaster/Question-Answer-Query-System/blob/5bab7f5a4464c49ccb9972dde08ef7afde43b2bc/Dataset%20for%20Email%20Subject%20Line%20Generation/Task1Val.csv

Testing data- https://github.com/quakthemaster/Question-Answer-Query-System/blob/5bab7f5a4464c49ccb9972dde08ef7afde43b2bc/Dataset%20for%20Email%20Subject%20Line%20Generation/Task1Test.csv

### Model Training

distil-GPT2 and GPT2-medium were fine tuned on this dataset.

emailsubgen-capstone-group3_v46.ipynb contains model training code with DISTILGPT-2 - https://github.com/quakthemaster/Question-Answer-Query-System/blob/5bab7f5a4464c49ccb9972dde08ef7afde43b2bc/Model%20Training/emailsubgen-capstone-group3_v46.ipynb

emailsubgen-capstone-group3_v51.ipynb contains model training code with GPT2-medium - https://github.com/quakthemaster/Question-Answer-Query-System/blob/5bab7f5a4464c49ccb9972dde08ef7afde43b2bc/Model%20Training/emailsubgen-capstone-group3_v51.ipynb

### Deployment

Deployment related files are available at https://github.com/quakthemaster/Question-Answer-Query-System/tree/5bab7f5a4464c49ccb9972dde08ef7afde43b2bc/Deployment%20Local/templates

## Question Answering System

### Data 
Custom data containing queries related to Artificial Intelligence and Machine Learning was created and used for training data.

### Model Training
qadataset-capstone-group3_v8.ipynb - GPT2 model fine tune - https://github.com/quakthemaster/Question-Answer-Query-System/blob/5bab7f5a4464c49ccb9972dde08ef7afde43b2bc/Model%20Training/qadataset-capstone-group3_v8.ipynb

### Deployment

Deployment on cloud was done using Hugging Face Spaces and Gradio

https://github.com/quakthemaster/Question-Answer-Query-System/tree/5bab7f5a4464c49ccb9972dde08ef7afde43b2bc/Deployment%20on%20Hugging%20Faces

## Deployed model can be accessed at 

https://huggingface.co/spaces/Augustya/ai-subject-answer-generator
