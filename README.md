# Automated Email Classification

An end-to-end Data Science + Machine Learning + NLP project that automatically classifies messages as Spam or Ham using TF-IDF feature extraction and a Linear Support Vector Machine (SVM).

## Live Demo

Try the deployed application:

https://automated-email-classification-c5w2zyk6r3uuxhpfrdhpkl.streamlit.app/

### Application Preview

![Spam Prediction](screenshots/spam_prediction.png)

## Project Overview

Spam messages are a common problem in email and messaging systems. This project applies Data Science and Natural Language Processing techniques to automatically identify whether a message is Spam or Ham.

The project follows a complete machine learning workflow:

Data Collection -> Data Cleaning -> EDA -> NLP Preprocessing -> Feature Engineering -> Model Training -> Model Evaluation -> Deployment

## Objectives

- Clean and preprocess text data
- Perform Exploratory Data Analysis (EDA)
- Analyze message length and word frequencies
- Apply Natural Language Processing techniques
- Convert text into numerical features using TF-IDF
- Train multiple machine learning models
- Compare model performance
- Select a suitable classification model
- Deploy the final model using Streamlit

## Dataset

Dataset: SMS Spam Collection

The processed dataset contains:

- Total messages: 5,169
- Ham messages: 4,516
- Spam messages: 653

### Dataset Features

| Column | Description |
|---|---|
| target | Original binary label |
| message | Original message text |
| transformed_text | Preprocessed text |

## Exploratory Data Analysis

The project includes:

- Ham vs Spam distribution
- Message length analysis
- Average message length comparison
- Message length distribution
- Word frequency analysis
- Dataset preview

## NLP Preprocessing

Text preprocessing was performed to prepare messages for machine learning.

The processed text was converted into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).

## Machine Learning Models

Three classification algorithms were evaluated:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear SVM

### Model Performance

| Model | Accuracy | Spam Precision | Spam Recall |
|---|---:|---:|---:|
| Multinomial Naive Bayes | 96.52% | 98.97% | 73.28% |
| Logistic Regression | 96.13% | 100.00% | 69.47% |
| Linear SVM | 97.78% | 95.76% | 86.26% |

The Linear SVM model was selected for deployment based on the evaluation performed in this project.

## Final Model

Linear Support Vector Machine (SVM)

Performance:

- Accuracy: 97.78%
- Spam Precision: 95.76%
- Spam Recall: 86.26%

## Streamlit Application

The project includes an interactive Streamlit application with two main sections.

### Email Prediction

Users can enter a message and classify it as:

- Spam
- Ham

The application also displays a model confidence indicator based on the SVM decision function.

Note: The confidence indicator is not a calibrated probability.

### Data Science Dashboard

The dashboard provides:

- Dataset statistics
- Ham/Spam distribution
- Message length analysis
- Word frequency analysis
- Model comparison
- Accuracy comparison
- Precision vs Recall comparison
- Final Linear SVM metrics
- Dataset preview
- Data Science workflow

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Natural Language Processing
- TF-IDF
- Linear SVM
- Joblib
- Streamlit
- Git
- GitHub

## Project Structure

```text
Automated_Email_Classification/
|
|-- app.py
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- data/
|   |-- raw/
|   |   `-- spam.csv
|   |
|   `-- processed/
|       `-- spam_processed.csv
|
|-- models/
|   |-- spam_classifier_svm.pkl
|   `-- tfidf_vectorizer.pkl
|
`-- notebooks/
    |-- 01_EDA.ipynb
    |-- 02_Preprocessing.ipynb
    `-- 03_Model_Building.ipynb
