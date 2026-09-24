# Automated Email Classification

## 📧 Project Overview

Automated Email Classification is a machine learning application that classifies text messages into two categories:

- Spam
- Ham (Legitimate)

The project uses Natural Language Processing (NLP), TF-IDF feature extraction, and a Linear Support Vector Machine (LinearSVC) classifier.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Linear SVM
- Joblib
- Streamlit
- Matplotlib
- Seaborn

## 📊 Dataset

The project uses the SMS Spam Collection dataset.

After preprocessing:

- Total messages: 5,169
- Ham messages: 4,516
- Spam messages: 653

## 🤖 Machine Learning Model

Multiple classification models were evaluated during model development.

The Linear SVM model was selected for the final application.

The model achieved approximately:

- Accuracy: 97.78%
- Spam Precision: 95.76%
- Spam Recall: 86.26%

## 🔄 Project Workflow

Dataset
↓
Data Cleaning
↓
Text Preprocessing
↓
TF-IDF Vectorization
↓
Model Training
↓
Model Evaluation
↓
Linear SVM
↓
Model Saving
↓
Streamlit Application
↓
Spam / Ham Prediction

## 🚀 How to Run

### 1. Clone or download the project

Open the project folder in VS Code.

### 2. Install dependencies

```bash
pip install -r requirements.txt