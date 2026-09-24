import streamlit as st
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Automated Email Classification",
    page_icon="📧",
    layout="wide"
)


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "spam_classifier_svm.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_vectorizer.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "spam_processed.csv"
)


# --------------------------------------------------
# Load Model, Vectorizer and Dataset
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_vectorizer():
    return joblib.load(VECTORIZER_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


svm_model = load_model()
tfidf_vectorizer = load_vectorizer()
df = load_data()


# --------------------------------------------------
# Prepare Dataset
# --------------------------------------------------

df["target"] = (
    df["target"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["category"] = df["target"].replace({
    "0": "ham",
    "1": "spam"
})

df["message"] = df["message"].fillna("").astype(str)

df["transformed_text"] = (
    df["transformed_text"]
    .fillna("")
    .astype(str)
)

df["message_length"] = df["message"].str.len()


# --------------------------------------------------
# Helper Function - Top Words
# --------------------------------------------------

def get_top_words(series, n=10):
    words = []

    for text in series:
        words.extend(text.split())

    word_counts = Counter(words)

    top_words = word_counts.most_common(n)

    return pd.DataFrame(
        top_words,
        columns=["Word", "Frequency"]
    )


# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------

st.sidebar.title("📧 Email Classifier")

page = st.sidebar.radio(
    "Navigation",
    [
        "🔍 Email Prediction",
        "📊 Data Science Dashboard"
    ]
)


# ==================================================
# EMAIL PREDICTION PAGE
# ==================================================

if page == "🔍 Email Prediction":

    st.title("📧 Automated Email Classification")

    st.write(
        "Classify a message as **Spam** or **Ham** using "
        "TF-IDF and a trained Linear SVM model."
    )

    st.divider()

    # --------------------------------------------------
    # Example Buttons
    # --------------------------------------------------

    st.subheader("📝 Try an Example")

    col1, col2, col3 = st.columns(3)

    if "email_text" not in st.session_state:
        st.session_state.email_text = ""

    with col1:
        if st.button("🚨 Spam Example", width="stretch"):
            st.session_state.email_text = (
                "Congratulations! You have won a free prize. "
                "Click now to claim your reward."
            )

    with col2:
        if st.button("✅ Ham Example", width="stretch"):
            st.session_state.email_text = (
                "Hey, are you coming to college tomorrow? "
                "Let me know."
            )

    with col3:
        if st.button("🗑️ Clear", width="stretch"):
            st.session_state.email_text = ""

    # --------------------------------------------------
    # Email Input
    # --------------------------------------------------

    email_text = st.text_area(
        "Enter your message:",
        value=st.session_state.email_text,
        height=180,
        placeholder="Type or paste a message here..."
    )

    st.session_state.email_text = email_text

    # --------------------------------------------------
    # Classification
    # --------------------------------------------------

    if st.button(
        "🔍 Classify Message",
        type="primary",
        width="stretch"
    ):

        if not email_text.strip():

            st.warning("Please enter a message before classification.")

        else:

            # Transform message using trained TF-IDF vectorizer
            email_tfidf = tfidf_vectorizer.transform([email_text])

            # Predict class
            prediction = svm_model.predict(email_tfidf)[0]

            # Decision score
            decision_score = svm_model.decision_function(
                email_tfidf
            )[0]

            # Confidence-like indicator
            confidence = (
                abs(decision_score)
                / (1 + abs(decision_score))
            ) * 100

            st.divider()

            if prediction == 1:

                st.error("🚨 SPAM MESSAGE")

                st.write(
                    "The model classified this message as **Spam**."
                )

            else:

                st.success("✅ HAM MESSAGE")

                st.write(
                    "The model classified this message as **Ham**."
                )

            st.metric(
                "Model Confidence Indicator",
                f"{confidence:.2f}%"
            )

            st.caption(
                "Note: This is a confidence-like score derived "
                "from the SVM decision function, not a calibrated probability."
            )


    # --------------------------------------------------
    # About Project
    # --------------------------------------------------

    with st.expander("ℹ️ About this Project"):

        st.write("""
        **Automated Email Classification** is a Data Science and
        Machine Learning project for detecting spam messages.

        **Technologies Used:**
        - Python
        - Pandas
        - NumPy
        - NLP
        - TF-IDF
        - Linear SVM
        - Scikit-learn
        - Streamlit

        The model was trained on the SMS Spam Collection dataset.
        """)


# ==================================================
# DATA SCIENCE DASHBOARD
# ==================================================

else:

    st.title("📊 Data Science Dashboard")

    st.write(
        "Exploratory Data Analysis, NLP insights and "
        "Machine Learning model evaluation."
    )

    st.divider()


    # --------------------------------------------------
    # Dataset Overview
    # --------------------------------------------------

    st.subheader("📁 Dataset Overview")

    total_messages = len(df)

    ham_count = (
        df["category"]
        .eq("ham")
        .sum()
    )

    spam_count = (
        df["category"]
        .eq("spam")
        .sum()
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Messages",
            total_messages
        )

    with col2:
        st.metric(
            "Ham Messages",
            ham_count
        )

    with col3:
        st.metric(
            "Spam Messages",
            spam_count
        )


    # --------------------------------------------------
    # Message Distribution
    # --------------------------------------------------

    st.subheader("📊 Message Distribution")

    distribution = pd.DataFrame({
        "Category": ["Ham", "Spam"],
        "Count": [ham_count, spam_count]
    })

    fig, ax = plt.subplots()

    ax.bar(
        distribution["Category"],
        distribution["Count"]
    )

    ax.set_xlabel("Message Category")
    ax.set_ylabel("Number of Messages")
    ax.set_title("Ham vs Spam Message Distribution")

    st.pyplot(fig)

    plt.close(fig)


    # --------------------------------------------------
    # Message Length Analysis
    # --------------------------------------------------

    st.subheader("📏 Message Length Analysis")

    ham_avg_length = df.loc[
        df["category"] == "ham",
        "message_length"
    ].mean()

    spam_avg_length = df.loc[
        df["category"] == "spam",
        "message_length"
    ].mean()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Average Ham Message Length",
            f"{ham_avg_length:.2f} characters"
        )

    with col2:
        st.metric(
            "Average Spam Message Length",
            f"{spam_avg_length:.2f} characters"
        )


    fig, ax = plt.subplots()

    ax.hist(
        df.loc[
            df["category"] == "ham",
            "message_length"
        ],
        bins=30,
        alpha=0.6,
        label="Ham"
    )

    ax.hist(
        df.loc[
            df["category"] == "spam",
            "message_length"
        ],
        bins=30,
        alpha=0.6,
        label="Spam"
    )

    ax.set_xlabel("Message Length")
    ax.set_ylabel("Frequency")
    ax.set_title("Message Length Distribution")

    ax.legend()

    st.pyplot(fig)

    plt.close(fig)


    # --------------------------------------------------
    # Word Frequency Analysis
    # --------------------------------------------------

    st.subheader("🔤 Word Frequency Analysis")

    st.write(
        "Most frequent words found in the preprocessed text "
        "for Ham and Spam messages."
    )

    ham_words = get_top_words(
        df.loc[
            df["category"] == "ham",
            "transformed_text"
        ]
    )

    spam_words = get_top_words(
        df.loc[
            df["category"] == "spam",
            "transformed_text"
        ]
    )

    col1, col2 = st.columns(2)

    # --------------------------------------------------
    # Ham Top Words
    # --------------------------------------------------

    with col1:

        st.markdown("### ✅ Top Words in Ham Messages")

        fig, ax = plt.subplots()

        ax.barh(
            ham_words["Word"][::-1],
            ham_words["Frequency"][::-1]
        )

        ax.set_xlabel("Frequency")
        ax.set_ylabel("Word")
        ax.set_title("Most Frequent Ham Words")

        st.pyplot(fig)

        plt.close(fig)

        st.dataframe(
            ham_words,
            width="stretch",
            hide_index=True
        )


    # --------------------------------------------------
    # Spam Top Words
    # --------------------------------------------------

    with col2:

        st.markdown("### 🚨 Top Words in Spam Messages")

        fig, ax = plt.subplots()

        ax.barh(
            spam_words["Word"][::-1],
            spam_words["Frequency"][::-1]
        )

        ax.set_xlabel("Frequency")
        ax.set_ylabel("Word")
        ax.set_title("Most Frequent Spam Words")

        st.pyplot(fig)

        plt.close(fig)

        st.dataframe(
            spam_words,
            width="stretch",
            hide_index=True
        )


    # --------------------------------------------------
    # Model Performance
    # --------------------------------------------------

    st.subheader("🤖 Model Performance")

    performance = pd.DataFrame({
        "Model": [
            "Multinomial Naive Bayes",
            "Logistic Regression",
            "Linear SVM"
        ],
        "Accuracy (%)": [
            96.52,
            96.13,
            97.78
        ],
        "Spam Precision (%)": [
            98.97,
            100.00,
            95.76
        ],
        "Spam Recall (%)": [
            73.28,
            69.47,
            86.26
        ]
    })

    st.dataframe(
        performance,
        width="stretch",
        hide_index=True
    )


    # --------------------------------------------------
    # Accuracy Comparison
    # --------------------------------------------------

    st.subheader("📈 Accuracy Comparison")

    fig, ax = plt.subplots()

    ax.bar(
        performance["Model"],
        performance["Accuracy (%)"]
    )

    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Model")
    ax.set_title("Model Accuracy Comparison")

    plt.xticks(rotation=15)

    st.pyplot(fig)

    plt.close(fig)


    # --------------------------------------------------
    # Precision vs Recall
    # --------------------------------------------------

    st.subheader("🎯 Spam Precision vs Recall")

    x = range(len(performance))

    width = 0.35

    fig, ax = plt.subplots()

    ax.bar(
        [i - width / 2 for i in x],
        performance["Spam Precision (%)"],
        width,
        label="Precision"
    )

    ax.bar(
        [i + width / 2 for i in x],
        performance["Spam Recall (%)"],
        width,
        label="Recall"
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        performance["Model"],
        rotation=15
    )

    ax.set_ylabel("Percentage")
    ax.set_title("Spam Precision vs Recall")

    ax.legend()

    st.pyplot(fig)

    plt.close(fig)


    # --------------------------------------------------
    # Linear SVM Metrics
    # --------------------------------------------------

    st.subheader("🏆 Linear SVM Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Accuracy",
            "97.78%"
        )

    with col2:
        st.metric(
            "Spam Precision",
            "95.76%"
        )

    with col3:
        st.metric(
            "Spam Recall",
            "86.26%"
        )


    # --------------------------------------------------
    # Data Science Workflow
    # --------------------------------------------------

    st.subheader("🔬 Data Science Workflow")

    st.write("""
    **1. Data Collection**
    
    SMS Spam Collection dataset was used.

    **2. Data Cleaning**
    
    Removed unnecessary columns, handled missing values
    and removed duplicate records.

    **3. Exploratory Data Analysis**
    
    Analyzed message categories, message lengths,
    distributions and word frequencies.

    **4. Natural Language Processing**
    
    Text preprocessing was performed to prepare messages
    for machine learning.

    **5. Feature Engineering**
    
    TF-IDF was used to convert text into numerical features.

    **6. Model Training**
    
    Multinomial Naive Bayes, Logistic Regression and
    Linear SVM were trained and evaluated.

    **7. Model Evaluation**
    
    Accuracy, Precision and Recall were compared.

    **8. Deployment**
    
    The final Linear SVM model was integrated into
    a Streamlit application.
    """)


    # --------------------------------------------------
    # Dataset Preview
    # --------------------------------------------------

    st.subheader("🔎 Dataset Preview")

    st.dataframe(
        df[
            [
                "target",
                "message",
                "transformed_text"
            ]
        ].head(10),
        width="stretch",
        hide_index=True
    )


    # --------------------------------------------------
    # Project Information
    # --------------------------------------------------

    st.subheader("📌 Project Information")

    st.write("""
    **Project:** Automated Email Classification

    **Domain:** Data Science + Machine Learning + NLP

    **Final Model:** Linear Support Vector Machine

    **Feature Extraction:** TF-IDF

    **Deployment:** Streamlit

    **Dataset:** SMS Spam Collection
    """)