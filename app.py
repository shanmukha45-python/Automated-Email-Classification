import streamlit as st
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Automated Email Classification",
    page_icon="📧",
    layout="wide"
)


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR, "models", "spam_classifier_svm.pkl"
)

tfidf_path = os.path.join(
    BASE_DIR, "models", "tfidf_vectorizer.pkl"
)

data_path = os.path.join(
    BASE_DIR, "data", "processed", "spam_processed.csv"
)


# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------

svm_model = joblib.load(model_path)
tfidf = joblib.load(tfidf_path)
df = pd.read_csv(data_path)


# --------------------------------------------------
# SIDEBAR
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
# EMAIL PREDICTION
# ==================================================

if page == "🔍 Email Prediction":

    st.title("📧 Automated Email Classification")

    st.write(
        "Classify an email or message as **Spam** or **Ham** "
        "using NLP, TF-IDF and Linear SVM."
    )

    st.divider()

    st.subheader("📝 Try an Example")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "🚨 Spam Example",
            width="stretch"
        ):
            st.session_state.email_text = (
                "Congratulations! You have won a free prize. "
                "Click now to claim your reward."
            )

    with col2:
        if st.button(
            "✅ Ham Example",
            width="stretch"
        ):
            st.session_state.email_text = (
                "Hi, are we meeting today at 6 PM? "
                "Please let me know."
            )

    if "email_text" not in st.session_state:
        st.session_state.email_text = ""

    email_text = st.text_area(
        "Enter your email or message:",
        value=st.session_state.email_text,
        height=180,
        placeholder="Type or paste your message here..."
    )

    col1, col2 = st.columns(2)

    with col1:
        classify_button = st.button(
            "🔍 Classify Message",
            width="stretch"
        )

    with col2:
        clear_button = st.button(
            "🗑️ Clear",
            width="stretch"
        )

    if clear_button:
        st.session_state.email_text = ""
        st.rerun()

    if classify_button:

        if not email_text.strip():

            st.warning(
                "⚠️ Please enter an email or message before classification."
            )

        else:

            email_tfidf = tfidf.transform([email_text])

            prediction = svm_model.predict(email_tfidf)[0]

            decision_score = svm_model.decision_function(
                email_tfidf
            )[0]

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
                "This is a confidence-like model indicator, "
                "not a calibrated probability."
            )

    st.divider()

    with st.expander("ℹ️ About This Project"):

        st.write(
            """
            **Automated Email Classification** is a Data Science
            and Machine Learning project for classifying messages
            into Spam and Ham categories.

            **Techniques used:**

            - Exploratory Data Analysis (EDA)
            - Natural Language Processing (NLP)
            - Text preprocessing
            - TF-IDF Vectorization
            - Multinomial Naive Bayes
            - Logistic Regression
            - Linear SVM
            - Model evaluation
            - Streamlit deployment
            """
        )


# ==================================================
# DATA SCIENCE DASHBOARD
# ==================================================

else:

    st.title("📊 Data Science Dashboard")

    st.write(
        "Explore the dataset, class distribution and model performance."
    )

    st.divider()

    # --------------------------------------------------
    # DATASET OVERVIEW
    # --------------------------------------------------

    st.subheader("📁 Dataset Overview")

    total_messages = len(df)

    # Convert target column to text so both
    # numeric and text labels are handled.
    target_values = (
        df["target"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    ham_count = int(
        (target_values == "ham").sum()
        + (target_values == "0").sum()
    )

    spam_count = int(
        (target_values == "spam").sum()
        + (target_values == "1").sum()
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
    # MESSAGE DISTRIBUTION
    # --------------------------------------------------

    st.write("### 📈 Message Distribution")

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

    # --------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------

    st.divider()

    st.subheader("🤖 Model Performance")

    model_data = pd.DataFrame({
        "Model": [
            "Multinomial Naive Bayes",
            "Logistic Regression",
            "Linear SVM"
        ],
        "Accuracy": [
            96.52,
            96.13,
            97.78
        ],
        "Spam Precision": [
            98.97,
            100.00,
            95.76
        ],
        "Spam Recall": [
            73.28,
            69.47,
            86.26
        ]
    })

    st.dataframe(
        model_data,
        width="stretch",
        hide_index=True
    )

    # --------------------------------------------------
    # ACCURACY CHART
    # --------------------------------------------------

    st.write("### 📊 Model Accuracy Comparison")

    fig2, ax2 = plt.subplots()

    ax2.bar(
        model_data["Model"],
        model_data["Accuracy"]
    )

    ax2.set_xlabel("Model")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Model Accuracy Comparison")
    ax2.set_ylim(0, 100)

    plt.xticks(rotation=15)

    st.pyplot(fig2)

    # --------------------------------------------------
    # PRECISION VS RECALL
    # --------------------------------------------------

    st.write("### 🎯 Spam Precision vs Recall")

    performance_data = model_data[
        [
            "Model",
            "Spam Precision",
            "Spam Recall"
        ]
    ]

    fig3, ax3 = plt.subplots()

    x = range(len(performance_data))
    width = 0.35

    ax3.bar(
        [i - width / 2 for i in x],
        performance_data["Spam Precision"],
        width,
        label="Spam Precision"
    )

    ax3.bar(
        [i + width / 2 for i in x],
        performance_data["Spam Recall"],
        width,
        label="Spam Recall"
    )

    ax3.set_xlabel("Model")
    ax3.set_ylabel("Score (%)")
    ax3.set_title("Spam Precision vs Recall")

    ax3.set_xticks(list(x))

    ax3.set_xticklabels(
        performance_data["Model"],
        rotation=15
    )

    ax3.set_ylim(0, 100)

    ax3.legend()

    st.pyplot(fig3)

    # --------------------------------------------------
    # LINEAR SVM PERFORMANCE
    # --------------------------------------------------

    st.divider()

    st.subheader("🏆 Linear SVM Performance")

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
    # DATA SCIENCE WORKFLOW
    # --------------------------------------------------

    st.divider()

    st.subheader("🔄 Data Science Workflow")

    st.write(
        """
        **1. Data Collection**  
        SMS Spam Collection dataset

        **2. Data Cleaning**  
        Removed unnecessary columns, handled duplicates
        and prepared labels.

        **3. Exploratory Data Analysis**  
        Studied message categories and dataset distribution.

        **4. Text Preprocessing**  
        Cleaned and transformed text data.

        **5. Feature Engineering**  
        Applied TF-IDF vectorization.

        **6. Model Building**  
        Trained Multinomial Naive Bayes,
        Logistic Regression and Linear SVM.

        **7. Model Evaluation**  
        Compared accuracy, spam precision and spam recall.

        **8. Deployment**  
        Integrated the trained Linear SVM model
        into a Streamlit application.
        """
    )

    # --------------------------------------------------
    # DATASET PREVIEW
    # --------------------------------------------------

    st.divider()

    st.subheader("🔎 Dataset Preview")

    st.dataframe(
        df.head(10),
        width="stretch"
    )

    # --------------------------------------------------
    # PROJECT INFORMATION
    # --------------------------------------------------

    st.divider()

    st.info(
        "This dashboard combines Data Science analysis "
        "with Machine Learning model evaluation and deployment."
    )