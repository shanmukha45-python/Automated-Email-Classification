import streamlit as st
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Automated Email Classification",
    page_icon="📧",
    layout="centered"
)


# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "models",
    "spam_classifier_svm.pkl"
)

tfidf_path = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_vectorizer.pkl"
)

svm_model = joblib.load(model_path)
tfidf = joblib.load(tfidf_path)


# ============================================================
# TITLE
# ============================================================

st.title("📧 Automated Email Classification")

st.write(
    "Enter an email message below to determine whether it is "
    "Spam or Ham (Legitimate)."
)

st.divider()


# ============================================================
# EMAIL INPUT
# ============================================================

st.subheader("✉️ Email Message")

st.write("You can enter your own message or try one of the examples below.")

col1, col2 = st.columns(2)

with col1:
    if st.button("🚨 Try Spam Example", use_container_width=True):
        st.session_state["email_text"] = (
            "Congratulations! You have won a free prize. "
            "Click here to claim your reward now!"
        )

with col2:
    if st.button("✅ Try Ham Example", use_container_width=True):
        st.session_state["email_text"] = (
            "Hi, please send me the project report before "
            "tomorrow's meeting."
        )

email_text = st.text_area(
    "Enter your email message:",
    value=st.session_state.get("email_text", ""),
    height=220,
    placeholder="Paste or type an email message here..."
)
if st.button("🗑️ Clear", use_container_width=True):
    st.session_state["email_text"] = ""
    st.rerun()

# ============================================================
# CLASSIFICATION
# ============================================================

if st.button("🔍 Classify Email", use_container_width=True):

    if email_text.strip() == "":
        st.warning("⚠️ Please enter an email message first.")

    else:
        # Convert email into TF-IDF features
        email_tfidf = tfidf.transform([email_text])

        # Make prediction
        prediction = svm_model.predict(email_tfidf)[0]

        # Get SVM decision score
        decision_score = svm_model.decision_function(email_tfidf)[0]

        # Convert decision score into a confidence-like percentage
        confidence = (
            abs(decision_score) /
            (1 + abs(decision_score))
        ) * 100

        st.divider()

        st.subheader("📊 Classification Result")

        if prediction == 1:
            st.error("🚨 SPAM EMAIL")
            st.write(
                "The message has been classified as **Spam** "
                "by the trained Linear SVM model."
            )

        else:
            st.success("✅ HAM — LEGITIMATE EMAIL")
            st.write(
                "The message has been classified as **Ham "
                "(Legitimate)** by the trained Linear SVM model."
            )

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

with st.expander("ℹ️ About this project"):
    st.write(
    """
    **Model Details**

    - Algorithm: Linear Support Vector Machine (LinearSVC)
    - Feature Extraction: TF-IDF
    - Classification: Binary classification
    - Classes: Spam and Ham
    - Dataset Size: 5,169 messages
    """
)
    
