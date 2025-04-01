import streamlit as st
import joblib
import os
from hmm import HMM_Tagger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def preprocess(text):
    return text.lower().strip()

def load_models():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    ml_model = joblib.load(os.path.join(BASE_DIR, "models", "tagging_model.pkl"))
    mlb = joblib.load(os.path.join(BASE_DIR, "models", "tagging_mlb.pkl"))

    hmm_model = HMM_Tagger()
    hmm_model.load_model(os.path.join(BASE_DIR, "models", "hmm_model.pkl"))

    return ml_model, mlb, hmm_model

def predict_ml(model, mlb, title, description, threshold=0.08):
    combined_text = title + " " + description
    probs = model.predict_proba([combined_text])[0]
    prob_dict = dict(zip(mlb.classes_, probs))
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    predicted_labels = [label for label, score in sorted_probs if score >= threshold]
    return predicted_labels, sorted_probs

def predict_hmm(hmm_model, title, description, threshold=0.1):
    combined_text = title + " " + description
    predicted_tags = hmm_model.predict(combined_text)

    input_sentence = preprocess(description)
    predicted_tags = list(set([preprocess(tag) for tag in predicted_tags]))

    all_text = [input_sentence] + predicted_tags
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_text)

    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    filtered_tags = [(tag, score) for tag, score in zip(predicted_tags, cosine_similarities) if score >= threshold]
    sorted_tags = sorted(filtered_tags, key=lambda x: x[1], reverse=True)
    return sorted_tags

# Streamlit UI
st.set_page_config(page_title="StackOverflow Tag Generator", layout="wide")
st.title("🚀 StackOverflow Tag Generator")
st.markdown("""
Welcome to the **StackOverflow AI Tagging System**! 🎯

This application allows you to predict tags for StackOverflow-style questions using powerful machine learning models. Choose your preferred model, provide a title and description, and click **Generate Tags** to get the most relevant suggestions.
""")

# Load models
with st.spinner("Loading models..."):
    ml_model, mlb, hmm_model = load_models()

# Description Input First
with st.form("desc_form"):
    description = st.text_area("📝 Question Description", placeholder="Provide more details about your issue, approach, error, etc.", height=200)
    submitted_desc = st.form_submit_button("Next: Choose Model")

if submitted_desc and description.strip():
    model_choice = st.selectbox("Select Tag Prediction Model:", ["Logistic Regression (ML)", "Hidden Markov Model (HMM)"])

    confirm = st.checkbox("✅ Are you sure about this model choice?")

    if confirm:
        # Proceed to title input and tagging
        with st.form("tag_form"):
            title = st.text_input("📌 Question Title", placeholder="e.g., How to merge dictionaries in Python?")
            submitted = st.form_submit_button("Generate Tags")

        if submitted:
            if not title.strip():
                st.warning("Please enter a question title.")
            else:
                with st.spinner("Generating tags..."):
                    if model_choice == "Logistic Regression (ML)":
                        tags, scores = predict_ml(ml_model, mlb, title, description)
                        st.subheader("🎯 Predicted Tags:")
                        st.write(", ".join(tags) if tags else "No tags above the threshold.")

                        st.subheader("📊 Top Tag Probabilities:")
                        for tag, score in scores[:10]:
                            st.write(f"**{tag}**: {score:.3f}")

                    else:
                        hmm_results = predict_hmm(hmm_model, title, description)
                        st.subheader("🎯 Predicted Tags:")
                        if hmm_results:
                            for tag, score in hmm_results[:10]:
                                st.write(f"**{tag}**: {score:.3f}")
                        else:
                            st.write("No relevant tags found.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Powered by Logistic Regression & HMM")