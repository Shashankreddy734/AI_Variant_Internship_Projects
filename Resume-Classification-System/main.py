import streamlit as st
import joblib
import re
import PyPDF2
import docx
import mammoth  # Added for handling .doc files
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd
import io

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load trained models and encoders
xgb_model = joblib.load("xgb_resume_classifier.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Cleans and preprocesses text."""
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    text = re.sub(r'[•ǁ:/\(\),#,-]', '', text)
    text = re.sub(r'\u2018|\u2019', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def extract_text_from_file(uploaded_file):
    """Extracts text from uploaded files based on file format."""
    file_extension = uploaded_file.name.split(".")[-1]
    text = ""
    try:
        if file_extension == "pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text() + " "
        elif file_extension == "docx":
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + " "
        elif file_extension == "doc":
            result = mammoth.extract_raw_text(uploaded_file)
            text = result.value
        elif file_extension == "txt":
            text = uploaded_file.getvalue().decode("utf-8")
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {e}")
    return text.strip()

def classify_resume(resume_text):
    """Classifies resume text into one of the predefined categories."""
    cleaned_text = clean_text(resume_text)
    vectorized_text = tfidf_vectorizer.transform([cleaned_text])
    prediction = xgb_model.predict(vectorized_text)[0]
    category = label_encoder.inverse_transform([prediction])[0]
    return category

# Streamlit App UI
st.title("\U0001F4DA AI-Powered Resume Classifier")
st.markdown("""
**Upload multiple resumes and classify them into categories:**
- React Developer
- Workday
- Peoplesoft
- SQL Developer
""")

# File Upload
uploaded_files = st.file_uploader("Upload Resumes (PDF, DOCX, DOC, TXT)", type=["pdf", "docx", "doc", "txt"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Classify Resumes"):
        results = []
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)
        
        for i, uploaded_file in enumerate(uploaded_files):
            resume_text = extract_text_from_file(uploaded_file)
            if resume_text:
                category = classify_resume(resume_text)
                results.append({"Filename": uploaded_file.name, "Predicted Category": category})
            progress_bar.progress((i + 1) / total_files)
        
        progress_bar.empty()
        
        # Display results in a styled DataFrame
        df_results = pd.DataFrame(results)
        st.write("### Classification Results")
        st.dataframe(df_results.style.set_properties(**{'text-align': 'center'}))

        # Option to download results
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", data=csv, file_name="resume_classification_results.csv", mime="text/csv")