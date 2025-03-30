import re
import PyPDF2
import docx
import joblib
import mammoth  # For handling .doc files
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

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


# Define functions
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


# Streamlit App UI with Enhanced Styling
st.set_page_config(page_title="Resume Classifier", page_icon="📄", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .title { 
            font-size: 40px; 
            color: #2E86C1; 
            text-align: center; 
            font-weight: bold; 
            margin-bottom: 10px; 
        }
        .subtitle { 
            font-size: 18px; 
            color: #566573; 
            text-align: center; 
            margin-bottom: 30px; 
        }
        .main-container { 
            background-color: #F4F6F6; 
            padding: 20px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0,0,0,0.1); 
        }
        .stButton>button { 
            background-color: #2E86C1; 
            color: white; 
            border-radius: 8px; 
            padding: 10px 20px; 
            font-size: 16px; 
        }
        .stButton>button:hover { 
            background-color: #1B4F72; 
        }
        .footer { 
            text-align: center; 
            margin-top: 40px; 
            font-size: 14px; 
            color: #7F8C8D; 
        }
        .category-list { 
            font-size: 16px; 
            color: #17202A; 
            line-height: 1.6; 
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="title">📄 AI-Powered Resume Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Classify resumes into job categories with ease</div>', unsafe_allow_html=True)

# Main Container
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # Description
    st.markdown("""
        <div class='category-list'>
        <strong>Supported Categories:</strong><br>
        - React Developer<br>
        - Workday<br>
        - Peoplesoft<br>
        - SQL Developer
        </div>
    """, unsafe_allow_html=True)

    # File Upload Section
    st.markdown("### Upload Resumes")
    uploaded_files = st.file_uploader(
        "Choose files (PDF, DOCX, DOC, TXT)",
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=True,
        help="Upload one or more resumes to classify"
    )

    if uploaded_files:
        if st.button("Classify Resumes 🚀"):
            results = []
            progress_bar = st.progress(0)
            total_files = len(uploaded_files)

            # Process each file
            for i, uploaded_file in enumerate(uploaded_files):
                resume_text = extract_text_from_file(uploaded_file)
                if resume_text:
                    category = classify_resume(resume_text)
                    results.append({"Filename": uploaded_file.name, "Predicted Category": category})
                progress_bar.progress((i + 1) / total_files)

            progress_bar.empty()

            # Display Results
            if results:
                st.markdown("### Classification Results 🎉")
                df_results = pd.DataFrame(results)
                styled_df = df_results.style.set_properties(**{
                    'text-align': 'center',
                    'background-color': '#ECF0F1',
                    'border-radius': '5px',
                    'padding': '8px'
                }).set_table_styles([
                    {'selector': 'th',
                     'props': [('background-color', '#2E86C1'), ('color', 'white'), ('text-align', 'center')]}
                ])
                st.dataframe(styled_df)

                # Download Button
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV 📥",
                    data=csv,
                    file_name="resume_classification_results.csv",
                    mime="text/csv",
                    help="Save the classification results as a CSV file"
                )
            else:
                st.warning("No valid resumes were processed.")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer with Your Name
st.markdown('<div class="footer">Developed by <strong>I SHASHANK REDDY</strong></div>', unsafe_allow_html=True)

