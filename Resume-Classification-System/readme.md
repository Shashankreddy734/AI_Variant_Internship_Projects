AI-Powered Resume Classification

Overview

This project is an AI-driven resume classification system that uses natural language processing (NLP) and machine learning to categorize resumes into predefined job roles. The system leverages a Streamlit-based web application to allow users to upload multiple resumes and classify them in real time.

Features

Supports multiple file formats: PDF, DOCX, DOC, TXT

Cleans and preprocesses resume text using NLTK

Extracts relevant features using TF-IDF vectorization

Classifies resumes into job categories using an XGBoost model

Provides real-time classification results

Allows users to download classification results as a CSV file

Tech Stack

Python

Streamlit (for the web interface)

XGBoost (for resume classification)

NLTK (for text processing)

Joblib (for model persistence)

Pandas (for handling data)

PyPDF2, docx, mammoth (for file processing)

Installation

Prerequisites

Ensure you have Python installed (version 3.7+ recommended). Install the required dependencies using:

pip install -r requirements.txt

Running the Application

Clone the repository:

git clone https://github.com/your-repo-name/resume-classifier.git
cd resume-classifier

Install dependencies (if not already installed):

pip install -r requirements.txt

Run the Streamlit application:

streamlit run main.py

Upload resumes and classify them into predefined job roles.

How It Works

Text Extraction: Reads resume content from uploaded files.

Preprocessing: Cleans and tokenizes text using NLTK.

Feature Extraction: Converts text data into numerical features using TF-IDF vectorization.

Classification: Uses a pre-trained XGBoost model to classify resumes.

Results Display: Outputs the predicted job category and allows results download as CSV.

Model and Dataset

The classification model is trained on a labeled dataset of resumes and saved using joblib. The dataset is vectorized using TF-IDF, and the model is trained using XGBoost.
