import streamlit as st
import tempfile
import os
import pandas as pd
import pdfplumber
import docx
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Text Extraction Helpers
# -----------------------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text(file, filename):
    if filename.endswith(".pdf"):
        return clean_text(extract_text_from_pdf(file))
    elif filename.endswith(".docx"):
        return clean_text(extract_text_from_docx(file))
    elif filename.endswith(".txt"):
        return clean_text(file.read().decode('utf-8'))
    else:
        return ""

# -----------------------------
# AI Matching Model
# -----------------------------
def rank_candidates(resume_texts, job_description):
    all_texts = list(resume_texts.values()) + [job_description]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    job_vec = tfidf_matrix[-1]
    resume_vecs = tfidf_matrix[:-1]

    similarities = cosine_similarity(resume_vecs, job_vec)

    results = []
    for i, (filename, text) in enumerate(resume_texts.items()):
        matched_keywords = [word for word in job_description.split() if word in text]
        results.append({
            'Candidate': filename,
            'Match Score': round(similarities[i][0] * 100, 2),
            'Matched Keywords': ", ".join(set(matched_keywords))
        })

    return sorted(results, key=lambda x: x['Match Score'], reverse=True)

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("🤖 AI-Powered Resume Screener")

with st.sidebar:
    st.header("Upload Files")
    job_file = st.file_uploader("Upload Job Description", type=["txt", "pdf", "docx"])
    resume_files = st.file_uploader("Upload Resumes", type=["txt", "pdf", "docx"], accept_multiple_files=True)

if job_file and resume_files:
    job_text = extract_text(job_file, job_file.name)
    resume_texts = {res.name: extract_text(res, res.name) for res in resume_files}

    with st.spinner("Ranking candidates..."):
        ranked = rank_candidates(resume_texts, job_text)
        df = pd.DataFrame(ranked)
        st.success("Done! Here are the results:")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", csv, "ranked_candidates.csv", "text/csv")

else:
    st.info("Please upload a job description and at least one resume to get started.")
