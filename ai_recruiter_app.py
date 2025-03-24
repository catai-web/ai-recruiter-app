import streamlit as st
st.set_page_config(page_title="AI Resume Screener", layout="wide")

import tempfile
import os
import pandas as pd
import pdfplumber
import docx
import re
import dropbox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Dropbox Setup
# -----------------------------
DROPBOX_TOKEN = st.text_input("Enter your Dropbox token", type="password")
dbx = dropbox.Dropbox(DROPBOX_TOKEN) if DROPBOX_TOKEN else None

# Connection Test
if DROPBOX_TOKEN:
    dbx = dropbox.Dropbox(DROPBOX_TOKEN)
    try:
        current_account = dbx.users_get_current_account()
        st.sidebar.success(f"✅ Connected to Dropbox as {current_account.name.display_name}")
    except Exception as e:
        st.sidebar.error("❌ Failed to connect to Dropbox. Check your token.")
        dbx = None
else:
    st.sidebar.error("❌ No Dropbox token entered. Please paste your token above.")

def list_dropbox_folders(path=""):
    try:
        entries = dbx.files_list_folder(path).entries
        folders = [entry.path_display for entry in entries if isinstance(entry, dropbox.files.FolderMetadata)]
        return folders
    except Exception as e:
        st.sidebar.error(f"Error accessing Dropbox folders: {e}")
        return []

def list_dropbox_files(folder):
    try:
        entries = dbx.files_list_folder(folder).entries
        files = [entry for entry in entries if isinstance(entry, dropbox.files.FileMetadata) and entry.name.lower().endswith((".pdf", ".docx", ".txt"))]
        return files
    except Exception as e:
        st.sidebar.error(f"Error accessing Dropbox files: {e}")
        return []

def list_root_files_if_no_folders():
    try:
        entries = dbx.files_list_folder("").entries
        files = [entry for entry in entries if isinstance(entry, dropbox.files.FileMetadata) and entry.name.lower().endswith((".pdf", ".docx", ".txt"))]
        return files
    except Exception as e:
        st.sidebar.error(f"Error accessing root files: {e}")
        return []

def download_and_extract_text(file_metadata):
    _, res = dbx.files_download(file_metadata.path_lower)
    name = file_metadata.name
    content = res.content
    if name.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        text = extract_text_from_pdf(tmp_path)
        os.unlink(tmp_path)
        return clean_text(text)
    elif name.endswith(".docx"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        text = extract_text_from_docx(tmp_path)
        os.unlink(tmp_path)
        return clean_text(text)
    elif name.endswith(".txt"):
        return clean_text(content.decode("utf-8"))
    return ""

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
st.title("🤖 AI-Powered Resume Screener")

with st.sidebar:
    st.header("Upload Job Description")
    job_file = st.file_uploader("Upload Job Description", type=["txt", "pdf", "docx"])

    st.header("Resume Source")
    use_local = st.checkbox("Use Local Uploads", value=True)
    use_dropbox = st.checkbox("Use Dropbox", value=True)

resume_texts = {}

if use_local:
    resume_files = st.sidebar.file_uploader("Upload Resumes", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    if resume_files:
        resume_texts.update({res.name: extract_text(res, res.name) for res in resume_files})

if use_dropbox and dbx:
    st.sidebar.subheader("Dropbox Resumes")

    try:
        all_entries = dbx.files_list_folder("").entries
        st.sidebar.markdown("**📦 Dropbox Root Contents:**")
        for entry in all_entries:
            st.sidebar.write(f"- {entry.name} ({type(entry).__name__})")
    except Exception as e:
        st.sidebar.error(f"Dropbox access error: {e}")

    folder_options = list_dropbox_folders("")

    if folder_options:
        selected_folder = st.sidebar.selectbox("Select Dropbox Folder", folder_options)
        if selected_folder:
            file_metadata_list = list_dropbox_files(selected_folder)
            if file_metadata_list:
                selected_files = st.sidebar.multiselect("Select Resumes", options=file_metadata_list, format_func=lambda f: f.name)
                for file_meta in selected_files:
                    resume_texts[file_meta.name] = download_and_extract_text(file_meta)
            else:
                st.sidebar.warning("No resumes found in selected Dropbox folder.")
    else:
        st.sidebar.info("No folders found. Checking root directory...")
        root_files = list_root_files_if_no_folders()
        if root_files:
            selected_files = st.sidebar.multiselect("Select Resumes from Root", options=root_files, format_func=lambda f: f.name)
            for file_meta in selected_files:
                resume_texts[file_meta.name] = download_and_extract_text(file_meta)
        else:
            st.sidebar.warning("No resumes found in root directory either.")

if job_file and resume_texts:
    job_text = extract_text(job_file, job_file.name)

    with st.spinner("Ranking candidates..."):
        ranked = rank_candidates(resume_texts, job_text)
        df = pd.DataFrame(ranked)
        st.success("Done! Here are the results:")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", csv, "ranked_candidates.csv", "text/csv")

elif not job_file:
    st.info("Please upload a job description to continue.")
elif not resume_texts:
    st.info("Please upload or select resumes to continue.")
