import streamlit as st
st.set_page_config(page_title="AI Resume Screener", layout="wide")

import tempfile
import os
import pandas as pd
import pdfplumber
import docx
import re
import dropbox
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Dropbox Setup
# -----------------------------
if "DROPBOX_TOKEN" not in st.session_state:
    st.session_state["DROPBOX_TOKEN"] = ""

st.session_state["DROPBOX_TOKEN"] = st.text_input(
    "Enter your Dropbox token",
    type="password",
    value=st.session_state["DROPBOX_TOKEN"]
)

DROPBOX_TOKEN = st.session_state["DROPBOX_TOKEN"]
dbx = None
selected_member_id = None

# Connect to Dropbox
if DROPBOX_TOKEN:
    try:
        dbx = dropbox.Dropbox(DROPBOX_TOKEN)
        current_account = dbx.users_get_current_account()
        st.sidebar.success(f"✅ Connected to Dropbox as {current_account.name.display_name} ({current_account.email})")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to connect to Dropbox: {e}")
        st.session_state["DROPBOX_TOKEN"] = ""
        dbx = None
else:
    st.sidebar.error("❌ No Dropbox token entered. Please paste your token above.")

# -----------------------------
# Dropbox Helpers
# -----------------------------
def list_dropbox_files(folder):
    try:
        entries = dbx.files_list_folder(folder).entries
        files = [entry for entry in entries if isinstance(entry, dropbox.files.FileMetadata) and entry.name.lower().endswith((".pdf", ".docx", ".txt"))]
        return files
    except Exception as e:
        st.sidebar.error(f"Error accessing Dropbox files: {e}")
        return []

# -----------------------------
# Text Extraction Helpers
# -----------------------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9@.\s+\-()]', '', text)
    return text.lower()

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text(file, filename):
    try:
        if filename.endswith(".pdf"):
            return clean_text(extract_text_from_pdf(file))
        elif filename.endswith(".docx"):
            return clean_text(extract_text_from_docx(file))
        elif filename.endswith(".txt"):
            return clean_text(file.read().decode('utf-8'))
        else:
            return ""
    except Exception as e:
        st.sidebar.error(f"Error extracting text from {filename}: {e}")
        return ""

def download_and_extract_text(file_metadata):
    try:
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
    except Exception as e:
        st.sidebar.error(f"Failed to extract text from {file_metadata.name}: {e}")
    return ""

def extract_email(text):
    match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    return match.group(0) if match else "Not found"

def extract_phone(text):
    match = re.search(r"\+?\d[\d\s\-()]{7,}\d", text)
    return match.group(0) if match else "Not found"

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
        email = extract_email(text)
        phone = extract_phone(text)
        results.append({
            'Candidate': filename,
            'Email': email,
            'Phone': phone,
            'Match Score': round(similarities[i][0] * 100, 2),
            'Matched Keywords': ", ".join(set(matched_keywords))
        })

    return results

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

# Local Upload (Multiple Files)
if use_local:
    st.sidebar.subheader("📁 Load Resumes from Local Folder")
    local_folder = st.sidebar.text_input("Enter local folder path", value="resumes/")

    if local_folder and os.path.exists(local_folder):
        with st.spinner(f"Scanning folder: {local_folder}"):
            supported_exts = ("*.pdf", "*.docx", "*.txt")
            all_files = []
            for ext in supported_exts:
                all_files.extend(glob.glob(os.path.join(local_folder, ext)))

            if all_files:
                for file_path in all_files:
                    filename = os.path.basename(file_path)
                    with open(file_path, "rb") as f:
                        resume_texts[filename] = extract_text(f, filename)
                st.sidebar.success(f"✅ Loaded {len(all_files)} resumes from {local_folder}")
            else:
                st.sidebar.warning("⚠️ No supported resume files found in this folder.")
    elif local_folder:
        st.sidebar.error("❌ Folder not found. Please check the path.")

# Dropbox Upload
if use_dropbox:
    if dbx:
        st.sidebar.subheader("📂 Dropbox Folder Input")
        dropbox_folder_path = st.sidebar.text_input(
            "Enter Dropbox Folder Path", 
            value="/Resumes/SRE", 
            help="Example: /Resumes/SRE or /Recruitment Candidates/Java"
        )

        if dropbox_folder_path:
            file_metadata_list = list_dropbox_files(dropbox_folder_path)
            if file_metadata_list:
                with st.spinner(f"📥 Loading resumes from: {dropbox_folder_path}"):
                    for file_meta in file_metadata_list:
                        resume_texts[file_meta.name] = download_and_extract_text(file_meta)
                st.sidebar.success(f"✅ Loaded {len(file_metadata_list)} resumes from {dropbox_folder_path}")
            else:
                st.sidebar.warning("⚠️ No resumes found in the specified Dropbox folder.")
    else:
        st.sidebar.warning("⚠️ Dropbox not connected. Check your token and try again.")

# Run Matching
if job_file and resume_texts:
    job_text = extract_text(job_file, job_file.name)

    with st.spinner("Ranking candidates..."):
        ranked = rank_candidates(resume_texts, job_text)
        df = pd.DataFrame(ranked)

        # Extract top job keywords based on TF-IDF from the job description
        job_vectorizer = TfidfVectorizer(stop_words='english')
        job_tfidf = job_vectorizer.fit_transform([job_text])
        tfidf_scores = list(zip(job_vectorizer.get_feature_names_out(), job_tfidf.toarray()[0]))
        sorted_keywords = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        top_keywords = [kw for kw, score in sorted_keywords[:10]]

        df.insert(1, 'Top JD Keywords', ", ".join(top_keywords))

        def top_keyword_score(matched_keywords_str):
            if not matched_keywords_str:
                return 0.0
            matched_set = set(matched_keywords_str.split(", "))
            return len(set(top_keywords).intersection(matched_set)) / len(top_keywords) if top_keywords else 0

        df['Top Keyword Match Ratio'] = df['Matched Keywords'].apply(top_keyword_score)
        top_weight = 0.7
        similarity_weight = 0.3
        df['Final Suitability Score'] = top_weight * df['Top Keyword Match Ratio'] + similarity_weight * (df['Match Score'] / 100)
        df = df.sort_values(by=["Final Suitability Score"], ascending=False)

        def evaluate_candidate(row):
            matched = set(row['Matched Keywords'].split(", "))
            top_matched = set(top_keywords[:5]).intersection(matched)
            missing = set(top_keywords[:5]) - matched
            reason = f"✅ Matched top keywords: {', '.join(top_matched)}."
            weakness = f"⚠️ Missing important keywords: {', '.join(missing)}."
            return reason + " " + weakness

        df['Evaluation Summary'] = df.apply(evaluate_candidate, axis=1)

        st.success("Done! Here are the results:")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", csv, "ranked_candidates.csv", "text/csv")

elif not job_file:
    st.info("Please upload a job description to continue.")
elif not resume_texts:
    st.info("Please upload or select resumes to continue.")
