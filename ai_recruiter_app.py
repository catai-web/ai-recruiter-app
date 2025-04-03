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

# Connection Test
if DROPBOX_TOKEN:
    try:
        dbx = dropbox.Dropbox(DROPBOX_TOKEN)
        current_account = dbx.users_get_current_account()
        st.sidebar.success(f"✅ Connected to Dropbox as {current_account.name.display_name} ({current_account.email})")
    except dropbox.exceptions.BadInputError as e:
        if "Dropbox-API-Select-User" in str(e):
            try:
                dbx_team = dropbox.DropboxTeam(DROPBOX_TOKEN)
                members = dbx_team.team_members_list().members

                member_options = {}
                for m in members:
                    status_tag_func = getattr(m.profile.status, 'tag', None)
                    if callable(status_tag_func) and status_tag_func() == "active":
                        member_options[m.profile.email] = m.profile.team_member_id
                    elif hasattr(m.profile.status, 'is_active') and m.profile.status.is_active:
                        member_options[m.profile.email] = m.profile.team_member_id

                if member_options:
                    selected_email = st.sidebar.selectbox("👤 Select a team member to act as", list(member_options.keys()))
                    selected_member_id = member_options[selected_email]

                    dbx = dbx_team.as_user(selected_member_id)
                    current_account = dbx.users_get_current_account()
                    st.sidebar.success(f"✅ Acting as: {current_account.name.display_name} ({current_account.email})")
                else:
                    st.sidebar.error("❌ No active team members found.")

            except Exception as member_error:
                st.sidebar.error(f"❌ Failed to impersonate team member: {member_error}")
                st.session_state["DROPBOX_TOKEN"] = ""
                dbx = None
        else:
            st.sidebar.error(f"❌ Failed to connect to Dropbox: {e}")
            st.session_state["DROPBOX_TOKEN"] = ""
            dbx = None
    except Exception as general_error:
        st.sidebar.error(f"❌ General Dropbox connection error: {general_error}")
        dbx = None

    if dbx:
        try:
            current_account = dbx.users_get_current_account()
            st.sidebar.success(f"✅ Connected to Dropbox as {current_account.name.display_name}")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to fetch Dropbox account details: {e}")
            dbx = None
else:
    st.sidebar.error("❌ No Dropbox token entered. Please paste your token above.")

# -----------------------------
# Dropbox Helpers
# -----------------------------
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

def walk_dropbox_folder(path="", depth=0, max_depth=2):
    if depth > max_depth:
        return
    try:
        entries = dbx.files_list_folder(path).entries
        for entry in entries:
            st.sidebar.write(f"- {entry.path_display} ({type(entry).__name__})")
            if isinstance(entry, dropbox.files.FolderMetadata):
                walk_dropbox_folder(entry.path_lower, depth + 1, max_depth)
    except Exception as e:
        st.sidebar.write(f"  ⚠️ Error accessing {path}: {e}")

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

if use_local:
    resume_files = st.sidebar.file_uploader("Upload Resumes", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    if resume_files:
        resume_texts.update({res.name: extract_text(res, res.name) for res in resume_files})

if use_dropbox:
    if dbx:
        st.sidebar.subheader("Dropbox Resumes")
        with st.spinner("Fetching Dropbox folder structure..."):
            walk_dropbox_folder()

        folder_options = list_dropbox_folders("/Recruitment Candidates")

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
            root_files = list_dropbox_files("/Recruitment Candidates")
            if root_files:
                selected_files = st.sidebar.multiselect("Select Resumes from Root", options=root_files, format_func=lambda f: f.name)
                for file_meta in selected_files:
                    resume_texts[file_meta.name] = download_and_extract_text(file_meta)
            else:
                st.sidebar.warning("No resumes found in root directory either.")
    else:
        st.sidebar.warning("⚠️ Dropbox not connected. Check your token and try again.")

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

        df.insert(1, 'Top JD Keywords', top_keywords)

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

        st.success("Done! Here are the results:")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", csv, "ranked_candidates.csv", "text/csv")

elif not job_file:
    st.info("Please upload a job description to continue.")
elif not resume_texts:
    st.info("Please upload or select resumes to continue.")
