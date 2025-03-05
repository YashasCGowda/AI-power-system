import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from the uploaded PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\\n"
    return text.strip() if text else "No readable text found."

# Function to rank resumes based on cosine similarity
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities

# Streamlit UI layout and styling
st.markdown("""
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #2a9d8f;
            text-align: center;
        }
        .subheader {
            font-size: 28px;
            color: #264653;
            margin-top: 20px;
        }
        .upload-box {
            border: 2px dashed #264653;
            padding: 20px;
            border-radius: 10px;
            background-color: #e9f5f5;
        }
        .button {
            background-color: #2a9d8f;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            font-size: 16px;
            cursor: pointer;
        }
        .resume-list {
            margin-top: 20px;
            font-size: 18px;
        }
        .resume-item {
            padding: 10px;
            background-color: #f4f4f4;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .score {
            font-weight: bold;
            color: #e76f51;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">AI Resume Screening & Ranking System</div>', unsafe_allow_html=True)

# Input for the job description
job_description = st.text_area("Enter the job description", height=200)

# Upload resumes
uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

# Display results if files and job description are uploaded
if uploaded_files and job_description:
    resumes = [extract_text_from_pdf(file) for file in uploaded_files]
    scores = rank_resumes(job_description, resumes)

    ranked_resumes = sorted(zip(uploaded_files, scores), key=lambda x: x[1], reverse=True)

    # Display ranked resumes
    st.markdown('<div class="subheader">Ranked Resumes</div>', unsafe_allow_html=True)
    for i, (file, score) in enumerate(ranked_resumes, start=1):
        st.markdown(f'<div class="resume-item">{i}. {file.name} - <span class="score">{score:.2f}</span></div>', unsafe_allow_html=True)

