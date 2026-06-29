import streamlit as st
import os
import shutil
from rag.vector_store import create_vector_store
from utils.pdf_utils import extract_text_from_pdf
from rag.chunking import chunk_resume
from rag.vector_store import create_vector_store

st.set_page_config(page_title="AskOne - AI Resume Interviewer", page_icon="🧑‍💻")

st.title("AI Resume Interviewer")
st.caption("Phase 1: Upload your resume and extract the text")

uploaded_file = st.file_uploader(
    label="Upload your resume (PDF only)",
    type=["pdf"],
)
if uploaded_file is not None:

    # Create uploads folder if it doesn't exist
    os.makedirs("uploads", exist_ok=True)

    # Path where the PDF will be saved
    file_path = os.path.join("uploads", uploaded_file.name)

    # Save uploaded file
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    st.success("Resume uploaded successfully!")
    resume_text = extract_text_from_pdf(file_path)
    resume_chunks = chunk_resume(resume_text)

    # Delete old vector database (for development only)
    if os.path.exists("vector_db"):
        shutil.rmtree("vector_db")

    db = create_vector_store(resume_chunks)
    retriever = db.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke("What skills does the candidate have?")
    for doc in docs:
      st.write(doc.page_content)