import streamlit as st
import os
import shutil

from rag.vector_store import create_vector_store
from utils.pdf_utils import extract_text_from_pdf
from rag.chunking import chunk_resume
from rag.engine import InterviewEngine

st.set_page_config(page_title="AskOne - AI Resume Interviewer", page_icon="🧑‍💻")

st.title("AI Resume Interviewer")
st.caption("AI-powered Resume Interview")

# -------------------------
# SESSION STATE
# -------------------------

if "started" not in st.session_state:
    st.session_state.started = False

if "engine" not in st.session_state:
    st.session_state.engine = None

if "current_question" not in st.session_state:
    st.session_state.current_question = None

if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================================
# BEFORE INTERVIEW STARTS
# ==========================================================

if not st.session_state.started:

    uploaded_file = st.file_uploader(
        "Upload your resume (PDF only)",
        type=["pdf"]
    )

    if uploaded_file is not None:

        st.success("Resume uploaded successfully!")

        if st.button("Start Interview"):

            os.makedirs("uploads", exist_ok=True)

            file_path = os.path.join("uploads", uploaded_file.name)

            with open(file_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            resume_text = extract_text_from_pdf(file_path)
            resume_chunks = chunk_resume(resume_text)

            if os.path.exists("vector_db"):
                shutil.rmtree("vector_db")

            db = create_vector_store(resume_chunks)

            retriever = db.as_retriever(
                search_kwargs={"k": 2}
            )

            docs = retriever.invoke("skills projects experience")

            context = "\n".join(
                doc.page_content
                for doc in docs
            )

            engine = InterviewEngine(context)

            st.session_state.engine = engine

            st.session_state.current_question = (
                engine.get_first_question()
            )

            st.session_state.started = True

            st.rerun()

# ==========================================================
# INTERVIEW
# ==========================================================

else:

    st.write("### AI Interviewer")

    st.write(st.session_state.current_question)

    user_answer = st.text_area("Your Answer")

    if st.button("Submit Answer"):

        st.session_state.history.append({
            "question": st.session_state.current_question,
            "answer": user_answer
        })

        next_question = (
            st.session_state.engine.get_next_question(user_answer)
        )

        st.session_state.current_question = next_question

        st.rerun()