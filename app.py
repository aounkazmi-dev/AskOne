import streamlit as st
import os
import shutil

from rag.vector_store import create_vector_store
from utils.pdf_utils import extract_text_from_pdf
from rag.chunking import chunk_resume
from rag.engine import InterviewEngine


st.set_page_config(page_title="AskOne - AI Resume Interviewer", page_icon="🧑‍💻")

st.title("AI Resume Interviewer")
st.caption("Phase 1: Upload your resume and extract the text")

uploaded_file = st.file_uploader(
    label="Upload your resume (PDF only)",
    type=["pdf"],
)

if uploaded_file is not None:

    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join("uploads", uploaded_file.name)

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    st.success("Resume uploaded successfully!")

    resume_text = extract_text_from_pdf(file_path)
    resume_chunks = chunk_resume(resume_text)

    if os.path.exists("vector_db"):
        shutil.rmtree("vector_db")

    db = create_vector_store(resume_chunks)
    retriever = db.as_retriever(search_kwargs={"k": 2})

    docs = retriever.invoke("skills")
    context = "\n".join([doc.page_content for doc in docs])

    # -------------------------
    # SESSION STATE SETUP
    # -------------------------
    if "engine" not in st.session_state:
        st.session_state.engine = InterviewEngine(context)

    if "history" not in st.session_state:
        st.session_state.history = []

    if "current_question" not in st.session_state:
        st.session_state.current_question = None

    if "step" not in st.session_state:
        st.session_state.step = "start"

    # -------------------------
    # FIRST QUESTION
    # -------------------------
    if st.session_state.step == "start":
        st.session_state.current_question = st.session_state.engine.get_first_question()
        st.session_state.step = "ask"

    # -------------------------
    # SHOW QUESTION
    # -------------------------
    st.write("AI:", st.session_state.current_question)

    # -------------------------
    # USER ANSWER
    # -------------------------
    user_answer = st.text_input("Your answer")

    # -------------------------
    # NEXT STEP
    # -------------------------
    if st.button("Submit") and user_answer:

        # store history
        st.session_state.history.append({
            "q": st.session_state.current_question,
            "a": user_answer
        })

        # get next question
        st.session_state.current_question = st.session_state.engine.get_next_question(user_answer)

        # clear input effect (Streamlit workaround)
        st.session_state.step = "ask"

        st.rerun()