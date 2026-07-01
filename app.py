import streamlit as st
import os
import shutil

from rag.vector_store import create_vector_store
from utils.pdf_utils import extract_text_from_pdf
from rag.chunking import chunk_resume
from rag.engine import InterviewEngine
from utils.report_utils import generate_interview_report

st.set_page_config(page_title="AskOne - AI Resume Interviewer", page_icon="🧑‍💻", layout="centered")

st.markdown(
    """
    <style>
    .stApp { background-color: #0f1117; }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-caption {
        text-align: center;
        color: #9aa0a6;
        margin-bottom: 2rem;
    }
    .question-card {
        background-color: #1a1d27;
        border: 1px solid #2a2e3a;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
    }
    .question-label {
        color: #7c9bff;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .question-text {
        font-size: 1.15rem;
        line-height: 1.5;
    }
    .end-card {
        background: linear-gradient(135deg, #1a1d27, #20242f);
        border: 1px solid #2a2e3a;
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .end-icon { font-size: 3rem; margin-bottom: 0.5rem; }
    .end-title { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.4rem; }
    .end-subtitle { color: #9aa0a6; margin-bottom: 1.5rem; }
    .history-item {
        background-color: #1a1d27;
        border: 1px solid #2a2e3a;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .history-q { color: #7c9bff; font-weight: 600; margin-bottom: 0.3rem; }
    .history-a { color: #d6d6d6; }
    .feedback-card {
        background-color: #1a1d27;
        border: 1px solid #2a2e3a;
        border-left: 4px solid #7c9bff;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        color: #d6d6d6;
        line-height: 1.7;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🧑‍💻 AI Resume Interviewer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-caption">AI-powered Resume Interview</div>', unsafe_allow_html=True)


if "started" not in st.session_state:
    st.session_state.started = False

if "engine" not in st.session_state:
    st.session_state.engine = None

if "current_question" not in st.session_state:
    st.session_state.current_question = None

if "history" not in st.session_state:
    st.session_state.history = []

if "interview_ended" not in st.session_state:
    st.session_state.interview_ended = False

if "feedback" not in st.session_state:
    st.session_state.feedback = ""


if st.session_state.interview_ended:

    st.markdown(
        """
        <div class="end-card">
            <div class="end-icon">✅</div>
            <div class="end-title">Interview Completed!</div>
            <div class="end-subtitle">Great job! Here's a summary and feedback for your session.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.history:
        st.write("### 📋 Interview Summary")
        for i, item in enumerate(st.session_state.history, start=1):
            st.markdown(
                f"""
                <div class="history-item">
                    <div class="history-q">Q{i}: {item['question']}</div>
                    <div class="history-a">{item['answer']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    if st.session_state.feedback:
        st.write("### 🤖 AI Feedback & Suggestions")
        st.markdown(
            f'<div class="feedback-card">{st.session_state.feedback}</div>',
            unsafe_allow_html=True
        )

    st.write("")

    pdf_bytes = generate_interview_report(st.session_state.history, st.session_state.feedback)
    st.download_button(
        label="📄 Download Interview Report (PDF)",
        data=pdf_bytes,
        file_name="interview_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    if st.button("🔄 Start a New Interview", use_container_width=True):
        st.session_state.started = False
        st.session_state.engine = None
        st.session_state.current_question = None
        st.session_state.history = []
        st.session_state.interview_ended = False
        st.session_state.feedback = ""
        st.rerun()

elif not st.session_state.started:

    uploaded_file = st.file_uploader(
        "Upload your resume (PDF only)",
        type=["pdf"]
    )

    if uploaded_file is not None:

        st.success("Resume uploaded successfully!")

        if st.button("Start Interview"):

            with st.status("Preparing your interview...", expanded=True) as status:

                st.write("📄 Reading your resume...")
                os.makedirs("uploads", exist_ok=True)
                file_path = os.path.join("uploads", uploaded_file.name)
                with open(file_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())
                resume_text = extract_text_from_pdf(file_path)

                st.write("✂️ Processing resume content...")
                resume_chunks = chunk_resume(resume_text)

                st.write("🗄️ Building knowledge base...")
                db = create_vector_store(resume_chunks)

                st.write("🔍 Extracting key information...")
                retriever = db.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke("skills projects experience")
                context = "\n".join(doc.page_content for doc in docs)

                st.write("🤖 Generating your first question...")
                engine = InterviewEngine(context)
                st.session_state.engine = engine
                st.session_state.current_question = engine.get_first_question()
                st.session_state.started = True

                status.update(label="✅ Interview ready!", state="complete", expanded=False)

            st.rerun()


else:

    st.write("### 🎙️ AI Interviewer")

    st.markdown(
        f"""
        <div class="question-card">
            <div class="question-label">Question</div>
            <div class="question-text">{st.session_state.current_question}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    user_answer = st.text_area("Your Answer")

    if st.button("Submit Answer"):

        st.session_state.history.append({
            "question": st.session_state.current_question,
            "answer": user_answer
        })

        next_question = st.session_state.engine.get_next_question(user_answer)

        if next_question is None:
            st.session_state.started = False
            st.session_state.interview_ended = True
            with st.spinner("Analysing your responses and generating feedback..."):
                st.session_state.feedback = st.session_state.engine.get_interview_feedback(
                    st.session_state.history
                )
        else:
            st.session_state.current_question = next_question

        st.rerun()