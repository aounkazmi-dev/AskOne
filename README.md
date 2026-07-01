# 🧑‍💻 AskOne — AI Resume Interviewer

AskOne is an AI-powered mock interview app. Upload your resume as a PDF, and it conducts a structured, resume-grounded technical interview using RAG (Retrieval-Augmented Generation) — then generates a downloadable PDF report with personalized feedback.

## ✨ Features

- 📄 **Resume-aware questions** — extracts and embeds your resume so every question is grounded in your actual projects, skills, and experience.
- 🎯 **Structured 10-question flow** — covers introduction, education, skills, projects, experience, and behavioral topics.
- 📈 **Adaptive difficulty** — questions get harder as the interview progresses (Easy → Medium → Hard).
- 🤖 **AI feedback report** — after the interview, get a strengths/weaknesses breakdown and actionable suggestions.
- 📥 **Downloadable PDF report** — full Q&A history plus feedback, exportable as a polished PDF.

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io/) |
| LLM (interview questions & feedback) | [Groq](https://groq.com/) — `llama-3.3-70b-versatile` via `langchain-groq` |
| Embeddings | Google Gemini — `models/gemini-embedding-001` via `langchain-google-genai` |
| Vector store | [Chroma](https://www.trychroma.com/) via `langchain-chroma` |
| PDF parsing | `pypdf` |
| PDF report generation | `fpdf2` |

## 📂 Project Structure

```
Askkone/
├── app.py                     # Streamlit UI and app flow
├── rag/
│   ├── chunking.py            # Splits resume text into chunks
│   ├── embeddings.py          # Gemini embedding model config
│   ├── vector_store.py        # Builds the Chroma vector store
│   └── engine.py              # InterviewEngine: question flow + feedback
├── utils/
│   ├── pdf_utils.py           # Extracts text from uploaded resume PDF
│   └── report_utils.py        # Generates the downloadable PDF report
├── fonts/                     # Fonts used by the PDF report
├── requirements.txt
└── .env                       # API keys (not committed)
```

## ⚙️ Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd Askkone
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your API keys

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

- Get a Google API key (for Gemini embeddings) from [Google AI Studio](https://aistudio.google.com/apikey).
- Get a Groq API key from [Groq Console](https://console.groq.com/keys).

### 3. Run locally

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

## ☁️ Deploying on Render

1. Push the project to GitHub (**do not commit your `.env` file**).
2. Create a new **Web Service** on [Render](https://render.com/), connect your repo.
3. Set the build command:
   ```
   pip install -r requirements.txt
   ```
4. Set the start command:
   ```
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
5. Add `GOOGLE_API_KEY` and `GROQ_API_KEY` as environment variables in Render's dashboard (Settings → Environment).
6. Deploy.

## 🖥️ How It Works

1. **Upload** a PDF resume.
2. The app **extracts text** (`pdf_utils.py`), **chunks** it (`chunking.py`), and embeds it into a **Chroma vector store** (`vector_store.py`) using Gemini embeddings.
3. Relevant resume context is retrieved and passed to `InterviewEngine` (`engine.py`), which uses Groq's Llama 3.3 70B to generate a structured, 10-question interview.
4. After each answer, the engine generates the next question — difficulty increases as the interview progresses.
5. At the end, the engine generates written feedback, and `report_utils.py` compiles the full session into a downloadable PDF.

## 📝 Notes

- Only PDF resumes are supported.
- Uploaded resumes are temporarily saved to a local `uploads/` folder during processing.
- Interview length is fixed at 10 questions (configurable in `engine.py` via `QUESTION_PLAN`).

## 📄 License

Add your preferred license here (e.g. MIT).
