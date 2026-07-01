from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

QUESTION_PLAN = {
    1:  ("introduction",   "Ask the candidate to briefly introduce themselves and their background."),
    2:  ("education",      "Ask a specific question about their educational background, degree, or relevant coursework."),
    3:  ("skills",         "Ask a conceptual or practical question about one of their listed technical skills."),
    4:  ("projects",       "Ask a specific technical question about one of their projects — focus on implementation details or challenges."),
    5:  ("skills",         "Ask how they used a specific skill in a real project or scenario mentioned on their resume."),
    6:  ("projects",       "Ask about a different project — focus on architecture decisions, tech stack choices, or problems solved."),
    7:  ("experience",     "Ask about their work experience or internship if mentioned, otherwise ask about a personal or academic achievement."),
    8:  ("skills",         "Ask a deeper or advanced question about a technology or tool they listed."),
    9:  ("projects",       "Ask about the outcome, impact, or what they would do differently in one of their projects."),
    10: ("behavioral",     "Ask a behavioral or situational question relevant to their background, such as handling a challenge or working under pressure."),
}

class InterviewEngine:

    def __init__(self, context):
        self.context = context
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=500,
        )
        self.history = []
        self.question_count = 0
        self.difficulty = "Easy"
        self.max_questions = len(QUESTION_PLAN)
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are a professional technical interviewer conducting a structured job interview.

You have the candidate's resume. Follow the instruction below exactly for this question.

YOUR TASK FOR THIS QUESTION:
{task}

Hard rules:
- Ask exactly ONE question based on the task above.
- Ground the question in specific details from the resume — use real project names, company names, skill names, or technologies mentioned.
- NEVER start with "You mentioned", "I see that", "According to your resume", or similar phrases.
- NEVER ask a question you have already asked in this conversation.
- NEVER answer your own question or provide hints.
- Speak naturally as a human interviewer would.

Current Difficulty: {difficulty}
Question Number: {question_number} of {total_questions}
"""
            ),
            (
                "human",
                """
Candidate Resume:

{context}
"""
            ),
            MessagesPlaceholder("history"),
            (
                "human",
                "Ask the question now."
            )
        ])

    def _get_task(self):
        return QUESTION_PLAN.get(self.question_count, (None, "Ask a relevant interview question based on the resume."))[1]

    def get_first_question(self):
        self.question_count += 1
        messages = self.prompt.invoke({
            "context": self.context,
            "history": self.history,
            "difficulty": self.difficulty,
            "question_number": self.question_count,
            "total_questions": self.max_questions,
            "task": self._get_task()
        })
        response = self.llm.invoke(messages)
        self.history.append(AIMessage(content=response.content))
        return response.content

    def get_next_question(self, user_answer):
        if self.question_count >= self.max_questions:
            return None
        self.question_count += 1
        self.update_difficulty()
        self.history.append(HumanMessage(content=user_answer))
        messages = self.prompt.invoke({
            "context": self.context,
            "history": self.history,
            "difficulty": self.difficulty,
            "question_number": self.question_count,
            "total_questions": self.max_questions,
            "task": self._get_task()
        })
        response = self.llm.invoke(messages)
        self.history.append(AIMessage(content=response.content))
        return response.content

    def update_difficulty(self):
        if self.question_count <= 3:
            self.difficulty = "Easy"
        elif self.question_count <= 6:
            self.difficulty = "Medium"
        else:
            self.difficulty = "Hard"

    def get_interview_feedback(self, history: list) -> str:
        qa_text = ""
        for i, item in enumerate(history, start=1):
            qa_text += f"Q{i}: {item['question']}\nAnswer: {item['answer']}\n\n"

        prompt = f"""
You are a senior technical interviewer. Review the following interview Q&A and provide:

1. **Overall Performance Summary** - A brief paragraph on how the candidate did overall.
2. **Strengths** - What the candidate answered well.
3. **Areas for Improvement** - Where the candidate was weak or unclear.
4. **Suggestions** - Specific, actionable tips to improve.

Interview Q&A:
{qa_text}

Be honest, constructive, and specific. Format clearly with headings.
"""
        response = self.llm.invoke(prompt)
        return response.content