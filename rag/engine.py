from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)
from dotenv import load_dotenv

load_dotenv()

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
        self.max_questions = 10
        self.prompt = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        """
        You are a professional technical interviewer.
        Conduct a realistic interview based on the candidate's resume.
        Use the resume to generate relevant questions.
        Cover different areas of the resume such as:
        - technical skills
        - projects
        - work experience
        - education
        - achievements
        Do not repeatedly ask about the same project or technology.
        Prefer asking follow-up questions only once before moving to another topic.
        Avoid repeating previous questions.

        Current Difficulty:
        {difficulty}

        Question Number:
        {question_number}

        Ask exactly ONE concise interview question.
        Do not provide hints, explanations, or answers.
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
                        "Ask the next interview question."
                    )
                ])
    def get_first_question(self):
        self.question_count += 1
        messages = self.prompt.invoke({
            "context": self.context,
            "history": self.history,
            "difficulty": self.difficulty,
            "question_number": self.question_count
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
            "question_number": self.question_count
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