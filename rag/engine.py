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
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional technical interviewer. Use the candidate's resume while asking questions. Ask ONLY ONE question at a time. Do not answer your own question."),
            ("human", "Candidate Resume:\n\n{context}"),
            MessagesPlaceholder("history"),
            ("human", "Ask the next interview question.")
        ])

    def get_first_question(self):
        messages = self.prompt.invoke({
            "context": self.context,
            "history": self.history
        })
        response = self.llm.invoke(messages)
        self.history.append(AIMessage(content=response.content))
        return response.content

    def get_next_question(self, user_answer):
        self.history.append(HumanMessage(content=user_answer))
        messages = self.prompt.invoke({
            "context": self.context,
            "history": self.history
        })
        response = self.llm.invoke(messages)
        self.history.append(AIMessage(content=response.content))
        return response.content
