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
    def get_first_question(self):
        # Generate the first question using the LLM
        first_question = self.llm.generate_first_question(self.context)
        return first_question
    


    