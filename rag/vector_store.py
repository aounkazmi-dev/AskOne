from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def create_vector_store(chunks):
    embedding_model = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model
    )
    
    return vector_store