from langchain_chroma import Chroma

from rag.embeddings import embedding_model


def create_vector_store(chunks):

    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory="vector_db"
    )

    return vector_store