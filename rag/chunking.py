from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_resume(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    return chunks