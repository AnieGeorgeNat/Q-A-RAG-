from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from backend.retrieval import collection
from backend.dependencies import embedding_function



def chunk_document(file_path, file_hash):
    """Splits document into chunks and stores embeddings in ChromaDB."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc_chunks = text_splitter.split_documents(documents)

    chunk_count = len(doc_chunks)
    
    for i, chunk in enumerate(doc_chunks):
        collection.add(
            ids=[f"{file_hash}_chunk_{i}"],
            documents=[chunk.page_content],
            embeddings=[embedding_function.embed_query(chunk.page_content)]
        )

    return chunk_count
