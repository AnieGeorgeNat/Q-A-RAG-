import chromadb
from fastapi import APIRouter, HTTPException
from langchain_huggingface import HuggingFaceEmbeddings

router = APIRouter()

CHROMA_PATH = "backend/vector_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="documents")

embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")

@router.get("/search/")
async def search_docs(query: str):
    """Retrieve top matching document chunks."""
    try:
        query_embedding = embedding_function.embed_query(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        if "documents" in results and results["documents"]:
            return {"matches": results["documents"][0]}
        else:
            return {"matches": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
