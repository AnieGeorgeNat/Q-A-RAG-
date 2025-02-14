# Calls an LLM for final answers
import google.generativeai as genai
import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.retrieval import collection

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Gemini API client
genai.configure(api_key=os.getenv("GENAI_API_KEY"))

# Initialize the model
client = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Initialize embedding model
embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")

# Create router
router = APIRouter()

class QuestionRequest(BaseModel):
    document: str
    question: str

def generate_answer(query: str, retrieved_docs: list) -> str:
    """Use Gemini to generate an answer based on retrieved documents."""
    try:
        context = "\n".join(retrieved_docs) if retrieved_docs else "No relevant documents found."
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        logging.info(f"Prompt sent to Gemini: {prompt}")

        response = client.generate_content(prompt)

        logging.info(f"Gemini response: {response}")

        if response and hasattr(response, "text"):
            return response.text.strip()
        else:
            return "No response generated."

    except Exception as e:
        logging.error(f"Gemini API call failed: {e}")
        return "Error generating response from LLM."


@router.post("/ask/")
async def ask_question(request: QuestionRequest):
    """Generate answer for a question about a specific document."""
    try:
        logging.info(f"Received question: {request.question} for document: {request.document}")

        # Create embedding for the question
        query_embedding = embedding_function.embed_query(request.question)

        # Debugging: Print query embedding
        logging.info(f"Query embedding: {query_embedding[:5]}...")

        # Search for relevant chunks in ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        # Debugging: Print retrieval results
        logging.info(f"ChromaDB results: {results}")

        # Get the retrieved documents
        retrieved_docs = results["documents"][0] if results.get("documents") else []
        
        if not retrieved_docs:
            logging.warning("No documents retrieved from ChromaDB.")

        # Generate answer using Gemini
        answer = generate_answer(request.question, retrieved_docs)

        return {"answer": answer}

    except Exception as e:
        logging.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

