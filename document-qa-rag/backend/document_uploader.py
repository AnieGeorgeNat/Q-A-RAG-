import os
import hashlib
import json
from fastapi import APIRouter, UploadFile, File, HTTPException
from langchain_community.document_loaders import PyPDFLoader
from backend.retrieval import collection
from backend.chunking import chunk_document
from backend.dependencies import embedding_function

from typing import Dict, List

router = APIRouter()

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(DATA_DIR, exist_ok=True)

METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")

# Load metadata if exists
if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, "r") as f:
        metadata_store = json.load(f)
else:
    metadata_store = {}

def hash_file(file_path):
    """Generate a unique hash for a file to prevent duplicate uploads."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def extract_metadata(file_path):
    """Extract metadata from the uploaded PDF (page count, summary)."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    page_count = len(documents)
    summary = documents[0].page_content[:200] if documents else "No content extracted."
    
    return page_count, summary

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_contents = await file.read()  # Read file once

        file_path = os.path.join(DATA_DIR, file.filename)
        temp_path = f"{file_path}.tmp"
        
        # Write the file to a temporary location
        with open(temp_path, "wb") as f:
            f.write(file_contents)

        file_hash = hash_file(temp_path)
        if file_hash in metadata_store:
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail="Duplicate file detected.")

        os.rename(temp_path, file_path)

        # Extract metadata
        page_count, summary = extract_metadata(file_path)

        # Process document into chunks & store embeddings
        chunk_count = chunk_document(file_path, file_hash)

        # Store metadata
        metadata_store[file_hash] = {
            "filename": file.filename,
            "path": file_path,
            "page_count": page_count,
            "summary": summary,
            "chunk_count": chunk_count
        }
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata_store, f)

        return {
            "message": f"File '{file.filename}' uploaded successfully",
            "metadata": metadata_store[file_hash]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")



@router.delete("/delete/{filename}")
async def delete_file(filename: str):
    """Delete a document and its corresponding embeddings."""
    try:
        file_hash = None
        for hash_key, data in metadata_store.items():
            if data["filename"] == filename:
                file_hash = hash_key
                break
        
        if not file_hash:
            raise HTTPException(status_code=404, detail="File not found.")

        # Remove from storage
        file_path = metadata_store[file_hash]["path"]
        if os.path.exists(file_path):
            os.remove(file_path)

        # Remove from ChromaDB
        collection.delete(where={"document_id": file_hash})  # Delete embedding by ID

        # Remove metadata entry
        del metadata_store[file_hash]
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata_store, f)

        return {"message": f"File '{filename}' deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


@router.get("/list/")
async def list_documents() -> Dict:
    """List all uploaded documents with their metadata."""
    documents = []
    for file_hash, data in metadata_store.items():
        documents.append({
            "filename": data["filename"],
            "page_count": data["page_count"],
            "summary": data["summary"],
            "chunk_count": data["chunk_count"]
        })
    return {"documents": documents}

@router.get("/get_chunk/{filename}/{chunk_number}")
async def get_chunk(filename: str, chunk_number: int):
    """Retrieve a specific chunk from a document."""
    try:
        # Find file hash from filename
        file_hash = None
        for hash_key, data in metadata_store.items():
            if data["filename"] == filename:
                file_hash = hash_key
                break
        
        if not file_hash:
            raise HTTPException(status_code=404, detail="File not found")
            
        # Get chunk from ChromaDB
        chunk_id = f"{file_hash}_chunk_{chunk_number}"
        results = collection.get(ids=[chunk_id])
        
        if not results or not results["documents"]:
            raise HTTPException(status_code=404, detail="Chunk not found")
            
        return {"content": results["documents"][0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chunk: {str(e)}")