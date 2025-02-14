from fastapi import FastAPI
from backend.document_uploader import router as upload_router
from backend.retrieval import router as retrieval_router
from backend.llm_query import router as query_router

app = FastAPI()

# Mount routers with appropriate prefixes
app.include_router(upload_router, prefix="/documents", tags=["Documents"])
app.include_router(retrieval_router, prefix="/documents", tags=["Documents"])
app.include_router(query_router, prefix="/documents", tags=["Documents"])

@app.get("/")
def root():
    return {"message": "Welcome to the Intelligent Document QA System!"}
