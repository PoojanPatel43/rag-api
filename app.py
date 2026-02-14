import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import chromadb
import ollama

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "tinyllama")

app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API with ChromaDB and Ollama",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection("docs")
ollama_client = ollama.Client(host=OLLAMA_HOST)


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy"}


@app.get("/stats")
def get_stats():
    """Get statistics about the document collection."""
    count = collection.count()
    return {
        "collection_name": "docs",
        "document_count": count
    }


@app.post("/query")
def query(q: str, n_results: int = 1):
    """Query the RAG system with a question."""
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if n_results < 1 or n_results > 10:
        raise HTTPException(status_code=400, detail="n_results must be between 1 and 10")

    results = collection.query(query_texts=[q], n_results=n_results)
    context = "\n\n".join(results["documents"][0]) if results["documents"] else ""

    answer = ollama_client.generate(
        model=MODEL_NAME,
        prompt=f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer clearly and concisely:"
    )

    return {"answer": answer["response"], "context_used": bool(context)}
