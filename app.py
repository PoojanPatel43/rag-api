from fastapi import FastAPI
import chromadb
import ollama

app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API with ChromaDB and Ollama",
    version="1.0.0"
)

chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection("docs")
ollama_client = ollama.Client(host="http://host.docker.internal:11434")


@app.get("/health")
def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy"}


@app.post("/query")
def query(q: str):
    results = collection.query(query_texts=[q], n_results=1)
    context = results["documents"][0][0] if results["documents"] else ""
    
    answer = ollama_client.generate(
    model="tinyllama",
    prompt=f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer clearly and concisely:"
)


    return {"answer": answer["response"]}
