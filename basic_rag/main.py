# Check out the blog post for in depth explanation on RAG: https://medium.com/@thisiskuhan/into-the-rag-verse-the-origin-and-the-why-0b80350d1e17

from fastapi import FastAPI, UploadFile, File, Form
from utils import make_md, embed, qdrant_ops, semantic_chunk_text, create_client, search
from openai import OpenAI
import os

# -----------------------------
# Config
# -----------------------------
QDRANT_URL = "https://your_url_here"
QDRANT_API_KEY = "your_api_key_here"
COLLECTION_NAME = "dummy"
OPENROUTER_API_KEY = "your_api_key_here"

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI()

# Create clients once
qdrant_client = create_client(QDRANT_URL, QDRANT_API_KEY)
openai_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)


# -----------------------------
# Ingest PDF Endpoint
# -----------------------------
@app.post("/ingest")
async def ingest_doc(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are allowed!"}
    
    # Save uploaded file
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Extract and clean content
    clean_content = make_md(file_path)

    # Chunk text
    chunks = semantic_chunk_text(clean_content, max_tokens=25, overlap=5)
    print(f"Created {len(chunks)} chunks")

    # Embed chunks
    embedded, size = embed(chunks)

    # Store chunks in Qdrant
    qdrant_ops(qdrant_client, COLLECTION_NAME, size, embedded, chunks)

    return {"message": f"Document ingested with {len(chunks)} chunks."}


# -----------------------------
# Raw Qdrant Query Endpoint
# -----------------------------
@app.post("/query")
async def query_docs(query: str = Form(...), top_k: int = Form(3)):
    """
    Query the Qdrant collection and return top K relevant chunks.
    """
    hits = search(qdrant_client, COLLECTION_NAME, query, top_k)

    # Format hits
    results = [
        {
            "id": hit.id,
            "score": hit.score,
            "text": hit.payload.get("text", "")
        }
        for hit in hits
    ]

    return {"query": query, "results": results}


# -----------------------------
# Qdrant + LLM Endpoint
# -----------------------------
@app.post("/query_llm")
async def query_llm(query: str = Form(...), top_k: int = Form(3)):
    """
    Retrieve top-K relevant chunks from Qdrant and pass them to an LLM for response.
    """
    # Step 1: Retrieve relevant chunks
    hits = search(qdrant_client, COLLECTION_NAME, query, top_k)
    
    # Combine retrieved text as context
    context_texts = [hit.payload.get("text", "") for hit in hits]
    context = "\n".join(context_texts)

    # Step 2: Create a prompt with retrieved context
    system_prompt = (
        "You are a helpful assistant. Use the retrieved context to answer the user query. "
        "If the answer is not in the context, say you don't know."
    )
    user_prompt = f"Context:\n{context}\n\nUser Question: {query}"

    # Step 3: Call OpenRouter/OpenAI chat model
    completion = openai_client.chat.completions.create(
        extra_body={},
        model="mistralai/mistral-small-3.2-24b-instruct:free",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    answer = completion.choices[0].message.content

    return {
        "query": query,
        "retrieved_context": context_texts,
        "llm_answer": answer
    }


# -----------------------------
# Run FastAPI
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
