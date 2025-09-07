# Naive RAG

A simple FastAPI project that allows you to ingest PDF documents, store their content in a Qdrant vector database, and query them using OpenRouter/OpenAI’s large language models

## Requirements

- Python 3.8+
- Qdrant account and API key
- OpenRouter/OpenAI API key

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/thisiskuhan/Into-The-Rag-Verse.git
   cd Into-The-Rag-Verse/basic_rag
   ```

2. Create a virtual environment

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

4. Configure the project

   - Update the `QDRANT_URL` and `QDRANT_API_KEY` in `main.py` to point to your Qdrant instance and API key.
   - Update the `OPENROUTER_API_KEY` in `main.py` to your OpenRouter/OpenAI API key.

5. Run the project

   ```bash
   python main.py
   ```

## Usage

Go to `http://localhost:8000/docs` to see the API documentation.

### Endpoints

- `/ingest`: Ingest a PDF file and store its content in the Qdrant database.
- `/query`: Query the Qdrant database for the top-K most relevant chunks.
- `/query_llm`: Query the LLM for the top-K most relevant chunks.

## Models

Mistral: Mistral Small 3.2 24B (free)

## Limitations

- This is a simple example and does not include error handling or robustness.
- This supports only text chunks and does not support data of other modalities.

## Read the blog post

Check out the [blog post](https://medium.com/@thisiskuhan/into-the-rag-verse-the-origin-and-the-why-0b80350d1e17) for in depth explanation on RAG.

## Author

Kuhan Sundaram (@thisiskuhan)

## Happy learning! 🚀
