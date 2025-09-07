# Check out the blog post for in depth explanation on RAG: https://medium.com/@thisiskuhan/into-the-rag-verse-the-origin-and-the-why-0b80350d1e17

"""
Utility functions for extracting, cleaning, chunking, embedding text,
and interacting with Qdrant for vector search.
"""

import re
from typing import List, Tuple, Union

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
import pymupdf4llm


# ----------------------------
# Markdown Extraction and Cleaning
# ----------------------------

def make_md(file_path: str) -> str:
    """
    Extract and clean text from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Cleaned plain text extracted from the PDF.
    """
    md_content = pymupdf4llm.to_markdown(file_path)
    cleaned_text = clean_md(md_content)
    soup = BeautifulSoup(cleaned_text, "html.parser")
    return soup.get_text()


def clean_md(text: str) -> str:
    """
    Clean Markdown-like text into plain text by removing formatting markers.

    Args:
        text (str): Raw markdown text.

    Returns:
        str: Cleaned plain text.
    """
    # Remove bold/italic markers
    text = re.sub(r"(\*{1,2}|_{1,2})(.*?)\1", r"\2", text)

    # Remove headings
    text = re.sub(r"#+\s*", "", text)

    # Remove straight lines made of underscores
    text = re.sub(r"_+", "", text)

    # Replace newlines with space
    text = text.replace("\n", " ")

    # Collapse multiple spaces
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


# ----------------------------
# Text Chunking
# ----------------------------

def semantic_chunk_text(text: str, max_tokens: int = 25, overlap: int = 5) -> List[str]:
    """
    Splits text into semantic chunks with overlap.

    Args:
        text (str): The text to split.
        max_tokens (int): Approximate maximum number of words per chunk.
        overlap (int): Number of overlapping words between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    word_count = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_len = len(sentence_words)

        if word_count + sentence_len > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_chunk.extend(sentence_words)
            word_count = len(current_chunk)
        else:
            current_chunk.extend(sentence_words)
            word_count += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ----------------------------
# Text Embedding
# ----------------------------

def embed(text: Union[str, List[str]]) -> Tuple[List[float], int]:
    """
    Generate embeddings for text using a pre-trained sentence-transformer model.

    Args:
        text (str or List[str]): Text or list of texts to embed.

    Returns:
        Tuple[List[float], int]: Embedding vector(s) and its/their dimension.
    """
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode(text)

    if isinstance(text, str):
        return embeddings.tolist(), len(embeddings)
    else:
        return embeddings.tolist(), len(embeddings[0])


# ----------------------------
# Qdrant Client Operations
# ----------------------------

def create_client(url: str, api_key: str) -> QdrantClient:
    """
    Create a Qdrant client instance.

    Args:
        url (str): Qdrant service URL.
        api_key (str): API key for authentication.

    Returns:
        QdrantClient: Qdrant client object.
    """
    return QdrantClient(url=url, api_key=api_key)


def create_collection(client: QdrantClient, name: str, size: int):
    """
    Create or recreate a collection in Qdrant.

    Args:
        client (QdrantClient): Qdrant client.
        name (str): Name of the collection.
        size (int): Size of vector embeddings.
    """
    client.recreate_collection(
        collection_name=name,
        vectors_config=rest.VectorParams(size=size, distance=rest.Distance.COSINE)
    )
    return client.get_collections()


def qdrant_ops(client: QdrantClient, name: str, size: int, embedding: List[List[float]], chunks: List[str]):
    """
    Upsert text chunks into a Qdrant collection.

    Args:
        client (QdrantClient): Qdrant client.
        name (str): Collection name.
        size (int): Size of vectors.
        embedding (List[List[float]]): List of embedding vectors.
        chunks (List[str]): List of text chunks.
    """
    create_collection(client, name, size)

    points = [
        rest.PointStruct(
            id=i,
            vector=embedding[i],
            payload={"text": chunks[i]}
        )
        for i in range(len(chunks))
    ]

    client.upsert(
        collection_name=name,
        points=points
    )


def search(client: QdrantClient, name: str, query: str, top_k: int = 3) -> List[rest.ScoredPoint]:
    """
    Perform a similarity search in a Qdrant collection.

    Args:
        client (QdrantClient): Qdrant client.
        name (str): Collection name.
        query (str): Query text.
        top_k (int): Number of top results to return.

    Returns:
        List[rest.ScoredPoint]: List of scored results.
    """
    query_vector, _ = embed(query)

    hits = client.search(
        collection_name=name,
        query_vector=query_vector,
        limit=top_k
    )

    return hits
