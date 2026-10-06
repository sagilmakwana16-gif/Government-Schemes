import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Configuration constants
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "admission_knowledge_base")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")


class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB compatible EmbeddingFunction that utilizes Google GenAI embeddings.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = GEMINI_EMBEDDING_MODEL):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required for Gemini embeddings.")
        self.model = model
        self.client = genai.Client(api_key=self.api_key)

    def __call__(self, input: Documents) -> Embeddings:
        """
        Embeds a list of documents in batches.
        """
        if not input:
            return []

        embeddings_result: List[List[float]] = []
        batch_size = 20  # Batch to stay well within API limits

        for i in range(0, len(input), batch_size):
            batch = input[i:i + batch_size]
            response = self.client.models.embed_content(
                model=self.model,
                contents=batch
            )
            for item in response.embeddings:
                embeddings_result.append(item.values)

        return embeddings_result


def get_chroma_client(persist_path: str = CHROMA_PERSIST_DIR) -> chromadb.PersistentClient:
    """
    Initializes and returns a ChromaDB PersistentClient.
    """
    os.makedirs(persist_path, exist_ok=True)
    return chromadb.PersistentClient(path=persist_path)


def get_or_create_collection(
    collection_name: str = CHROMA_COLLECTION_NAME,
    client: Optional[chromadb.PersistentClient] = None
):
    """
    Retrieves or creates a ChromaDB collection configured with the Gemini embedding function.
    """
    if client is None:
        client = get_chroma_client()

    embedding_fn = GeminiEmbeddingFunction()

    # Chroma collections retain embedding function for query and insert
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"description": "PGDCA and CS Admission Knowledge Base"}
    )
    return collection


def add_chunks_to_vector_store(chunks: List[Dict[str, Any]], collection=None) -> int:
    """
    Upserts chunk dictionaries to the Chroma collection.
    Each chunk dict is expected to have:
      - 'id': str
      - 'text': str
      - 'metadata': dict
    Returns the count of upserted items.
    """
    if not chunks:
        return 0

    if collection is None:
        collection = get_or_create_collection()

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # upsert handles both insertion and updating without duplicate errors
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    return len(chunks)


def query_vector_store(
    query_text: str,
    n_results: int = 5,
    collection=None
) -> List[Dict[str, Any]]:
    """
    Queries the vector store for documents similar to query_text.
    Returns a list of structured result items.
    """
    if collection is None:
        collection = get_or_create_collection()

    if collection.count() == 0:
        return []

    # Ensure n_results does not exceed total documents
    n_results = min(n_results, collection.count())

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )

    output = []
    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(docs)
        distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)
        ids = results["ids"][0] if "ids" in results and results["ids"] else [""] * len(docs)

        for doc_id, doc, meta, dist in zip(ids, docs, metas, distances):
            output.append({
                "id": doc_id,
                "text": doc,
                "metadata": meta,
                "distance": dist
            })

    return output


def get_collection_stats(collection=None) -> Dict[str, Any]:
    """
    Returns statistics about the current Chroma collection.
    """
    if collection is None:
        collection = get_or_create_collection()

    return {
        "name": collection.name,
        "count": collection.count()
    }