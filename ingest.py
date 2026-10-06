import os
import argparse
from typing import Optional
import chromadb
from utils.pdf_to_read import extract_pdf_pages, chunk_pages
<<<<<<< HEAD
from utils.chroma_dab_confing import (
=======
from utils.chroma_db_config import (
>>>>>>> 1671daa284be8d5f44e7f97c31b551b0711cc43c
    get_or_create_collection,
    add_chunks_to_vector_store,
    get_collection_stats,
    get_chroma_client,
    CHROMA_COLLECTION_NAME
)


def ingest_documents(data_dir: str = "data", reset: bool = False) -> int:
    """
    Scans the given data directory for PDF files, extracts and chunks text,
    and indexes them into the persistent ChromaDB collection.
    
    Returns the total number of chunks indexed.
    """
    if not os.path.exists(data_dir):
        print(f"[Ingest] Directory '{data_dir}' does not exist.")
        return 0

    client = get_chroma_client()

    if reset:
        print(f"[Ingest] Resetting collection '{CHROMA_COLLECTION_NAME}'...")
        try:
            client.delete_collection(name=CHROMA_COLLECTION_NAME)
        except Exception:
            pass

    collection = get_or_create_collection(client=client)

    pdf_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"[Ingest] No PDF files found in '{data_dir}'.")
        return 0

    print(f"[Ingest] Found {len(pdf_files)} PDF file(s) in '{data_dir}'.")

    all_chunks = []
    for pdf_path in pdf_files:
        # Skip empty files
        if os.path.getsize(pdf_path) == 0:
            print(f"[Ingest] Skipping empty file: {pdf_path}")
            continue

        try:
            print(f"[Ingest] Processing: {pdf_path}...")
            pages = extract_pdf_pages(pdf_path)
            chunks = chunk_pages(pages, chunk_size=700, chunk_overlap=150)
            print(f"         Extracted {len(pages)} page(s) -> Generated {len(chunks)} chunk(s).")
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"[Ingest] Error processing {pdf_path}: {e}")

    if not all_chunks:
        print("[Ingest] No chunks generated to index.")
        return 0

    print(f"[Ingest] Storing and embedding {len(all_chunks)} chunk(s) into ChromaDB...")
    indexed_count = add_chunks_to_vector_store(all_chunks, collection=collection)

    stats = get_collection_stats(collection=collection)
    print(f"[Ingest] Successfully indexed {indexed_count} chunk(s). Collection '{stats['name']}' now has {stats['count']} total chunks.")
    return indexed_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB Persistent Vector Store")
    parser.add_argument("--data-dir", default="data", help="Directory containing PDF files (default: data)")
    parser.add_argument("--reset", action="store_true", help="Reset collection before ingesting")
    args = parser.parse_args()

    ingest_documents(data_dir=args.data_dir, reset=args.reset)