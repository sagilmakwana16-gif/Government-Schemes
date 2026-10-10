import argparse
import os
from typing import Optional
import chromadb
from utils.chroma_db_config import (
    CHROMA_COLLECTION_NAME,
    add_chunks_to_vector_store,
    get_chroma_client,
    get_collection_stats,
    get_or_create_collection,
)
from utils.logger import get_logger
from utils.pdf_to_read import chunk_pages, extract_pdf_pages

logger = get_logger("ingest")


def ingest_documents(data_dir: str = "data", reset: bool = False) -> int:
    """Scans the given data directory for PDF files, extracts and chunks text,

    and indexes them into the persistent ChromaDB collection.
    """
    if not os.path.exists(data_dir):
        logger.warning("Ingest directory '%s' does not exist.", data_dir)
        return 0

    client = get_chroma_client()

    if reset:
        logger.info("Resetting collection '%s'...", CHROMA_COLLECTION_NAME)
        try:
            client.delete_collection(name=CHROMA_COLLECTION_NAME)
        except Exception as e:
            logger.warning("Could not delete collection during reset: %s", e)

    collection = get_or_create_collection(client=client)

    pdf_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        logger.warning("No PDF files found in '%s'.", data_dir)
        return 0

    logger.info("Found %d PDF file(s) in '%s'.", len(pdf_files), data_dir)

    all_chunks = []
    for pdf_path in pdf_files:
        if os.path.getsize(pdf_path) == 0:
            logger.info("Skipping empty file: %s", pdf_path)
            continue

        try:
            logger.info("Processing PDF: %s...", pdf_path)
            pages = extract_pdf_pages(pdf_path)
            chunks = chunk_pages(pages, chunk_size=700, chunk_overlap=150)
            logger.info(
                "Extracted %d page(s) -> Generated %d chunk(s) from %s.",
                len(pages),
                len(chunks),
                pdf_path,
            )
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error("Error processing PDF '%s': %s", pdf_path, e)

    if all_chunks:
        added_count = add_chunks_to_vector_store(
            all_chunks, collection=collection
        )
        logger.info(
            "Ingestion complete. Total chunks indexed: %d", added_count
        )
        return added_count

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into ChromaDB Vector Store"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to folder containing PDF documents",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset existing collection before indexing",
    )
    args = parser.parse_args()

    ingest_documents(data_dir=args.data_dir, reset=args.reset)