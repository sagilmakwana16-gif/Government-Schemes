import os
import re
from PyPDF2 import PdfReader


def extract_pdf_pages(pdf_path):
    """
    Extracts text page-by-page from a PDF file.
    Returns a list of dicts with text and metadata.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    file_name = os.path.basename(pdf_path)
    pages_data = []

    for idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        cleaned_text = re.sub(r"[ \t]+", " ", raw_text).strip()
        if cleaned_text:
            pages_data.append({
                "source": file_name,
                "path": str(pdf_path),
                "page_number": idx + 1,
                "text": cleaned_text
            })

    return pages_data


def chunk_pages(pages_data, chunk_size=700, chunk_overlap=150):
    """
    Splits page text into overlapping chunks, associating metadata with each chunk.
    Returns a list of chunk dictionaries:
    [
        {
            "id": "Python_pdf_p1_c0",
            "text": "...",
            "metadata": {"source": "Python.pdf", "page": 1, "chunk_index": 0}
        }, ...
    ]
    """
    chunks = []
    for page in pages_data:
        text = page["text"]
        source = page["source"]
        page_num = page["page_number"]

        safe_source = re.sub(r"[^a-zA-Z0-9_-]", "_", source)

        if len(text) <= chunk_size:
            chunk_id = f"{safe_source}_p{page_num}_c0"
            chunks.append({
                "id": chunk_id,
                "text": text,
                "metadata": {
                    "source": source,
                    "page": page_num,
                    "chunk_index": 0
                }
            })
            continue

        start = 0
        chunk_idx = 0
        while start < len(text):
            end = start + chunk_size
            chunk_slice = text[start:end]

            # Try to break at a sentence or newline boundary if possible
            if end < len(text):
                last_break = max(
                    chunk_slice.rfind(". "),
                    chunk_slice.rfind("\n"),
                    chunk_slice.rfind(" ")
                )
                if last_break > chunk_size // 2:
                    chunk_slice = chunk_slice[:last_break + 1]
                    end = start + last_break + 1

            chunk_text = chunk_slice.strip()
            if chunk_text:
                chunk_id = f"{safe_source}_p{page_num}_c{chunk_idx}"
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "source": source,
                        "page": page_num,
                        "chunk_index": chunk_idx
                    }
                })
                chunk_idx += 1

            start = end - chunk_overlap
            if start >= len(text) or end >= len(text):
                break

    return chunks


def pdf_to_text_extract(pdf_path):
    """
    Legacy helper: Extracts and concatenates text from all pages of the PDF.
    Maintained for backwards-compatibility.
    """
    pages = extract_pdf_pages(pdf_path)
    return "\n\n".join([f"--- Page {p['page_number']} ---\n{p['text']}" for p in pages])