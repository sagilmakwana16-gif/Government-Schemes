from flask import Flask, render_template, jsonify, request

from ai_services.servam_api import generate_answer
from utils.pdf_to_read import pdf_to_text_extract
from utils.chroma_db_config import (
    query_vector_store,
    get_collection_stats,
    get_or_create_collection
)
from ingest import ingest_documents

app = Flask(__name__)
def init_knowledge_base():
    """
    Checks if ChromaDB vector store contains indexed chunks.
    If empty, automatically triggers PDF ingestion from the data/ directory.
    """
    try:
        col = get_or_create_collection()
        stats = get_collection_stats(col)
        if stats.get("count", 0) == 0:
            print("[Startup] Chroma collection is empty. Performing initial ingestion from 'data/'...")
            ingest_documents(data_dir="data")
        else:
            print(f"[Startup] Chroma knowledge base ready with {stats['count']} chunk(s).")
    except Exception as e:
        print(f"[Startup] Error verifying/initializing ChromaDB knowledge base: {e}")

# Initialize collection on application load
init_knowledge_base()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get_response', methods=['POST'])
def get_response():

    user_data = request.get_json()

    user_message = user_data.get('message', '')

    if not user_message:
        return jsonify({
              "reply": "Please provide a valid question or topic.",
              "sources": []
        })

    # Retrieve top relevant context chunks from ChromaDB PersistentClient
    chunks = query_vector_store(user_message, n_results=5)
     # Format deduplicated sources for the UI
    sources = []
    seen_sources = set()
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        source_name = meta.get("source", "Document")
        page_num = meta.get("page")
        label = f"{source_name} (Page {page_num})" if page_num else source_name
        if label not in seen_sources:
            seen_sources.add(label)
            sources.append(label)

     # Generate answer grounded in the retrieved chunks
    answer = generate_answer(user_message, chunks)



    return jsonify({
        "reply": answer,
        "sources": sources
    })

@app.route("/api/kb/stats", methods=["GET"])
def get_kb_stats():
    """
    Returns statistics about the current ChromaDB vector store.
    """
    try:
        stats = get_collection_stats()
        return jsonify({"status": "success", "stats": stats})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)